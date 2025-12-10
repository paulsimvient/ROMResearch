"""Training utilities for ML-ROM models."""
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

from ..models.ml_rom_base import MLROMBase
from ..utils.data_loader import DataLoader
from ..utils.preprocessor import Preprocessor


class MLROMTrainer:
    """Trainer for ML-ROM models with utilities."""
    
    def __init__(
        self,
        model: MLROMBase,
        preprocessor: Optional[Preprocessor] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: ML-ROM model to train
            preprocessor: Optional data preprocessor
        """
        self.model = model
        self.preprocessor = preprocessor or Preprocessor(method="standard")
        self.training_history = None
    
    def prepare_data(
        self,
        snapshots: np.ndarray,
        val_ratio: float = 0.2,
        sequence_length: int = 10
    ) -> tuple:
        """
        Prepare training data.
        
        Args:
            snapshots: Snapshot data [n_features, n_samples] or [n_samples, n_features]
            val_ratio: Validation ratio
            sequence_length: Sequence length for sequence models
        
        Returns:
            train_data, val_data, preprocessor
        """
        # Normalize
        if snapshots.ndim == 2:
            if snapshots.shape[0] > snapshots.shape[1]:
                snapshots = snapshots.T
        
        self.preprocessor.fit(snapshots)
        normalized = self.preprocessor.transform(snapshots)
        
        # Split
        train_data, val_data = DataLoader.split_train_val(normalized, val_ratio)
        
        return train_data, val_data, self.preprocessor
    
    def train(
        self,
        snapshots: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        val_ratio: float = 0.2,
        sequence_length: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train ML-ROM model.
        
        Args:
            snapshots: Training snapshots
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            val_ratio: Validation ratio
            sequence_length: Sequence length
            **kwargs: Additional training parameters
        
        Returns:
            Training history
        """
        train_data, val_data, _ = self.prepare_data(
            snapshots, val_ratio, sequence_length
        )
        
        history = self.model.train(
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seq_len=sequence_length,
            **kwargs
        )
        
        self.training_history = history
        return history
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history."""
        if self.training_history is None:
            raise RuntimeError("No training history available. Train model first.")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        if "train_losses" in self.training_history:
            axes[0].plot(self.training_history["train_losses"], label="Train")
            if "val_losses" in self.training_history:
                axes[0].plot(self.training_history["val_losses"], label="Validation")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Training Loss")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot GAN losses if available
        if "generator_losses" in self.training_history:
            axes[1].plot(self.training_history["generator_losses"], label="Generator")
            axes[1].plot(self.training_history["discriminator_losses"], label="Discriminator")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].set_title("GAN Losses")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        else:
            plt.show()
    
    def evaluate(
        self,
        test_snapshots: np.ndarray,
        n_steps: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            test_snapshots: Test snapshots
            n_steps: Number of prediction steps
        
        Returns:
            Evaluation metrics
        """
        if not self.model.is_trained:
            raise RuntimeError("Model not trained.")
        
        # Normalize test data
        if test_snapshots.ndim == 2:
            if test_snapshots.shape[0] > test_snapshots.shape[1]:
                test_snapshots = test_snapshots.T
        
        normalized_test = self.preprocessor.transform(test_snapshots)
        
        # Predict
        initial_state = normalized_test[0]
        predictions = self.model.predict(initial_state, n_steps)
        
        # Inverse transform
        if predictions.ndim == 2:
            predictions = predictions.T
        
        predictions_original = self.preprocessor.inverse_transform(predictions.T)
        ground_truth = test_snapshots[:n_steps+1]
        
        # Compute metrics
        mse = np.mean((predictions_original - ground_truth)**2)
        relative_error = np.linalg.norm(predictions_original - ground_truth) / np.linalg.norm(ground_truth)
        
        return {
            "mse": float(mse),
            "relative_error": float(relative_error),
            "predictions": predictions_original.tolist(),
            "ground_truth": ground_truth.tolist()
        }
