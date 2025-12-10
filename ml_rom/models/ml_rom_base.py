"""Base class for Machine Learning Reduced Order Models."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class MLROMBase(ABC):
    """Base class for all ML-based ROM implementations."""
    
    def __init__(
        self,
        model_id: str,
        name: str,
        input_dim: int,
        latent_dim: int,
        device: Optional[str] = None
    ):
        """
        Initialize ML-ROM base.
        
        Args:
            model_id: Unique identifier
            name: Human-readable name
            input_dim: Full state dimension
            latent_dim: Latent/reduced dimension
            device: PyTorch device ('cpu', 'cuda', or None for auto)
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for ML-ROM. Install with: pip install torch"
            )
        
        self.model_id = model_id
        self.name = name
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Build the neural network model architecture."""
        pass
    
    @abstractmethod
    def train(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the ML-ROM model.
        
        Args:
            train_data: Training snapshots [n_samples, input_dim] or [n_samples, seq_len, input_dim]
            val_data: Validation snapshots (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            **kwargs: Additional training parameters
        
        Returns:
            Training history dictionary
        """
        pass
    
    @abstractmethod
    def predict(self, initial_state: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Predict future states.
        
        Args:
            initial_state: Initial condition [input_dim] or [batch, input_dim]
            n_steps: Number of prediction steps
        
        Returns:
            Predicted states [input_dim, n_steps] or [batch, input_dim, n_steps]
        """
        pass
    
    def save(self, path: str) -> None:
        """Save model to file."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'model_id': self.model_id,
            'name': self.name,
            'is_trained': self.is_trained,
        }, path)
    
    def load(self, path: str) -> None:
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = checkpoint['input_dim']
        self.latent_dim = checkpoint['latent_dim']
        self.model_id = checkpoint['model_id']
        self.name = checkpoint['name']
        self.is_trained = checkpoint['is_trained']
        
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to model device."""
        return tensor.to(self.device)
    
    def metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "id": self.model_id,
            "name": self.name,
            "type": "ML-ROM",
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "device": str(self.device),
            "is_trained": self.is_trained,
            "pytorch_available": TORCH_AVAILABLE,
        }
