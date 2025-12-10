"""Data loading utilities for ML-ROM training."""
from typing import Tuple, Optional
import numpy as np


class DataLoader:
    """Data loader for ML-ROM training data."""
    
    @staticmethod
    def load_from_snapshots(
        snapshots: np.ndarray,
        sequence_length: int = 10,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load sequences from snapshot data.
        
        Args:
            snapshots: Snapshot array [n_features, n_samples] or [n_samples, n_features]
            sequence_length: Length of input sequences
            stride: Stride for sliding window
        
        Returns:
            X: Input sequences [n_sequences, sequence_length, n_features]
            Y: Target sequences [n_sequences, sequence_length, n_features]
        """
        if snapshots.ndim == 2:
            if snapshots.shape[0] > snapshots.shape[1]:
                # [n_features, n_samples] -> [n_samples, n_features]
                snapshots = snapshots.T
        
        n_samples, n_features = snapshots.shape
        
        sequences = []
        targets = []
        
        for i in range(0, n_samples - sequence_length, stride):
            seq = snapshots[i:i+sequence_length]
            target = snapshots[i+1:i+sequence_length+1]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    @staticmethod
    def split_train_val(
        data: np.ndarray,
        val_ratio: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets.
        
        Args:
            data: Data array
            val_ratio: Validation set ratio
        
        Returns:
            train_data, val_data
        """
        n_samples = len(data)
        n_val = int(n_samples * val_ratio)
        
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        return data[train_indices], data[val_indices]
    
    @staticmethod
    def normalize(
        data: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize data to zero mean and unit variance.
        
        Args:
            data: Data array [n_samples, n_features]
            mean: Pre-computed mean (optional)
            std: Pre-computed std (optional)
        
        Returns:
            normalized_data, mean, std
        """
        if data.ndim == 2:
            if mean is None:
                mean = np.mean(data, axis=0, keepdims=True)
            if std is None:
                std = np.std(data, axis=0, keepdims=True)
                std = np.where(std == 0, 1.0, std)
            
            normalized = (data - mean) / std
            return normalized, mean, std
        else:
            raise ValueError("Data must be 2D array")
