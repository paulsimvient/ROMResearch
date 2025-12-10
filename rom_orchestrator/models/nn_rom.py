"""Neural Network Reduced-Order Model implementation."""
from typing import Dict, Any, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from .base import ModelAdapter


class NeuralNetworkROM(ModelAdapter):
    """Neural Network Reduced-Order Model."""
    
    def __init__(
        self,
        model_id: str,
        name: str,
        model: Optional[Any] = None,
        input_shape: Optional[tuple] = None,
        output_shape: Optional[tuple] = None
    ):
        """
        Initialize Neural Network ROM.
        
        Args:
            model_id: Unique identifier
            name: Human-readable name
            model: Pre-trained PyTorch model (optional)
            input_shape: Expected input shape
            output_shape: Expected output shape
        """
        super().__init__(model_id, name, "ROM")
        self.model = model
        self._input_shape = input_shape
        self._output_shape = output_shape
        
        if model is not None and TORCH_AVAILABLE:
            self.model.eval()  # Set to evaluation mode
    
    def fit(self, data: np.ndarray, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001) -> None:
        """
        Train neural network ROM on snapshot data.
        
        Args:
            data: Snapshot matrix [n_features, n_samples] or [n_samples, n_features]
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Neural Network ROM. Install with: pip install torch")
        
        # Ensure data is [n_samples, n_features]
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Use input_shape to determine if we need to transpose
        # Orchestrator passes data as [n_samples, n_features] after transposing
        # But if input_shape is set, verify the format matches
        if data.ndim == 2:
            if self._input_shape and len(self._input_shape) > 0:
                expected_features = self._input_shape[0]
                # If first dimension matches expected features, it's [n_features, n_samples] - transpose
                if data.shape[0] == expected_features and data.shape[0] != data.shape[1]:
                    data = data.T
            # Otherwise assume orchestrator already transposed correctly
        
        n_samples, n_features = data.shape
        
        # Verify dimensions match input_shape if set
        if self._input_shape and len(self._input_shape) > 0:
            if n_features != self._input_shape[0]:
                raise ValueError(
                    f"Feature dimension mismatch: data has {n_features} features, "
                    f"but ROM expects {self._input_shape[0]} features. "
                    f"Data shape: {data.shape}"
                )
        
        # Create model if not provided
        if self.model is None:
            # Ensure reasonable hidden dimension (at least 32, but not too large)
            hidden_dim = max(32, min(128, n_features * 2))
            self.model = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_features)
            )
        
        # Set to training mode
        self.model.train()
        
        # Convert data to tensor
        data_tensor = torch.from_numpy(data).float()
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop - learn to predict next state from current state
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(n_samples - 1)
            
            total_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples - 1, batch_size):
                batch_indices = indices[i:min(i+batch_size, n_samples-1)]
                batch_current = data_tensor[batch_indices]
                batch_next = data_tensor[batch_indices + 1]
                
                optimizer.zero_grad()
                
                # Predict next state
                pred_next = self.model(batch_current)
                
                # Loss
                loss = criterion(pred_next, batch_next)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
        
        # Set to evaluation mode
        self.model.eval()
        
        self._input_shape = (n_features,)
        self._output_shape = (n_features,)
    
    def simulate(self, input_params: Dict[str, Any]) -> np.ndarray:
        """
        Run Neural Network ROM simulation.
        
        Args:
            input_params: Dictionary with:
                - "x0": Initial condition [n_features] or [batch, n_features]
                - "t": Time array (optional)
                - "n_steps": Number of time steps (if t not provided)
        
        Returns:
            Time series [n_features, n_steps] or [batch, n_features, n_steps]
        """
        if self.model is None:
            raise RuntimeError("Neural network model not loaded")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Neural Network ROM. Install with: pip install torch")
        
        # Get initial condition
        x0 = input_params.get("x0")
        if x0 is None:
            raise ValueError("Neural Network simulation requires 'x0' initial condition")
        x0 = np.array(x0)
        
        # Get time array
        t = input_params.get("t")
        if t is None:
            n_steps = input_params.get("n_steps", 100)
            t = np.arange(n_steps)
        else:
            t = np.array(t)
            n_steps = len(t)
        
        # Ensure x0 is 2D [batch, features] or 1D [features]
        if x0.ndim == 1:
            x0 = x0.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, n_features = x0.shape
        
        # Convert to torch tensor
        x_torch = torch.from_numpy(x0).float()
        
        # Run simulation
        results = []
        x_current = x_torch
        
        with torch.no_grad():
            for i in range(n_steps):
                # Forward pass
                if hasattr(self.model, '__call__'):
                    x_next = self.model(x_current)
                else:
                    raise AttributeError("Model must be callable")
                
                results.append(x_next.cpu().numpy())
                x_current = x_next
        
        # Stack results: [n_steps, batch, features] -> [batch, features, n_steps]
        result_array = np.stack(results, axis=0)  # [n_steps, batch, features]
        result_array = np.transpose(result_array, (1, 2, 0))  # [batch, features, n_steps]
        
        if squeeze_output:
            result_array = result_array[0]  # [features, n_steps]
        
        return result_array
    
    def metadata(self) -> Dict[str, Any]:
        """Get Neural Network ROM metadata."""
        return {
            "id": self.model_id,
            "name": self.name,
            "type": self.model_type,
            "method": "NeuralNetwork",
            "input_shape": self._input_shape,
            "output_shape": self._output_shape,
            "pytorch_available": TORCH_AVAILABLE,
        }

