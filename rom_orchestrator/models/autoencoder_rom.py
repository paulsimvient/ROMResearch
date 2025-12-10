"""Autoencoder-based Reduced-Order Model (Encoder → Latent Dynamics → Decoder)."""
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


class AutoencoderROM(ModelAdapter):
    """Autoencoder-based ROM: Encoder → Latent Dynamics → Decoder."""
    
    def __init__(
        self,
        model_id: str,
        name: str,
        encoder: Optional[Any] = None,
        decoder: Optional[Any] = None,
        latent_dynamics: Optional[Any] = None,
        latent_dim: Optional[int] = None,
        input_shape: Optional[tuple] = None,
        output_shape: Optional[tuple] = None
    ):
        """
        Initialize Autoencoder ROM.
        
        Args:
            model_id: Unique identifier
            name: Human-readable name
            encoder: PyTorch encoder model (maps full state → latent)
            decoder: PyTorch decoder model (maps latent → full state)
            latent_dynamics: PyTorch model for latent space dynamics (optional)
            latent_dim: Dimension of latent space
            input_shape: Expected input shape
            output_shape: Expected output shape
        """
        super().__init__(model_id, name, "ROM")
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dynamics = latent_dynamics
        self.latent_dim = latent_dim
        self._input_shape = input_shape
        self._output_shape = output_shape
        
        if encoder is not None and TORCH_AVAILABLE:
            self.encoder.eval()
        if decoder is not None and TORCH_AVAILABLE:
            self.decoder.eval()
        if latent_dynamics is not None and TORCH_AVAILABLE:
            self.latent_dynamics.eval()
    
    def fit(self, data: np.ndarray, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001) -> None:
        """
        Train autoencoder on snapshot data.
        
        Args:
            data: Snapshot matrix [n_features, n_samples] or [n_samples, n_features]
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Autoencoder ROM. Install with: pip install torch")
        
        # Ensure data is [n_samples, n_features]
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Use input_shape to determine format - orchestrator should pass [n_samples, n_features]
        # But check if we need to transpose based on input_shape
        if data.ndim == 2 and self._input_shape and len(self._input_shape) > 0:
            expected_features = self._input_shape[0]
            # If first dimension matches expected features count, it's [n_features, n_samples] - transpose
            if data.shape[0] == expected_features and data.shape[0] != data.shape[1]:
                data = data.T
        
        n_samples, n_features = data.shape
        
        # Verify dimensions match input_shape if set
        if self._input_shape and len(self._input_shape) > 0:
            if n_features != self._input_shape[0]:
                raise ValueError(
                    f"Feature dimension mismatch: data has {n_features} features, "
                    f"but ROM expects {self._input_shape[0]} features. "
                    f"Data shape: {data.shape}, input_shape: {self._input_shape}"
                )
        
        # Set latent dimension if not set
        if self.latent_dim is None:
            self.latent_dim = min(20, n_features // 4)
        
        # Create encoder and decoder if not provided
        if self.encoder is None:
            hidden_dim = max(32, n_features // 2)  # Ensure minimum hidden dimension
            self.encoder = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.latent_dim)
            )
        
        if self.decoder is None:
            hidden_dim = max(32, n_features // 2)  # Ensure minimum hidden dimension
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_features)
            )
        
        # Create latent dynamics model (simple linear)
        if self.latent_dynamics is None:
            self.latent_dynamics = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.Tanh(),
                nn.Linear(self.latent_dim, self.latent_dim)
            )
        
        # Set to training mode
        self.encoder.train()
        self.decoder.train()
        self.latent_dynamics.train()
        
        # Convert data to tensor
        data_tensor = torch.from_numpy(data).float()
        
        # Optimizer
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.latent_dynamics.parameters()),
            lr=learning_rate
        )
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(n_samples)
            data_shuffled = data_tensor[indices]
            
            total_loss = 0.0
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            for i in range(0, n_samples, batch_size):
                batch = data_shuffled[i:i+batch_size]
                
                optimizer.zero_grad()
                
                # Encode
                z = self.encoder(batch)
                
                # Decode
                x_recon = self.decoder(z)
                
                # Reconstruction loss
                recon_loss = criterion(x_recon, batch)
                
                # Latent dynamics loss (predict next state from current)
                if i + batch_size < n_samples:
                    batch_next = data_shuffled[i+1:i+batch_size+1]
                    z_next_true = self.encoder(batch_next)
                    z_next_pred = self.latent_dynamics(z)
                    dynamics_loss = criterion(z_next_pred, z_next_true.detach())
                else:
                    dynamics_loss = torch.tensor(0.0)
                
                # Total loss
                loss = recon_loss + 0.1 * dynamics_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        # Set to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        self.latent_dynamics.eval()
        
        self._input_shape = (n_features,)
        self._output_shape = (n_features,)
    
    def simulate(self, input_params: Dict[str, Any]) -> np.ndarray:
        """
        Run Autoencoder ROM simulation.
        
        Args:
            input_params: Dictionary with:
                - "x0": Initial condition [n_features] or [batch, n_features]
                - "t": Time array (optional)
                - "n_steps": Number of time steps (if t not provided)
        
        Returns:
            Time series [n_features, n_steps] or [batch, n_features, n_steps]
        """
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("Encoder and decoder must be provided")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Autoencoder ROM. Install with: pip install torch")
        
        # Get initial condition
        x0 = input_params.get("x0")
        if x0 is None:
            raise ValueError("Autoencoder simulation requires 'x0' initial condition")
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
        
        # Encode to latent space
        with torch.no_grad():
            z0 = self.encoder(x_torch)  # [batch, latent_dim]
        
        # Run simulation in latent space
        results = []
        z_current = z0
        
        with torch.no_grad():
            for i in range(n_steps):
                # Evolve in latent space
                if self.latent_dynamics is not None:
                    # Use learned latent dynamics model
                    z_next = self.latent_dynamics(z_current)
                else:
                    # Simple identity (no dynamics) - just decode
                    z_next = z_current
                
                # Decode back to full space
                x_reconstructed = self.decoder(z_next)
                results.append(x_reconstructed.cpu().numpy())
                
                # Update for next step
                z_current = z_next
        
        # Stack results: [n_steps, batch, features] -> [batch, features, n_steps]
        result_array = np.stack(results, axis=0)  # [n_steps, batch, features]
        result_array = np.transpose(result_array, (1, 2, 0))  # [batch, features, n_steps]
        
        if squeeze_output:
            result_array = result_array[0]  # [features, n_steps]
        
        return result_array
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode full state to latent space.
        
        Args:
            x: Full state [n_features] or [batch, n_features]
        
        Returns:
            Latent representation [latent_dim] or [batch, latent_dim]
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not available")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        x_torch = torch.from_numpy(np.array(x)).float()
        if x_torch.ndim == 1:
            x_torch = x_torch.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        with torch.no_grad():
            z = self.encoder(x_torch)
        
        z_np = z.cpu().numpy()
        if squeeze:
            z_np = z_np[0]
        
        return z_np
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent representation to full state.
        
        Args:
            z: Latent representation [latent_dim] or [batch, latent_dim]
        
        Returns:
            Full state [n_features] or [batch, n_features]
        """
        if self.decoder is None:
            raise RuntimeError("Decoder not available")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        z_torch = torch.from_numpy(np.array(z)).float()
        if z_torch.ndim == 1:
            z_torch = z_torch.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        with torch.no_grad():
            x = self.decoder(z_torch)
        
        x_np = x.cpu().numpy()
        if squeeze:
            x_np = x_np[0]
        
        return x_np
    
    def metadata(self) -> Dict[str, Any]:
        """Get Autoencoder ROM metadata."""
        return {
            "id": self.model_id,
            "name": self.name,
            "type": self.model_type,
            "method": "Autoencoder",
            "latent_dim": self.latent_dim,
            "input_shape": self._input_shape,
            "output_shape": self._output_shape,
            "has_latent_dynamics": self.latent_dynamics is not None,
            "pytorch_available": TORCH_AVAILABLE,
        }

