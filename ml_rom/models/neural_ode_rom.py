"""Neural ODE ROM: Uses neural networks to learn continuous-time dynamics."""
from typing import Dict, Any, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    try:
        from torchdiffeq import odeint
        TORCHDIFFEQ_AVAILABLE = True
    except ImportError:
        TORCHDIFFEQ_AVAILABLE = False
        odeint = None
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TORCHDIFFEQ_AVAILABLE = False
    torch = None
    nn = None
    odeint = None

from .ml_rom_base import MLROMBase


if TORCH_AVAILABLE and nn is not None:
    class ODEFunc(nn.Module):
    """Neural ODE function: dx/dt = f(x, t)."""
    
    def __init__(self, dim: int, hidden_dims: list = [64, 64]):
        super().__init__()
        layers = []
        input_dim = dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, t, x):
        return self.net(x)
else:
    class ODEFunc:
        pass


class NeuralODEROM(MLROMBase):
    """Neural ODE Reduced Order Model.
    
    Learns continuous-time dynamics using neural ODEs.
    """
    
    def __init__(
        self,
        model_id: str,
        name: str,
        input_dim: int,
        latent_dim: Optional[int] = None,
        hidden_dims: list = [64, 64],
        device: Optional[str] = None
    ):
        """
        Initialize Neural ODE ROM.
        
        Args:
            model_id: Unique identifier
            name: Human-readable name
            input_dim: Full state dimension
            latent_dim: Latent dimension (if None, uses input_dim)
            hidden_dims: Hidden layer dimensions for ODE function
            device: PyTorch device
        """
        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError(
                "torchdiffeq is required. Install with: pip install torchdiffeq"
            )
        
        if latent_dim is None:
            latent_dim = input_dim
        
        super().__init__(model_id, name, input_dim, latent_dim, device)
        self.hidden_dims = hidden_dims
        self.model = self._build_model()
        self.model.to(self.device)
    
    def _build_model(self) -> nn.Module:
        """Build Neural ODE model."""
        return ODEFunc(self.input_dim, self.hidden_dims)
    
    def train(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        **kwargs
    ) -> Dict[str, Any]:
        """Train Neural ODE ROM."""
        if train_data.ndim == 2:
            # [n_samples, input_dim] - convert to sequences
            train_data = train_data.reshape(-1, 1, self.input_dim)
        
        n_samples, seq_len, _ = train_data.shape
        
        # Prepare training pairs (x_t, x_{t+1})
        X = train_data[:, :-1, :].reshape(-1, self.input_dim)
        Y = train_data[:, 1:, :].reshape(-1, self.input_dim)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            # Mini-batch training
            indices = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                batch_idx = indices[i:i+batch_size]
                x_batch = torch.FloatTensor(X[batch_idx]).to(self.device)
                y_batch = torch.FloatTensor(Y[batch_idx]).to(self.device)
                
                # Integrate ODE from t=0 to t=1
                t = torch.tensor([0.0, 1.0]).to(self.device)
                pred = odeint(self.model, x_batch, t)[-1]
                
                loss = criterion(pred, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            train_losses.append(avg_loss)
            
            # Validation
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    if val_data.ndim == 2:
                        val_data = val_data.reshape(-1, 1, self.input_dim)
                    val_X = val_data[:, :-1, :].reshape(-1, self.input_dim)
                    val_Y = val_data[:, 1:, :].reshape(-1, self.input_dim)
                    
                    val_x = torch.FloatTensor(val_X).to(self.device)
                    val_y = torch.FloatTensor(val_Y).to(self.device)
                    t = torch.tensor([0.0, 1.0]).to(self.device)
                    val_pred = odeint(self.model, val_x, t)[-1]
                    val_loss = criterion(val_pred, val_y).item()
                    val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        self.is_trained = True
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
        }
    
    def predict(self, initial_state: np.ndarray, n_steps: int) -> np.ndarray:
        """Predict future states using Neural ODE."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        # Handle batch dimension
        if initial_state.ndim == 1:
            initial_state = initial_state.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = initial_state.shape[0]
        
        with torch.no_grad():
            x0 = torch.FloatTensor(initial_state).to(self.device)
            t = torch.linspace(0, n_steps, n_steps + 1).to(self.device)
            
            # Integrate ODE
            trajectory = odeint(self.model, x0, t)
            trajectory = trajectory.cpu().numpy()
        
        # Reshape: [n_steps+1, batch, input_dim] -> [batch, input_dim, n_steps+1]
        trajectory = np.transpose(trajectory, (1, 2, 0))
        
        if squeeze_output:
            trajectory = trajectory[0]  # [input_dim, n_steps+1]
        
        return trajectory
