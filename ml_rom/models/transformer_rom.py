"""Transformer ROM: Uses Transformer architecture for sequence modeling."""
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

from .ml_rom_base import MLROMBase


class TransformerROM(MLROMBase):
    """Transformer Reduced Order Model.
    
    Uses Transformer architecture for sequential state prediction.
    """
    
    def __init__(
        self,
        model_id: str,
        name: str,
        input_dim: int,
        latent_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize Transformer ROM.
        
        Args:
            model_id: Unique identifier
            name: Human-readable name
            input_dim: Full state dimension
            latent_dim: Latent/reduced dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            device: PyTorch device
        """
        super().__init__(model_id, name, input_dim, latent_dim, device)
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.model = self._build_model()
        self.model.to(self.device)
    
    def _build_model(self) -> nn.Module:
        """Build Transformer model."""
        class TransformerModel(nn.Module):
            def __init__(self, input_dim, latent_dim, d_model, nhead, num_layers, dim_feedforward):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, d_model)
                self.output_proj = nn.Linear(d_model, input_dim)
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            def forward(self, x, mask=None):
                x = self.input_proj(x)
                x = self.transformer(x, mask=mask)
                x = self.output_proj(x)
                return x
        
        return TransformerModel(
            self.input_dim, self.latent_dim, self.d_model,
            self.nhead, self.num_layers, self.dim_feedforward
        )
    
    def train(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        seq_len: int = 20,
        **kwargs
    ) -> Dict[str, Any]:
        """Train Transformer ROM."""
        if train_data.ndim == 2:
            sequences = []
            for i in range(len(train_data) - seq_len):
                sequences.append(train_data[i:i+seq_len+1])
            train_data = np.array(sequences)
        
        X = train_data[:, :-1, :]
        Y = train_data[:, 1:, :]
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            indices = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                batch_idx = indices[i:i+batch_size]
                x_batch = torch.FloatTensor(X[batch_idx]).to(self.device)
                y_batch = torch.FloatTensor(Y[batch_idx]).to(self.device)
                
                pred = self.model(x_batch)
                loss = criterion(pred, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            train_losses.append(avg_loss)
            
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    if val_data.ndim == 2:
                        val_sequences = []
                        for i in range(len(val_data) - seq_len):
                            val_sequences.append(val_data[i:i+seq_len+1])
                        val_data = np.array(val_sequences)
                    
                    val_X = val_data[:, :-1, :]
                    val_Y = val_data[:, 1:, :]
                    val_x = torch.FloatTensor(val_X).to(self.device)
                    val_y = torch.FloatTensor(val_Y).to(self.device)
                    val_pred = self.model(val_x)
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
        """Predict future states using Transformer."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        if initial_state.ndim == 1:
            initial_state = initial_state.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        with torch.no_grad():
            current = torch.FloatTensor(initial_state).unsqueeze(1).to(self.device)
            trajectory = [initial_state]
            
            for _ in range(n_steps):
                pred = self.model(current)
                next_state = pred[:, -1:, :]
                trajectory.append(next_state.cpu().numpy()[:, 0, :])
                current = torch.cat([current, next_state], dim=1)
        
        trajectory = np.array(trajectory)
        trajectory = np.transpose(trajectory, (2, 0, 1)) if trajectory.ndim == 3 else trajectory
        
        if squeeze_output:
            trajectory = trajectory[:, :, 0] if trajectory.ndim == 3 else trajectory
        
        return trajectory
