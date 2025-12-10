"""LSTM ROM: Uses LSTM networks for sequential state prediction."""
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


class LSTMROM(MLROMBase):
    """LSTM Reduced Order Model.
    
    Uses LSTM networks to learn sequential dynamics.
    """
    
    def __init__(
        self,
        model_id: str,
        name: str,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        device: Optional[str] = None
    ):
        """
        Initialize LSTM ROM.
        
        Args:
            model_id: Unique identifier
            name: Human-readable name
            input_dim: Full state dimension
            latent_dim: Latent/reduced dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            device: PyTorch device
        """
        super().__init__(model_id, name, input_dim, latent_dim, device)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = self._build_model()
        self.model.to(self.device)
    
    def _build_model(self) -> nn.Module:
        """Build LSTM model with encoder-decoder architecture."""
        class LSTMEncoderDecoder(nn.Module):
            def __init__(self, input_dim, latent_dim, hidden_dim, num_layers):
                super().__init__()
                # Encoder: input_dim -> latent_dim
                self.encoder = nn.LSTM(
                    input_dim, hidden_dim, num_layers, batch_first=True
                )
                self.encoder_proj = nn.Linear(hidden_dim, latent_dim)
                
                # Decoder: latent_dim -> input_dim
                self.decoder = nn.LSTM(
                    latent_dim, hidden_dim, num_layers, batch_first=True
                )
                self.decoder_proj = nn.Linear(hidden_dim, input_dim)
            
            def forward(self, x, future_steps=1):
                # Encode
                enc_out, (h_n, c_n) = self.encoder(x)
                latent = self.encoder_proj(enc_out[:, -1, :])  # Last timestep
                
                # Decode
                latent_seq = latent.unsqueeze(1).repeat(1, future_steps, 1)
                dec_out, _ = self.decoder(latent_seq, (h_n, c_n))
                output = self.decoder_proj(dec_out)
                
                return output
        
        return LSTMEncoderDecoder(
            self.input_dim, self.latent_dim, self.hidden_dim, self.num_layers
        )
    
    def train(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        seq_len: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Train LSTM ROM."""
        # Prepare sequences
        if train_data.ndim == 2:
            # [n_samples, input_dim] - create sliding windows
            sequences = []
            for i in range(len(train_data) - seq_len):
                sequences.append(train_data[i:i+seq_len+1])
            train_data = np.array(sequences)
        
        n_samples, seq_len_full, _ = train_data.shape
        X = train_data[:, :-1, :]  # [n_samples, seq_len, input_dim]
        Y = train_data[:, 1:, :]   # [n_samples, seq_len, input_dim]
        
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
                
                pred = self.model(x_batch, future_steps=y_batch.shape[1])
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
                    val_pred = self.model(val_x, future_steps=val_y.shape[1])
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
        """Predict future states using LSTM."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        if initial_state.ndim == 1:
            initial_state = initial_state.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = initial_state.shape[0]
        
        with torch.no_grad():
            # Use initial state as input sequence
            x_input = torch.FloatTensor(initial_state).unsqueeze(1).to(self.device)
            pred = self.model(x_input, future_steps=n_steps)
            trajectory = pred.cpu().numpy()
        
        # Reshape: [batch, n_steps, input_dim] -> [batch, input_dim, n_steps]
        trajectory = np.transpose(trajectory, (0, 2, 1))
        
        if squeeze_output:
            trajectory = trajectory[0]
        
        return trajectory
