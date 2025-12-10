"""VAE ROM: Variational Autoencoder for ROM with latent dynamics."""
from typing import Dict, Any, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from .ml_rom_base import MLROMBase


class VAEEncoder(nn.Module):
    """VAE Encoder."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list = [128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x):
        h = self.features(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """VAE Decoder."""
    
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: list = [64, 128]):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.decoder(z)


class LatentDynamics(nn.Module):
    """Latent space dynamics model."""
    
    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, z):
        return self.net(z)


class VAEROM(MLROMBase):
    """Variational Autoencoder Reduced Order Model.
    
    Uses VAE for dimensionality reduction with learned latent dynamics.
    """
    
    def __init__(
        self,
        model_id: str,
        name: str,
        input_dim: int,
        latent_dim: int,
        encoder_dims: list = [128, 64],
        decoder_dims: list = [64, 128],
        dynamics_hidden: int = 64,
        device: Optional[str] = None
    ):
        """
        Initialize VAE ROM.
        
        Args:
            model_id: Unique identifier
            name: Human-readable name
            input_dim: Full state dimension
            latent_dim: Latent dimension
            encoder_dims: Encoder hidden dimensions
            decoder_dims: Decoder hidden dimensions
            dynamics_hidden: Latent dynamics hidden dimension
            device: PyTorch device
        """
        super().__init__(model_id, name, input_dim, latent_dim, device)
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.dynamics_hidden = dynamics_hidden
        self.model = self._build_model()
        self.model.to(self.device)
    
    def _build_model(self) -> nn.Module:
        """Build VAE model with encoder, decoder, and latent dynamics."""
        class VAEModel(nn.Module):
            def __init__(self, encoder, decoder, dynamics):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.dynamics = dynamics
            
            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            
            def forward(self, x):
                mu, logvar = self.encoder(x)
                z = self.reparameterize(mu, logvar)
                recon = self.decoder(z)
                return recon, mu, logvar, z
        
        encoder = VAEEncoder(self.input_dim, self.latent_dim, self.encoder_dims)
        decoder = VAEDecoder(self.latent_dim, self.input_dim, self.decoder_dims)
        dynamics = LatentDynamics(self.latent_dim, self.dynamics_hidden)
        
        return VAEModel(encoder, decoder, dynamics)
    
    def train(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        beta: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Train VAE ROM."""
        if train_data.ndim == 3:
            train_data = train_data.reshape(-1, train_data.shape[-1])
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            indices = np.random.permutation(len(train_data))
            for i in range(0, len(train_data), batch_size):
                batch_idx = indices[i:i+batch_size]
                x_batch = torch.FloatTensor(train_data[batch_idx]).to(self.device)
                
                recon, mu, logvar, z = self.model(x_batch)
                
                # Reconstruction loss
                recon_loss = F.mse_loss(recon, x_batch, reduction='sum')
                
                # KL divergence
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                # Latent dynamics loss (if we have pairs)
                if len(batch_idx) > 1:
                    z_next = self.model.dynamics(z)
                    z_next_pred = z[1:] if len(z) > 1 else z
                    dynamics_loss = F.mse_loss(z_next[:-1], z_next_pred) if len(z) > 1 else torch.tensor(0.0)
                else:
                    dynamics_loss = torch.tensor(0.0)
                
                loss = recon_loss + beta * kl_loss + 0.1 * dynamics_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            train_losses.append(avg_loss)
            
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    if val_data.ndim == 3:
                        val_data = val_data.reshape(-1, val_data.shape[-1])
                    
                    val_x = torch.FloatTensor(val_data).to(self.device)
                    recon, mu, logvar, _ = self.model(val_x)
                    val_loss = F.mse_loss(recon, val_x).item()
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
        """Predict future states using VAE ROM."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        if initial_state.ndim == 1:
            initial_state = initial_state.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        with torch.no_grad():
            x0 = torch.FloatTensor(initial_state).to(self.device)
            mu, logvar = self.model.encoder(x0)
            z = mu  # Use mean for prediction
            
            trajectory = [initial_state]
            z_current = z
            
            for _ in range(n_steps):
                z_next = self.model.dynamics(z_current)
                x_next = self.model.decoder(z_next)
                trajectory.append(x_next.cpu().numpy())
                z_current = z_next
        
        trajectory = np.array(trajectory)
        trajectory = np.transpose(trajectory, (2, 0, 1)) if trajectory.ndim == 3 else trajectory
        
        if squeeze_output:
            trajectory = trajectory[:, :, 0] if trajectory.ndim == 3 else trajectory
        
        return trajectory
