"""GAN ROM: Generative Adversarial Network for ROM."""
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


class Generator(nn.Module):
    """GAN Generator for state prediction."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list = [128, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, input_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """GAN Discriminator."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class GANROM(MLROMBase):
    """Generative Adversarial Network Reduced Order Model.
    
    Uses GAN for learning state distributions and dynamics.
    """
    
    def __init__(
        self,
        model_id: str,
        name: str,
        input_dim: int,
        latent_dim: int,
        generator_dims: list = [128, 256, 128],
        discriminator_dims: list = [128, 64],
        device: Optional[str] = None
    ):
        """
        Initialize GAN ROM.
        
        Args:
            model_id: Unique identifier
            name: Human-readable name
            input_dim: Full state dimension
            latent_dim: Latent dimension
            generator_dims: Generator hidden dimensions
            discriminator_dims: Discriminator hidden dimensions
            device: PyTorch device
        """
        super().__init__(model_id, name, input_dim, latent_dim, device)
        self.generator_dims = generator_dims
        self.discriminator_dims = discriminator_dims
        self.model = self._build_model()
        self.model.to(self.device)
    
    def _build_model(self) -> nn.Module:
        """Build GAN model."""
        class GANModel(nn.Module):
            def __init__(self, generator, discriminator):
                super().__init__()
                self.generator = generator
                self.discriminator = discriminator
        
        generator = Generator(self.input_dim, self.latent_dim, self.generator_dims)
        discriminator = Discriminator(self.input_dim, self.discriminator_dims)
        
        return GANModel(generator, discriminator)
    
    def train(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 2e-4,
        n_critic: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Train GAN ROM."""
        if train_data.ndim == 3:
            train_data = train_data.reshape(-1, train_data.shape[-1])
        
        g_optimizer = torch.optim.Adam(
            self.model.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
        )
        d_optimizer = torch.optim.Adam(
            self.model.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
        )
        
        criterion = nn.BCELoss()
        
        g_losses = []
        d_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            g_epoch_loss = 0.0
            d_epoch_loss = 0.0
            n_batches = 0
            
            indices = np.random.permutation(len(train_data))
            for i in range(0, len(train_data), batch_size):
                batch_idx = indices[i:i+batch_size]
                real_data = torch.FloatTensor(train_data[batch_idx]).to(self.device)
                batch_size_actual = len(batch_idx)
                
                # Train Discriminator
                for _ in range(n_critic):
                    d_optimizer.zero_grad()
                    
                    # Real data
                    real_labels = torch.ones(batch_size_actual, 1).to(self.device)
                    d_real_output = self.model.discriminator(real_data)
                    d_real_loss = criterion(d_real_output, real_labels)
                    
                    # Fake data
                    z = torch.randn(batch_size_actual, self.latent_dim).to(self.device)
                    fake_data = self.model.generator(z)
                    fake_labels = torch.zeros(batch_size_actual, 1).to(self.device)
                    d_fake_output = self.model.discriminator(fake_data.detach())
                    d_fake_loss = criterion(d_fake_output, fake_labels)
                    
                    d_loss = d_real_loss + d_fake_loss
                    d_loss.backward()
                    d_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                z = torch.randn(batch_size_actual, self.latent_dim).to(self.device)
                fake_data = self.model.generator(z)
                g_output = self.model.discriminator(fake_data)
                g_loss = criterion(g_output, real_labels)
                g_loss.backward()
                g_optimizer.step()
                
                g_epoch_loss += g_loss.item()
                d_epoch_loss += d_loss.item()
                n_batches += 1
            
            avg_g_loss = g_epoch_loss / n_batches if n_batches > 0 else 0.0
            avg_d_loss = d_epoch_loss / n_batches if n_batches > 0 else 0.0
            g_losses.append(avg_g_loss)
            d_losses.append(avg_d_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, G Loss: {avg_g_loss:.6f}, D Loss: {avg_d_loss:.6f}")
        
        self.is_trained = True
        
        return {
            "generator_losses": g_losses,
            "discriminator_losses": d_losses,
            "final_g_loss": g_losses[-1] if g_losses else None,
            "final_d_loss": d_losses[-1] if d_losses else None,
        }
    
    def predict(self, initial_state: np.ndarray, n_steps: int) -> np.ndarray:
        """Predict future states using GAN ROM."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        self.model.generator.eval()
        
        if initial_state.ndim == 1:
            initial_state = initial_state.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = initial_state.shape[0]
        
        with torch.no_grad():
            # Encode initial state to latent (simplified - use projection)
            # In practice, you'd train an encoder
            x0 = torch.FloatTensor(initial_state).to(self.device)
            z0 = torch.randn(batch_size, self.latent_dim).to(self.device) * 0.1
            
            trajectory = [initial_state]
            z_current = z0
            
            for _ in range(n_steps):
                # Evolve in latent space (simple random walk + generator)
                z_next = z_current + torch.randn_like(z_current) * 0.01
                x_next = self.model.generator(z_next)
                trajectory.append(x_next.cpu().numpy())
                z_current = z_next
        
        trajectory = np.array(trajectory)
        trajectory = np.transpose(trajectory, (2, 0, 1)) if trajectory.ndim == 3 else trajectory
        
        if squeeze_output:
            trajectory = trajectory[:, :, 0] if trajectory.ndim == 3 else trajectory
        
        return trajectory
