# ML-ROM: Machine Learning Reduced Order Models

A complete implementation of Machine Learning-based Reduced Order Models (ML-ROM) for high-dimensional dynamical systems.

## Overview

This directory contains state-of-the-art ML-ROM implementations using deep learning techniques:

- **Neural ODE ROM**: Continuous-time dynamics using neural ODEs
- **LSTM ROM**: Sequential modeling with LSTM networks
- **Transformer ROM**: Attention-based sequence modeling
- **VAE ROM**: Variational Autoencoder with latent dynamics
- **GAN ROM**: Generative Adversarial Network for state generation

## Structure

```
ml_rom/
├── models/          # ML-ROM model implementations
├── utils/           # Data loading and preprocessing utilities
├── trainers/        # Training utilities and helpers
└── README.md        # This file
```

## Installation

```bash
pip install torch torchdiffeq scikit-learn matplotlib numpy
```

## Quick Start

```python
from ml_rom import NeuralODEROM, MLROMTrainer
import numpy as np

# Create model
model = NeuralODEROM(
    model_id="neural_ode_001",
    name="Neural ODE ROM",
    input_dim=100,
    latent_dim=50,
    hidden_dims=[64, 64]
)

# Prepare data (snapshots from FOM)
snapshots = np.random.randn(200, 100)  # [n_samples, n_features]

# Train
trainer = MLROMTrainer(model)
history = trainer.train(snapshots, epochs=100, batch_size=32)

# Predict
initial_state = snapshots[0]
predictions = model.predict(initial_state, n_steps=50)

# Evaluate
results = trainer.evaluate(snapshots, n_steps=50)
print(f"Relative Error: {results['relative_error']*100:.2f}%")
```

## Model Types

### Neural ODE ROM
Learns continuous-time dynamics: `dx/dt = f(x, t)` where `f` is a neural network.

```python
model = NeuralODEROM(
    model_id="node_rom",
    name="Neural ODE",
    input_dim=100,
    latent_dim=50,
    hidden_dims=[64, 64]
)
```

### LSTM ROM
Uses LSTM for sequential state prediction with encoder-decoder architecture.

```python
model = LSTMROM(
    model_id="lstm_rom",
    name="LSTM ROM",
    input_dim=100,
    latent_dim=50,
    hidden_dim=128,
    num_layers=2
)
```

### Transformer ROM
Attention-based sequence modeling for long-range dependencies.

```python
model = TransformerROM(
    model_id="transformer_rom",
    name="Transformer ROM",
    input_dim=100,
    latent_dim=50,
    d_model=128,
    nhead=8,
    num_layers=4
)
```

### VAE ROM
Variational Autoencoder with learned latent dynamics.

```python
model = VAEROM(
    model_id="vae_rom",
    name="VAE ROM",
    input_dim=100,
    latent_dim=50,
    encoder_dims=[128, 64],
    decoder_dims=[64, 128]
)
```

### GAN ROM
Generative Adversarial Network for state distribution learning.

```python
model = GANROM(
    model_id="gan_rom",
    name="GAN ROM",
    input_dim=100,
    latent_dim=50,
    generator_dims=[128, 256, 128],
    discriminator_dims=[128, 64]
)
```

## Training

All models support flexible training:

```python
history = model.train(
    train_data=snapshots,
    val_data=val_snapshots,  # Optional
    epochs=100,
    batch_size=32,
    learning_rate=1e-3
)
```

## Saving and Loading

```python
# Save
model.save("model_checkpoint.pt")

# Load
model = NeuralODEROM(...)
model.load("model_checkpoint.pt")
```

## Integration with ROM Orchestrator

ML-ROM models can be integrated with the main ROM orchestrator:

```python
from rom_orchestrator.models import ModelAdapter
from ml_rom import NeuralODEROM

# Wrap ML-ROM as ModelAdapter
class MLROMAdapter(ModelAdapter):
    def __init__(self, ml_rom_model):
        super().__init__(ml_rom_model.model_id, ml_rom_model.name, "ROM")
        self.ml_rom = ml_rom_model
    
    def simulate(self, input_params):
        x0 = input_params["x0"]
        n_steps = input_params.get("n_steps", 100)
        result = self.ml_rom.predict(x0, n_steps)
        return result
    
    def metadata(self):
        return self.ml_rom.metadata()
```

## Examples

See `examples/` directory for complete usage examples.

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- SciPy
- scikit-learn
- matplotlib
- torchdiffeq (for Neural ODE)
