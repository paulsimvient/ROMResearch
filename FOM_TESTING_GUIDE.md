# Testing ROMs with Hard FOM Models

This guide explains how to test ROMs using realistic Full Order Models (FOMs).

## Available FOM Models

### 1. Heat Equation FOM
- **Equation**: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
- **Type**: 2D PDE, discretized on a grid
- **DOF**: nx × ny (default: 50×50 = 2500 DOF)
- **Parameters**:
  - `nx`: Grid points in x direction (default: 50)
  - `ny`: Grid points in y direction (default: 50)
  - `alpha`: Thermal diffusivity (default: 0.1)

### 2. Coupled Oscillators FOM
- **Type**: System of N coupled nonlinear Duffing oscillators
- **DOF**: 2 × n_oscillators (position + velocity for each)
- **Parameters**:
  - `n_oscillators`: Number of oscillators (default: 100)
  - `coupling_strength`: Coupling between neighbors (default: 0.1)
  - `nonlinearity`: Nonlinearity parameter (default: 0.5)

### 3. Burgers' Equation FOM
- **Equation**: ∂u/∂t + u(∂u/∂x) = ν(∂²u/∂x²)
- **Type**: 1D nonlinear PDE with shock formation
- **DOF**: nx (default: 200)
- **Parameters**:
  - `nx`: Number of grid points (default: 200)
  - `nu`: Viscosity coefficient (default: 0.01)

## Workflow: FOM → ROM

### Step 1: Create a FOM
1. Open the web app at http://127.0.0.1:5000
2. Go to "Create FOM Model" section
3. Select FOM type (e.g., "CoupledOscillators")
4. Enter Model ID (e.g., "osc_fom_001")
5. Optionally set parameters in JSON format:
   ```json
   {"n_oscillators": 100, "coupling_strength": 0.1, "nonlinearity": 0.5}
   ```
6. Click "Create FOM"

### Step 2: Generate Snapshots from FOM
1. Go to "Generate Snapshots from FOM" section
2. Select your FOM model
3. Set number of snapshots (e.g., 200)
4. Optionally set end time (or leave empty for default)
5. Click "Generate Snapshots"
6. This runs the FOM simulation and collects snapshots for ROM training

### Step 3: Create and Fit ROM
1. Create a ROM model (e.g., DMD with rank=20)
2. Go to "Fit Model" section
3. Select your ROM
4. Click "Fit Model" (uses the snapshots generated from FOM)

### Step 4: Compare ROM vs FOM
1. Run FOM simulation with a test initial condition
2. Run ROM simulation with the same initial condition
3. Compare results in the plots

## Example: Complete Workflow

### Via Web Interface:
1. **Create FOM**: CoupledOscillators, ID: "test_fom", params: `{"n_oscillators": 50}`
2. **Generate Snapshots**: 200 snapshots, t_end: 10.0
3. **Create ROM**: DMD, ID: "test_rom", params: `{"rank": 15, "implementation": "numpy"}`
4. **Fit ROM**: Use generated snapshots
5. **Simulate**: Compare FOM and ROM predictions

### Via Python API:
```python
from rom_orchestrator.models import (
    get_default_registry,
    CoupledOscillatorsFOM,
    DMDROM
)
import numpy as np

registry = get_default_registry()

# Create FOM
fom = CoupledOscillatorsFOM("fom_001", "Test FOM", n_oscillators=100)
registry.register(fom)

# Generate snapshots
snapshots = fom.simulate({"n_steps": 200, "t_end": 10.0})
# snapshots shape: [200 DOF, 200 snapshots]

# Create and fit ROM
rom = DMDROM("rom_001", "Test ROM", rank=20, implementation="numpy")
rom.fit(snapshots)
registry.register(rom)

# Compare predictions
x0 = snapshots[:, 0]  # Initial condition from FOM
fom_result = fom.simulate({"y0": x0, "n_steps": 50, "t_end": 5.0})
rom_result = rom.simulate({"x0": x0, "n_steps": 50})

# Compare error
error = np.linalg.norm(fom_result - rom_result) / np.linalg.norm(fom_result)
print(f"Relative error: {error:.4f}")
```

## Tips

- **Start small**: Test with smaller DOF first (e.g., 50 oscillators, 20×20 grid)
- **Rank selection**: For DMD, use rank << DOF (e.g., rank = DOF/10)
- **Snapshot count**: Use 2-5× more snapshots than desired ROM rank
- **Time range**: Make sure snapshots cover the dynamics you want to capture

## Performance Notes

- Heat Equation: Fast for small grids (< 50×50), slower for large grids
- Coupled Oscillators: Fast, scales well
- Burgers' Equation: Moderate speed, good for testing nonlinear ROMs
