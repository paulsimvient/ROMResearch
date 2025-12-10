"""FMU adapter for Functional Mock-up Interface models."""
from typing import Dict, Any
import numpy as np
from pathlib import Path

try:
    from pyfmi import load_fmu
    PYFMI_AVAILABLE = True
except ImportError:
    PYFMI_AVAILABLE = False
    load_fmu = None

from .base import ModelAdapter


class FMUAdapter(ModelAdapter):
    """Adapter for FMU (Functional Mock-up Unit) models."""
    
    def __init__(self, model_id: str, name: str, fmu_path: str):
        """
        Initialize FMU adapter.
        
        Args:
            model_id: Unique identifier
            name: Human-readable name
            fmu_path: Path to .fmu file
        """
        if not PYFMI_AVAILABLE:
            raise ImportError("pyfmi is required for FMU support. Install with: pip install pyfmi")
        
        super().__init__(model_id, name, "FOM")
        self.fmu_path = Path(fmu_path)
        if not self.fmu_path.exists():
            raise FileNotFoundError(f"FMU file not found: {fmu_path}")
        
        self._model = None
        self._input_shape = None
        self._output_shape = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load FMU model."""
        self._model = load_fmu(str(self.fmu_path))
        # Infer input/output shapes from FMU variables
        # This is a simplified version - real implementation would query FMU variables
        self._input_shape = (1,)  # Placeholder
        self._output_shape = (1,)  # Placeholder
    
    def simulate(self, input_params: Dict[str, Any]) -> np.ndarray:
        """
        Run FMU simulation.
        
        Args:
            input_params: Dictionary with:
                - "t": time array
                - "u": input signal (optional)
                - Other FMU-specific parameters
        
        Returns:
            Simulation output array
        """
        if self._model is None:
            raise RuntimeError("FMU model not loaded")
        
        # Extract time vector
        t = input_params.get("t", np.linspace(0, 1, 100))
        if isinstance(t, (list, tuple)):
            t = np.array(t)
        
        # Reset FMU
        self._model.reset()
        
        # Set initial conditions if provided
        if "initial_conditions" in input_params:
            for var, value in input_params["initial_conditions"].items():
                self._model.set(var, value)
        
        # Run simulation
        opts = self._model.simulate_options()
        opts["ncp"] = len(t)  # Number of communication points
        
        result = self._model.simulate(start_time=t[0], final_time=t[-1], options=opts)
        
        # Extract output (simplified - would need to know actual output variable names)
        # For now, return time series of first output variable
        output_vars = result.keys()
        if len(output_vars) > 0:
            # Return first available output variable
            output_key = list(output_vars)[0]
            return result[output_key]
        else:
            # Fallback: return time vector
            return result["time"]
    
    def metadata(self) -> Dict[str, Any]:
        """Get FMU metadata."""
        return {
            "id": self.model_id,
            "name": self.name,
            "type": self.model_type,
            "fmu_path": str(self.fmu_path),
            "input_shape": self._input_shape,
            "output_shape": self._output_shape,
            "fmu_version": getattr(self._model, "get_version", lambda: "unknown")(),
        }

