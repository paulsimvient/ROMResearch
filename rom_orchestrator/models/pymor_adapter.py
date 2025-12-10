"""PyMOR adapter for reduced-order models."""
from typing import Dict, Any, Optional
import numpy as np

try:
    import pymor
    PYMOR_AVAILABLE = True
except ImportError:
    PYMOR_AVAILABLE = False
    pymor = None

from .base import ModelAdapter


class PyMORAdapter(ModelAdapter):
    """Adapter for PyMOR reduced-order models."""
    
    def __init__(self, model_id: str, name: str, pymor_model, input_shape: tuple, output_shape: tuple):
        """
        Initialize PyMOR adapter.
        
        Args:
            model_id: Unique identifier
            name: Human-readable name
            pymor_model: PyMOR model object
            input_shape: Expected input shape
            output_shape: Expected output shape
        """
        if not PYMOR_AVAILABLE:
            raise ImportError("pymor is required. Install with: pip install pymor")
        
        super().__init__(model_id, name, "ROM")
        self.pymor_model = pymor_model
        self._input_shape = input_shape
        self._output_shape = output_shape
    
    def simulate(self, input_params: Dict[str, Any]) -> np.ndarray:
        """
        Run PyMOR model simulation.
        
        Args:
            input_params: Dictionary with:
                - "mu": Parameter vector (PyMOR parameter)
                - "t": Time array (optional)
        
        Returns:
            Simulation output array
        """
        # Extract parameter
        mu = input_params.get("mu")
        if mu is None:
            raise ValueError("PyMOR models require 'mu' parameter")
        
        # Convert to PyMOR parameter if needed
        if not hasattr(mu, '__iter__'):
            mu = [mu]
        
        # Run model
        try:
            # PyMOR models typically have a solve() method
            if hasattr(self.pymor_model, 'solve'):
                result = self.pymor_model.solve(mu)
            elif hasattr(self.pymor_model, '__call__'):
                result = self.pymor_model(mu)
            else:
                raise AttributeError("PyMOR model does not have solve() or __call__() method")
            
            # Convert to numpy array
            if hasattr(result, 'to_numpy'):
                return result.to_numpy()
            elif hasattr(result, 'data'):
                return np.array(result.data)
            else:
                return np.array(result)
        except Exception as e:
            raise RuntimeError(f"PyMOR simulation failed: {e}")
    
    def metadata(self) -> Dict[str, Any]:
        """Get PyMOR model metadata."""
        return {
            "id": self.model_id,
            "name": self.name,
            "type": self.model_type,
            "input_shape": self._input_shape,
            "output_shape": self._output_shape,
            "pymor_version": pymor.__version__ if PYMOR_AVAILABLE else None,
        }

