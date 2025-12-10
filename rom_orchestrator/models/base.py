"""Base model adapter interface."""
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class ModelAdapter(ABC):
    """Base class for all model adapters (FOMs and ROMs)."""
    
    def __init__(self, model_id: str, name: str, model_type: str):
        """
        Initialize model adapter.
        
        Args:
            model_id: Unique identifier for the model
            name: Human-readable name
            model_type: "FOM" or "ROM"
        """
        self.model_id = model_id
        self.name = name
        self.model_type = model_type
    
    @abstractmethod
    def simulate(self, input_params: Dict[str, Any]) -> np.ndarray:
        """
        Run simulation with given input parameters.
        
        Args:
            input_params: Dictionary of input parameters (e.g., {"t": time_array, "u": input_signal})
        
        Returns:
            Array of simulation outputs
        """
        pass
    
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            Dictionary with model information (id, name, type, input_shape, output_shape, etc.)
        """
        pass
    
    def get_input_shape(self) -> tuple:
        """Get expected input shape."""
        meta = self.metadata()
        return tuple(meta.get("input_shape", []))
    
    def get_output_shape(self) -> tuple:
        """Get expected output shape."""
        meta = self.metadata()
        return tuple(meta.get("output_shape", []))

