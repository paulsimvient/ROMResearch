"""Central model registry for managing FOMs and ROMs."""
from typing import Dict, Optional, Type, Callable, Any
from .base import ModelAdapter


class ROMTypeRegistry:
    """Registry for ROM type factories and metadata."""
    
    def __init__(self):
        """Initialize empty ROM type registry."""
        self._rom_types: Dict[str, Dict[str, Any]] = {}
    
    def register_rom_type(
        self,
        rom_type: str,
        factory: Callable,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a ROM type with its factory function.
        
        Args:
            rom_type: Unique identifier for the ROM type (e.g., "DMD", "Koopman")
            factory: Factory function that creates ROM instances
            description: Human-readable description
            metadata: Additional metadata about the ROM type
        """
        if rom_type in self._rom_types:
            raise ValueError(f"ROM type '{rom_type}' already registered")
        
        self._rom_types[rom_type] = {
            "factory": factory,
            "description": description,
            "metadata": metadata or {},
        }
    
    def get_rom_type(self, rom_type: str) -> Optional[Dict[str, Any]]:
        """
        Get ROM type information.
        
        Args:
            rom_type: ROM type identifier
        
        Returns:
            Dictionary with factory, description, and metadata, or None if not found
        """
        return self._rom_types.get(rom_type)
    
    def list_rom_types(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered ROM types.
        
        Returns:
            Dictionary of rom_type -> {factory, description, metadata}
        """
        return self._rom_types.copy()
    
    def create_rom(
        self,
        rom_type: str,
        model_id: str,
        name: str,
        **kwargs
    ) -> ModelAdapter:
        """
        Create a ROM instance using registered factory.
        
        Args:
            rom_type: Registered ROM type identifier
            model_id: Unique model identifier
            name: Human-readable name
            **kwargs: Additional arguments passed to factory
        
        Returns:
            ModelAdapter instance
        """
        rom_info = self.get_rom_type(rom_type)
        if rom_info is None:
            raise ValueError(
                f"ROM type '{rom_type}' not registered. "
                f"Available types: {list(self._rom_types.keys())}"
            )
        
        factory = rom_info["factory"]
        return factory(model_id=model_id, name=name, **kwargs)
    
    def remove_rom_type(self, rom_type: str) -> bool:
        """
        Remove ROM type from registry.
        
        Args:
            rom_type: ROM type identifier
        
        Returns:
            True if removed, False if not found
        """
        if rom_type in self._rom_types:
            del self._rom_types[rom_type]
            return True
        return False


class ModelRegistry:
    """Registry for storing and retrieving model adapters."""
    
    def __init__(self):
        """Initialize empty registry."""
        self._models: Dict[str, ModelAdapter] = {}
        self._rom_type_registry = ROMTypeRegistry()
    
    def register(self, model: ModelAdapter) -> None:
        """
        Register a model adapter.
        
        Args:
            model: ModelAdapter instance to register
        """
        if model.model_id in self._models:
            raise ValueError(f"Model {model.model_id} already registered")
        self._models[model.model_id] = model
    
    def get(self, model_id: str) -> Optional[ModelAdapter]:
        """
        Get model by ID.
        
        Args:
            model_id: Unique model identifier
        
        Returns:
            ModelAdapter or None if not found
        """
        return self._models.get(model_id)
    
    def list_models(self, model_type: Optional[str] = None) -> Dict[str, ModelAdapter]:
        """
        List all registered models, optionally filtered by type.
        
        Args:
            model_type: Optional filter ("FOM" or "ROM")
        
        Returns:
            Dictionary of model_id -> ModelAdapter
        """
        if model_type is None:
            return self._models.copy()
        return {
            mid: model for mid, model in self._models.items()
            if model.model_type == model_type
        }
    
    def remove(self, model_id: str) -> bool:
        """
        Remove model from registry.
        
        Args:
            model_id: Unique model identifier
        
        Returns:
            True if removed, False if not found
        """
        if model_id in self._models:
            del self._models[model_id]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all models from registry."""
        self._models.clear()
    
    @property
    def rom_types(self) -> ROMTypeRegistry:
        """Get the ROM type registry."""
        return self._rom_type_registry

