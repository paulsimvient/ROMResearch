"""Example usage of ROM type registration system."""

import numpy as np
from .registry import ModelRegistry
from . import register_default_rom_types, get_default_registry


def example_basic_usage():
    """Example of basic ROM type registration and usage."""
    
    # Create a registry
    registry = ModelRegistry()
    
    # Register default ROM types
    register_default_rom_types(registry)
    
    # List available ROM types
    print("Available ROM types:")
    for rom_type, info in registry.rom_types.list_rom_types().items():
        print(f"  - {rom_type}: {info['description']}")
    
    # Create a DMD ROM using the factory
    dmd_rom = registry.rom_types.create_rom(
        rom_type="DMD",
        model_id="dmd_001",
        name="My DMD Model",
        rank=10,
        implementation="numpy"  # or "pymor"
    )
    
    # Fit the model with some data
    # Generate example snapshot data
    n_features = 100
    n_samples = 200
    data = np.random.randn(n_features, n_samples)
    dmd_rom.fit(data)
    
    # Register the model
    registry.register(dmd_rom)
    
    # Create and register a Koopman ROM
    koopman_rom = registry.rom_types.create_rom(
        rom_type="Koopman",
        model_id="koopman_001",
        name="My Koopman Model",
        n_modes=15
    )
    koopman_rom.fit(data)
    registry.register(koopman_rom)
    
    # List all registered models
    print("\nRegistered models:")
    for model_id, model in registry.list_models().items():
        print(f"  - {model_id}: {model.name} ({model.model_type})")
    
    # Simulate with a model
    x0 = np.random.randn(n_features)
    result = dmd_rom.simulate({"x0": x0, "n_steps": 50})
    print(f"\nDMD simulation result shape: {result.shape}")


def example_custom_rom_registration():
    """Example of registering a custom ROM type."""
    
    from .base import ModelAdapter
    
    # Define a custom ROM class
    class CustomROM(ModelAdapter):
        def __init__(self, model_id: str, name: str, **kwargs):
            super().__init__(model_id, name, "ROM")
            self.custom_param = kwargs.get("custom_param", 1.0)
        
        def simulate(self, input_params):
            # Custom simulation logic
            return np.array([[1.0, 2.0, 3.0]])
        
        def metadata(self):
            return {
                "id": self.model_id,
                "name": self.name,
                "type": self.model_type,
                "method": "Custom",
                "custom_param": self.custom_param,
            }
    
    # Create registry and register default types
    registry = ModelRegistry()
    register_default_rom_types(registry)
    
    # Register custom ROM type
    def create_custom(model_id: str, name: str, **kwargs):
        return CustomROM(model_id=model_id, name=name, **kwargs)
    
    registry.rom_types.register_rom_type(
        rom_type="Custom",
        factory=create_custom,
        description="Custom ROM implementation",
        metadata={"version": "1.0"}
    )
    
    # Now you can create custom ROMs using the factory
    custom_rom = registry.rom_types.create_rom(
        rom_type="Custom",
        model_id="custom_001",
        name="My Custom ROM",
        custom_param=2.5
    )
    
    registry.register(custom_rom)
    print(f"Created custom ROM: {custom_rom.metadata()}")


def example_default_registry():
    """Example using the default global registry."""
    
    # Get the default registry (automatically has all ROM types registered)
    registry = get_default_registry()
    
    # Create ROMs directly
    dmd_rom = registry.rom_types.create_rom(
        rom_type="DMD",
        model_id="dmd_default",
        name="Default DMD",
        rank=5
    )
    
    print(f"Created ROM: {dmd_rom.metadata()}")


if __name__ == "__main__":
    print("=== Basic Usage ===")
    example_basic_usage()
    
    print("\n=== Custom ROM Registration ===")
    example_custom_rom_registration()
    
    print("\n=== Default Registry ===")
    example_default_registry()
