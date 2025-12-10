"""Model adapters for FOMs and ROMs."""
from .base import ModelAdapter
from .registry import ModelRegistry, ROMTypeRegistry
from .dmd_rom import DMDROM
from .koopman_rom import KoopmanROM
from .nn_rom import NeuralNetworkROM
from .autoencoder_rom import AutoencoderROM
from .fmu_adapter import FMUAdapter
from .pymor_adapter import PyMORAdapter
from .fom_models import HeatEquationFOM, CoupledOscillatorsFOM, BurgersEquationFOM
from .ezyrb_rom import EZyRBROM

__all__ = [
    "ModelAdapter",
    "ModelRegistry",
    "ROMTypeRegistry",
    "DMDROM",
    "KoopmanROM",
    "NeuralNetworkROM",
    "AutoencoderROM",
    "FMUAdapter",
    "PyMORAdapter",
    "HeatEquationFOM",
    "CoupledOscillatorsFOM",
    "BurgersEquationFOM",
    "EZyRBROM",
    "register_default_rom_types",
    "get_default_registry",
]


def register_default_rom_types(registry: ModelRegistry) -> None:
    """
    Register all default ROM types in the registry.
    
    Args:
        registry: ModelRegistry instance to register ROM types in
    """
    rom_types = registry.rom_types
    
    # Register DMD ROM
    def create_dmd(model_id: str, name: str, **kwargs):
        return DMDROM(model_id=model_id, name=name, **kwargs)
    
    rom_types.register_rom_type(
        rom_type="DMD",
        factory=create_dmd,
        description="Dynamic Mode Decomposition ROM (supports NumPy and PyMOR implementations)",
        metadata={
            "implementations": ["numpy", "pymor"],
            "default_implementation": "numpy",
        }
    )
    
    # Register Koopman ROM
    def create_koopman(model_id: str, name: str, **kwargs):
        return KoopmanROM(model_id=model_id, name=name, **kwargs)
    
    rom_types.register_rom_type(
        rom_type="Koopman",
        factory=create_koopman,
        description="Koopman Operator ROM (supports PyKoopman and custom implementations)",
        metadata={
            "implementations": ["pykoopman", "custom"],
            "default_implementation": "pykoopman",
        }
    )
    
    # Register Neural Network ROM
    def create_nn(model_id: str, name: str, **kwargs):
        return NeuralNetworkROM(model_id=model_id, name=name, **kwargs)
    
    rom_types.register_rom_type(
        rom_type="NeuralNetwork",
        factory=create_nn,
        description="Neural Network Reduced-Order Model",
        metadata={
            "framework": "pytorch",
        }
    )
    
    # Register Autoencoder ROM
    def create_autoencoder(model_id: str, name: str, **kwargs):
        return AutoencoderROM(model_id=model_id, name=name, **kwargs)
    
    rom_types.register_rom_type(
        rom_type="Autoencoder",
        factory=create_autoencoder,
        description="Autoencoder-based ROM: Encoder → Latent Dynamics → Decoder",
        metadata={
            "framework": "pytorch",
            "components": ["encoder", "decoder", "latent_dynamics"],
        }
    )
    
    # Register EZyRB ROM
    def create_ezyrb(model_id: str, name: str, **kwargs):
        return EZyRBROM(model_id=model_id, name=name, **kwargs)
    
    rom_types.register_rom_type(
        rom_type="EZyRB",
        factory=create_ezyrb,
        description="EZyRB (Easy Reduced Basis) - Data-driven parametric ROM with POD and interpolation",
        metadata={
            "interpolators": ["RBF", "GPR", "ANN"],
            "default_interpolator": "RBF",
            "method": "POD + interpolation",
        }
    )


# Global default registry instance
_default_registry: ModelRegistry = None


def get_default_registry() -> ModelRegistry:
    """
    Get or create the default model registry with all ROM types registered.
    
    Returns:
        ModelRegistry instance with default ROM types registered
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ModelRegistry()
        register_default_rom_types(_default_registry)
    return _default_registry

