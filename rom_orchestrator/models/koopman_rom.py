"""Koopman operator ROM implementation."""
from typing import Dict, Any, Optional
import numpy as np
from scipy.linalg import svd, pinv

try:
    from pykoopman import Koopman
    PYKOOPMAN_AVAILABLE = True
except ImportError:
    PYKOOPMAN_AVAILABLE = False
    Koopman = None

from .base import ModelAdapter


class KoopmanROM(ModelAdapter):
    """Koopman Operator Reduced-Order Model."""
    
    def __init__(
        self,
        model_id: str,
        name: str,
        n_modes: int = 10,
        fit_data: Optional[np.ndarray] = None
    ):
        """
        Initialize Koopman ROM.
        
        Args:
            model_id: Unique identifier
            name: Human-readable name
            n_modes: Number of Koopman modes
            fit_data: Training data [n_features, n_samples] for fitting
        """
        super().__init__(model_id, name, "ROM")
        self.n_modes = n_modes
        self.koopman_model = None
        self._input_shape = None
        self._output_shape = None
        
        if fit_data is not None:
            self.fit(fit_data)
    
    def fit(self, data: np.ndarray) -> None:
        """
        Fit Koopman model from snapshot data.
        
        Args:
            data: Snapshot matrix [n_features, n_samples]
        """
        if data.ndim != 2:
            raise ValueError("Data must be 2D array [n_features, n_samples]")
        
        if not PYKOOPMAN_AVAILABLE:
            # Fallback: simple DMD-based approximation
            self._fit_simple_koopman(data)
            return
        
        # Check data format - should be [n_features, n_samples]
        if data.shape[0] > data.shape[1]:
            # Likely [n_samples, n_features], transpose to [n_features, n_samples]
            data = data.T
        
        n_features, n_samples = data.shape
        
        # Use PyKoopman if available
        try:
            from pykoopman import observables
            from pykoopman.observables import Polynomial
            
            # Prepare data for PyKoopman (expects [n_samples, n_features])
            X = data.T
            
            # Create observables
            obs = Polynomial(degree=2)
            
            # Fit Koopman model
            self.koopman_model = Koopman(observables=obs, regressor=1e-6)
            self.koopman_model.fit(X, n_inputs=n_features)
            
            self._input_shape = (n_features,)
            self._output_shape = (n_features,)
        except Exception as e:
            # Fallback to simple implementation
            print(f"PyKoopman fitting failed: {e}, using simple implementation")
            self._fit_simple_koopman(data)
    
    def _fit_simple_koopman(self, data: np.ndarray) -> None:
        """Simple Koopman approximation using DMD-like approach."""
        n_features, n_samples = data.shape
        
        # Split into X and Y
        X = data[:, :-1]
        Y = data[:, 1:]
        
        # SVD
        U, s, Vh = svd(X, full_matrices=False)
        
        # Truncate to n_modes
        if self.n_modes < len(s):
            U = U[:, :self.n_modes]
            s = s[:self.n_modes]
            Vh = Vh[:self.n_modes, :]
        
        # Compute Koopman operator approximation
        S_inv = np.diag(1.0 / s)
        K_tilde = U.T @ Y @ Vh.T @ S_inv
        
        # Store for simulation
        self.koopman_model = {
            "U": U,
            "s": s,
            "Vh": Vh,
            "K_tilde": K_tilde,
        }
        
        self._input_shape = (n_features,)
        self._output_shape = (n_features,)
    
    def simulate(self, input_params: Dict[str, Any]) -> np.ndarray:
        """
        Run Koopman ROM simulation.
        
        Args:
            input_params: Dictionary with:
                - "x0": Initial condition [n_features]
                - "t": Time array (optional)
                - "n_steps": Number of time steps (if t not provided)
        
        Returns:
            Time series [n_features, n_steps]
        """
        if self.koopman_model is None:
            raise RuntimeError("Koopman model not fitted. Call fit() first.")
        
        # Get initial condition
        x0 = input_params.get("x0")
        if x0 is None:
            raise ValueError("Koopman simulation requires 'x0' initial condition")
        x0 = np.array(x0).flatten()
        
        # Get time array
        t = input_params.get("t")
        if t is None:
            n_steps = input_params.get("n_steps", 100)
            t = np.arange(n_steps)
        else:
            t = np.array(t)
            n_steps = len(t)
        
        # Use PyKoopman if available
        if PYKOOPMAN_AVAILABLE and hasattr(self.koopman_model, 'simulate'):
            try:
                result = self.koopman_model.simulate(x0.reshape(1, -1), n_steps=n_steps)
                return result.T  # [n_features, n_steps]
            except Exception:
                pass
        
        # Fallback: use simple implementation
        if isinstance(self.koopman_model, dict):
            U = self.koopman_model["U"]
            K_tilde = self.koopman_model["K_tilde"]
            
            # Project initial condition
            z0 = U.T @ x0
            
            # Simulate in lifted space
            result = np.zeros((len(x0), n_steps))
            z = z0.copy()
            for i in range(n_steps):
                result[:, i] = U @ z
                if i < n_steps - 1:
                    z = K_tilde @ z
            
            return result
        
        raise RuntimeError("Koopman model not properly initialized")
    
    def metadata(self) -> Dict[str, Any]:
        """Get Koopman ROM metadata."""
        return {
            "id": self.model_id,
            "name": self.name,
            "type": self.model_type,
            "method": "Koopman",
            "n_modes": self.n_modes,
            "input_shape": self._input_shape,
            "output_shape": self._output_shape,
        }

