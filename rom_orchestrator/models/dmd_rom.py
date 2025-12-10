"""Dynamic Mode Decomposition (DMD) ROM implementation."""
from typing import Dict, Any, Optional, Literal
import numpy as np
from scipy.linalg import svd, pinv

try:
    import pymor
    # Try to import DMD - location may vary by PyMOR version
    try:
        from pymor.algorithms.dmd import dmd
    except ImportError:
        try:
            from pymor.algorithms import dmd
        except ImportError:
            dmd = None
    PYMOR_AVAILABLE = True
except ImportError:
    PYMOR_AVAILABLE = False
    pymor = None
    dmd = None

from .base import ModelAdapter


class DMDROM(ModelAdapter):
    """Dynamic Mode Decomposition Reduced-Order Model.
    
    Supports both PyMOR and custom NumPy implementations.
    """
    
    def __init__(
        self,
        model_id: str,
        name: str,
        rank: Optional[int] = None,
        fit_data: Optional[np.ndarray] = None,
        implementation: Literal["numpy", "pymor"] = "numpy"
    ):
        """
        Initialize DMD ROM.
        
        Args:
            model_id: Unique identifier
            name: Human-readable name
            rank: Truncation rank (number of modes). If None, uses all modes.
            fit_data: Training data matrix [n_features, n_samples] for fitting
            implementation: "numpy" for custom NumPy implementation, "pymor" for PyMOR
        """
        super().__init__(model_id, name, "ROM")
        self.rank = rank
        self.implementation = implementation
        self.A_dmd = None  # DMD operator
        self.modes = None  # DMD modes
        self.eigenvalues = None  # DMD eigenvalues
        self.pymor_dmd_model = None  # PyMOR DMD model
        self._input_shape = None
        self._output_shape = None
        
        if implementation == "pymor" and not PYMOR_AVAILABLE:
            raise ImportError(
                "PyMOR is required for 'pymor' implementation. "
                "Install with: pip install pymor"
            )
        
        if fit_data is not None:
            self.fit(fit_data)
    
    def fit(self, data: np.ndarray) -> None:
        """
        Fit DMD model from snapshot data.
        
        Args:
            data: Snapshot matrix [n_features, n_samples]
        """
        if data.ndim != 2:
            raise ValueError("Data must be 2D array [n_features, n_samples]")
        
        # Check data format - should be [n_features, n_samples]
        # DMD expects [n_features, n_samples] where typically n_features >= n_samples
        # The orchestrator should pass data in correct format, but verify
        # Heuristic: if first dim < second dim, likely [n_samples, n_features] - transpose
        # For typical FOMs: n_features (n_dof) is usually > 100, n_samples is usually < 1000
        original_shape = data.shape
        if data.shape[0] < data.shape[1]:
            # First dim is smaller - likely [n_samples, n_features]
            # Additional check: if first dim is reasonable for n_samples (< 1000) and second is large (> 100)
            if data.shape[0] < 1000 and data.shape[1] > 100:
                data = data.T
                print(f"DMD: Transposed data from {original_shape} to {data.shape} (detected [n_samples, n_features] format)")
        
        n_features, n_samples = data.shape
        
        # Log dimensions for debugging
        print(f"DMD fit: data shape = ({n_features}, {n_samples}), rank = {self.rank}")
        
        # Verify we have reasonable dimensions
        if n_samples < 2:
            raise ValueError(f"DMD requires at least 2 snapshots, got {n_samples}")
        
        if n_features == 0:
            raise ValueError(f"DMD requires at least 1 feature, got {n_features}")
        
        if self.implementation == "pymor":
            self._fit_pymor(data)
        else:
            self._fit_numpy(data)
    
    def _fit_numpy(self, data: np.ndarray) -> None:
        """Fit DMD using custom NumPy implementation."""
        n_features, n_samples = data.shape
        
        if n_samples < 2:
            raise ValueError(f"DMD requires at least 2 snapshots, got {n_samples}")
        
        # Split into X and Y (shifted by one time step)
        X = data[:, :-1]  # [n_features, n_samples-1]
        Y = data[:, 1:]   # [n_features, n_samples-1]
        
        # SVD of X
        # X should be [n_features, n_samples-1]
        if X.shape[0] != n_features or X.shape[1] != n_samples - 1:
            raise ValueError(
                f"DMD: X matrix has wrong shape. Expected [{n_features}, {n_samples-1}], "
                f"got {X.shape}. Original data shape: [{n_features}, {n_samples}]"
            )
        U, s, Vh = svd(X, full_matrices=False)
        
        # SVD returns:
        #   U: [n_features, min(n_features, n_samples-1)]
        #   s: [min(n_features, n_samples-1)]
        #   Vh: [min(n_features, n_samples-1), n_samples-1]
        # When n_features >= n_samples-1: U is [n_features, n_samples-1], Vh is [n_samples-1, n_samples-1]
        # When n_features < n_samples-1: U is [n_features, n_features], Vh is [n_features, n_samples-1]
        
        # Determine rank - can't exceed the number of singular values or the matrix dimensions
        max_rank = min(len(s), n_samples - 1, U.shape[1], Vh.shape[0])
        if self.rank is not None:
            rank = min(self.rank, max_rank)
        else:
            rank = max_rank
        
        # Ensure rank is at least 1
        if rank < 1:
            raise ValueError(f"DMD: Invalid rank {rank}. Need at least rank=1")
        
        # Truncate to rank
        # U: [n_features, min(n_features, n_samples-1)] -> [n_features, rank]
        U = U[:, :rank]
        # s: [min(n_features, n_samples-1)] -> [rank]
        s = s[:rank]
        # Vh: [min(n_features, n_samples-1), n_samples-1] -> [rank, n_samples-1]
        Vh = Vh[:rank, :]
        
        # Verify shapes after truncation
        if U.shape != (n_features, rank):
            raise ValueError(f"DMD: U shape mismatch after truncation. Expected ({n_features}, {rank}), got {U.shape}")
        if len(s) != rank:
            raise ValueError(f"DMD: s length mismatch after truncation. Expected length={rank}, got {len(s)}")
        if Vh.shape != (rank, n_samples - 1):
            raise ValueError(f"DMD: Vh shape mismatch after truncation. Expected ({rank}, {n_samples-1}), got {Vh.shape}")
        
        # Avoid division by zero
        s = np.maximum(s, 1e-10)
        
        # Compute DMD operator: A_tilde = U^T @ Y @ V @ S^{-1}
        # Where V = Vh^T (conjugate transpose)
        # Dimensions:
        #   U: [n_features, rank]
        #   U.T: [rank, n_features]
        #   Y: [n_features, n_samples-1]
        #   Vh: [rank, n_samples-1]
        #   Vh.T: [n_samples-1, rank]
        #   S_inv: [rank, rank]
        #   U.T @ Y: [rank, n_features] @ [n_features, n_samples-1] = [rank, n_samples-1]
        #   (U.T @ Y) @ Vh.T: [rank, n_samples-1] @ [n_samples-1, rank] = [rank, rank]
        #   A_tilde: [rank, rank] @ [rank, rank] = [rank, rank]
        
        # Log shapes for debugging
        print(f"DMD: U shape = {U.shape}, s length = {len(s)}, Vh shape = {Vh.shape}, rank = {rank}")
        print(f"DMD: Y shape = {Y.shape}, n_features = {n_features}, n_samples-1 = {n_samples-1}")
        
        S_inv = np.diag(1.0 / s)
        
        # Verify dimensions before multiplication
        U_T = U.T  # Should be [rank, n_features]
        if U_T.shape != (rank, n_features):
            raise ValueError(
                f"DMD: U.T has wrong shape. Expected ({rank}, {n_features}), got {U_T.shape}. "
                f"U shape: {U.shape}"
            )
        if U_T.shape[1] != Y.shape[0]:
            raise ValueError(
                f"DMD dimension mismatch: U.T shape {U_T.shape}, Y shape {Y.shape}. "
                f"Expected U.T columns ({U_T.shape[1]}) to match Y rows ({Y.shape[0]})"
            )
        
        temp = U_T @ Y  # Should be [rank, n_samples-1]
        if temp.shape != (rank, n_samples - 1):
            raise ValueError(
                f"DMD: (U.T @ Y) has wrong shape. Expected ({rank}, {n_samples-1}), got {temp.shape}"
            )
        
        Vh_T = Vh.T  # Should be [n_samples-1, rank]
        if Vh_T.shape != (n_samples - 1, rank):
            raise ValueError(
                f"DMD: Vh.T has wrong shape. Expected ({n_samples-1}, {rank}), got {Vh_T.shape}. "
                f"Vh shape: {Vh.shape}"
            )
        
        if temp.shape[1] != Vh_T.shape[0]:
            raise ValueError(
                f"DMD dimension mismatch: (U.T @ Y) shape {temp.shape}, Vh.T shape {Vh_T.shape}. "
                f"Expected (U.T @ Y) columns ({temp.shape[1]}) to match Vh.T rows ({Vh_T.shape[0]})"
            )
        
        A_tilde = temp @ Vh_T @ S_inv
        
        # Eigendecomposition of A_tilde
        eigenvals, eigenvecs = np.linalg.eig(A_tilde)
        
        # Compute DMD modes: modes = Y @ Vh.T @ S_inv @ eigenvecs
        # Dimensions:
        #   Y: [n_features, n_samples-1]
        #   Vh.T: [n_samples-1, rank]
        #   Y @ Vh.T: [n_features, n_samples-1] @ [n_samples-1, rank] = [n_features, rank]
        #   S_inv: [rank, rank]
        #   (Y @ Vh.T) @ S_inv: [n_features, rank] @ [rank, rank] = [n_features, rank]
        #   eigenvecs: [rank, rank] (from eigendecomposition of A_tilde)
        #   modes: [n_features, rank] @ [rank, rank] = [n_features, rank]
        Vh_T = Vh.T  # [n_samples-1, rank]
        if Vh_T.shape != (n_samples - 1, rank):
            raise ValueError(
                f"DMD: Vh.T has wrong shape for modes computation. Expected ({n_samples-1}, {rank}), got {Vh_T.shape}"
            )
        
        temp = Y @ Vh_T  # Should be [n_features, rank]
        if temp.shape != (n_features, rank):
            raise ValueError(
                f"DMD: (Y @ Vh.T) has wrong shape. Expected ({n_features}, {rank}), got {temp.shape}. "
                f"Y shape: {Y.shape}, Vh.T shape: {Vh_T.shape}"
            )
        
        if temp.shape[1] != S_inv.shape[0]:
            raise ValueError(
                f"DMD dimension mismatch: (Y @ Vh.T) shape {temp.shape}, S_inv shape {S_inv.shape}"
            )
        
        temp2 = temp @ S_inv  # Should be [n_features, rank]
        if temp2.shape != (n_features, rank):
            raise ValueError(
                f"DMD: ((Y @ Vh.T) @ S_inv) has wrong shape. Expected ({n_features}, {rank}), got {temp2.shape}"
            )
        
        if temp2.shape[1] != eigenvecs.shape[0]:
            raise ValueError(
                f"DMD dimension mismatch: ((Y @ Vh.T) @ S_inv) shape {temp2.shape}, eigenvecs shape {eigenvecs.shape}"
            )
        
        self.modes = temp2 @ eigenvecs  # [n_features, rank]
        
        self.A_dmd = A_tilde
        self.eigenvalues = eigenvals
        
        # Set shapes
        self._input_shape = (n_features,)
        self._output_shape = (n_features,)
    
    def _fit_pymor(self, data: np.ndarray) -> None:
        """Fit DMD using PyMOR implementation."""
        if not PYMOR_AVAILABLE:
            raise ImportError("PyMOR is required for PyMOR implementation")
        
        if dmd is None:
            raise ImportError("PyMOR DMD algorithm not available. Falling back to NumPy.")
        
        n_features, n_samples = data.shape
        
        try:
            # PyMOR expects data in a specific format
            # Convert numpy array to PyMOR VectorArray if needed
            from pymor.vectorarrays.numpy import NumpyVectorSpace
            
            # Create vector arrays from data
            space = NumpyVectorSpace(n_features)
            vectors = [space.from_numpy(data[:, i]) for i in range(n_samples)]
            
            # Use PyMOR DMD algorithm
            # Note: PyMOR's DMD interface may vary by version
            # This is a general approach
            if self.rank is not None:
                # Try to use rank parameter if available
                try:
                    dmd_result = dmd(vectors, modes=self.rank)
                except TypeError:
                    # Fallback if rank parameter not supported
                    dmd_result = dmd(vectors)
            else:
                dmd_result = dmd(vectors)
            
            # Extract DMD components from PyMOR result
            # PyMOR DMD result structure may vary
            if hasattr(dmd_result, 'modes'):
                self.modes = dmd_result.modes.to_numpy()
            if hasattr(dmd_result, 'eigenvalues'):
                self.eigenvalues = dmd_result.eigenvalues
            if hasattr(dmd_result, 'operator'):
                self.A_dmd = dmd_result.operator.matrix
            
            self.pymor_dmd_model = dmd_result
            
            # Set shapes
            self._input_shape = (n_features,)
            self._output_shape = (n_features,)
            
        except Exception as e:
            # Fallback to NumPy implementation if PyMOR fails
            print(f"PyMOR DMD fitting failed: {e}, falling back to NumPy implementation")
            self.implementation = "numpy"
            self._fit_numpy(data)
    
    def simulate(self, input_params: Dict[str, Any]) -> np.ndarray:
        """
        Run DMD ROM simulation.
        
        Args:
            input_params: Dictionary with:
                - "x0": Initial condition [n_features]
                - "t": Time array (optional, defaults to [0, 1, ..., n_steps-1])
                - "n_steps": Number of time steps (if t not provided)
        
        Returns:
            Time series [n_features, n_steps]
        """
        if self.implementation == "pymor" and self.pymor_dmd_model is not None:
            return self._simulate_pymor(input_params)
        else:
            return self._simulate_numpy(input_params)
    
    def _simulate_numpy(self, input_params: Dict[str, Any]) -> np.ndarray:
        """Simulate using NumPy implementation."""
        if self.A_dmd is None:
            raise RuntimeError("DMD model not fitted. Call fit() first.")
        
        # Get initial condition
        x0 = input_params.get("x0")
        if x0 is None:
            raise ValueError("DMD simulation requires 'x0' initial condition")
        x0 = np.array(x0).flatten()
        
        # Get time array
        t = input_params.get("t")
        if t is None:
            n_steps = input_params.get("n_steps", 100)
            t = np.arange(n_steps)
        else:
            t = np.array(t)
            n_steps = len(t)
        
        # Project initial condition onto DMD modes
        if self.modes is None:
            raise RuntimeError("DMD modes not computed")
        
        # Compute coefficients
        coeffs = pinv(self.modes) @ x0
        
        # Simulate using DMD eigenvalues
        result = np.zeros((len(x0), n_steps))
        for i, ti in enumerate(t):
            # Time evolution: x(t) = modes @ (eigenvalues^t * coeffs)
            if i == 0:
                result[:, i] = x0
            else:
                dt = t[i] - t[i-1] if i > 0 else 1.0
                evolved_coeffs = coeffs * (self.eigenvalues ** (i * dt))
                result[:, i] = self.modes @ evolved_coeffs
        
        return result
    
    def _simulate_pymor(self, input_params: Dict[str, Any]) -> np.ndarray:
        """Simulate using PyMOR implementation."""
        if self.pymor_dmd_model is None:
            raise RuntimeError("PyMOR DMD model not fitted. Call fit() first.")
        
        # Get initial condition
        x0 = input_params.get("x0")
        if x0 is None:
            raise ValueError("DMD simulation requires 'x0' initial condition")
        x0 = np.array(x0).flatten()
        
        # Get time array
        t = input_params.get("t")
        if t is None:
            n_steps = input_params.get("n_steps", 100)
            t = np.arange(n_steps)
        else:
            t = np.array(t)
            n_steps = len(t)
        
        try:
            # Use PyMOR DMD model for simulation
            from pymor.vectorarrays.numpy import NumpyVectorSpace
            space = NumpyVectorSpace(len(x0))
            x0_vec = space.from_numpy(x0)
            
            # Simulate using PyMOR DMD
            # PyMOR DMD models typically have a reconstruct method
            if hasattr(self.pymor_dmd_model, 'reconstruct'):
                result = self.pymor_dmd_model.reconstruct(x0_vec, t)
                if hasattr(result, 'to_numpy'):
                    return result.to_numpy()
                else:
                    return np.array(result)
            else:
                # Fallback to NumPy implementation
                return self._simulate_numpy(input_params)
        except Exception as e:
            # Fallback to NumPy implementation if PyMOR fails
            print(f"PyMOR DMD simulation failed: {e}, falling back to NumPy")
            return self._simulate_numpy(input_params)
    
    def metadata(self) -> Dict[str, Any]:
        """Get DMD ROM metadata."""
        return {
            "id": self.model_id,
            "name": self.name,
            "type": self.model_type,
            "method": "DMD",
            "implementation": self.implementation,
            "rank": self.rank,
            "input_shape": self._input_shape,
            "output_shape": self._output_shape,
            "n_modes": len(self.eigenvalues) if self.eigenvalues is not None else None,
            "pymor_available": PYMOR_AVAILABLE,
        }

