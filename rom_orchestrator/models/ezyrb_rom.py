"""EZyRB (Easy Reduced Basis) ROM implementation."""
from typing import Dict, Any, Optional
import numpy as np

try:
    import ezyrb
    from ezyrb import Database, POD, RBF, GPR, ANN, ReducedOrderModel
    EZYRB_AVAILABLE = True
except ImportError:
    EZYRB_AVAILABLE = False
    ezyrb = None
    Database = None
    POD = None
    RBF = None
    GPR = None
    ANN = None
    ReducedOrderModel = None

from .base import ModelAdapter


class EZyRBROM(ModelAdapter):
    """EZyRB (Easy Reduced Basis) Reduced-Order Model.
    
    Data-driven parametric ROM that works with black-box simulations.
    Supports POD (Proper Orthogonal Decomposition) for basis reduction
    and various interpolation methods (RBF, GPR, ANN) for parameter space.
    """
    
    def __init__(
        self,
        model_id: str,
        name: str,
        pod_rank: Optional[int] = None,
        interpolator_type: str = "RBF",
        fit_data: Optional[np.ndarray] = None,
        parameters: Optional[np.ndarray] = None
    ):
        """
        Initialize EZyRB ROM.
        
        Args:
            model_id: Unique identifier
            name: Human-readable name
            pod_rank: Rank for POD reduction (number of modes). If None, uses all modes.
            interpolator_type: Type of interpolator ("RBF", "GPR", or "ANN")
            fit_data: Training snapshots [n_features, n_samples] for fitting
            parameters: Parameter values [n_params, n_samples] corresponding to snapshots
        """
        super().__init__(model_id, name, "ROM")
        
        if not EZYRB_AVAILABLE:
            raise ImportError(
                "EZyRB is required. Install with: pip install ezyrb"
            )
        
        self.pod_rank = pod_rank
        self.interpolator_type = interpolator_type
        self.database = None
        self.pod = None
        self.interpolator = None
        self.rom = None  # EZyRB ReducedOrderModel
        self.reduced_basis = None
        self._input_shape = None
        self._output_shape = None
        self._parameter_shape = None
        
        if fit_data is not None:
            if parameters is None:
                raise ValueError("Parameters must be provided when fitting with data")
            self.fit(fit_data, parameters)
    
    def fit(self, snapshots: np.ndarray, parameters: np.ndarray) -> None:
        """
        Fit EZyRB model from snapshot data and parameters.
        
        Args:
            snapshots: Snapshot matrix [n_features, n_samples]
            parameters: Parameter matrix [n_params, n_samples] or [n_samples] for scalar params
        """
        if not EZYRB_AVAILABLE:
            raise ImportError("EZyRB is required")
        
        if snapshots.ndim != 2:
            raise ValueError("Snapshots must be 2D array [n_features, n_samples]")
        
        if parameters.ndim == 1:
            parameters = parameters.reshape(1, -1)
        elif parameters.ndim != 2:
            raise ValueError("Parameters must be 1D or 2D array")
        
        n_features, n_samples = snapshots.shape
        n_params, n_param_samples = parameters.shape
        
        if n_samples != n_param_samples:
            raise ValueError(f"Number of snapshots ({n_samples}) must match number of parameter samples ({n_param_samples})")
        
        # EZyRB Database expects:
        # - parameters: list of parameter tuples/arrays, one per snapshot
        # - snapshots: list of snapshot arrays, one per snapshot
        # So len(parameters) == len(snapshots) == n_samples
        
        # Convert to list of parameter tuples (one per sample)
        param_list = []
        snapshots_list = []
        
        for i in range(n_samples):
            # Parameter tuple for this sample
            param_tuple = tuple(parameters[j, i] for j in range(n_params))
            param_list.append(param_tuple)
            
            # Snapshot for this sample
            snapshots_list.append(snapshots[:, i])
        
        self.database = Database(param_list, snapshots_list)
        
        # Create POD reducer
        if self.pod_rank is not None:
            self.pod = POD(rank=self.pod_rank)
        else:
            self.pod = POD()
        
        # Create interpolator
        if self.interpolator_type.upper() == "RBF":
            self.interpolator = RBF()
        elif self.interpolator_type.upper() == "GPR":
            self.interpolator = GPR()
        elif self.interpolator_type.upper() == "ANN":
            self.interpolator = ANN()
        else:
            raise ValueError(f"Unknown interpolator type: {self.interpolator_type}. Use 'RBF', 'GPR', or 'ANN'")
        
        # Build reduced order model using EZyRB's ReducedOrderModel
        self.rom = ReducedOrderModel(self.database, self.pod, self.interpolator)
        
        # Fit the ROM (if it has a fit method)
        if hasattr(self.rom, 'fit'):
            self.rom.fit()
        
        # Store shapes
        self._input_shape = (n_params,)
        self._output_shape = (n_features,)
        self._parameter_shape = (n_params,)
    
    def simulate(self, input_params: Dict[str, Any]) -> np.ndarray:
        """
        Run EZyRB ROM simulation.
        
        Args:
            input_params: Dictionary with:
                - "mu": Parameter values [n_params] or scalar if n_params=1
                - "t": Time array (optional, for time-dependent problems)
        
        Returns:
            Reconstructed solution [n_features] or [n_features, n_steps] if time provided
        """
        if self.rom is None:
            raise RuntimeError("EZyRB model not fitted. Call fit() first.")
        
        # Get parameters
        mu = input_params.get("mu")
        if mu is None:
            raise ValueError("EZyRB simulation requires 'mu' parameter values")
        
        mu = np.array(mu)
        if mu.ndim == 0:
            mu = mu.reshape(1)
        elif mu.ndim > 1:
            mu = mu.flatten()
        
        # Check parameter dimension
        if len(mu) != self._parameter_shape[0]:
            raise ValueError(
                f"Parameter dimension mismatch: expected {self._parameter_shape[0]}, got {len(mu)}"
            )
        
        # Convert to tuple format for EZyRB
        mu_tuple = tuple(mu)
        
        # Get time array if provided
        t = input_params.get("t")
        if t is not None:
            t = np.array(t)
            n_steps = len(t)
            
            # For time-dependent problems, interpolate at each time step
            # EZyRB is parametric, so time would need to be included as a parameter
            # For now, just use mu (assuming time is handled separately)
            results = []
            for ti in t:
                # Use mu for prediction (time would need to be part of parameter space)
                result_db = self.rom.predict([mu_tuple])
                # Extract values from Database _pairs
                if hasattr(result_db, '_pairs') and len(result_db._pairs) > 0:
                    pair = result_db._pairs[0]
                    if len(pair) >= 2:
                        snapshot = pair[1]
                        if hasattr(snapshot, 'values'):
                            results.append(np.array(snapshot.values).flatten())
                        elif hasattr(snapshot, 'array'):
                            results.append(np.array(snapshot.array).flatten())
                        elif isinstance(snapshot, np.ndarray):
                            results.append(snapshot.flatten())
                        else:
                            results.append(np.array(snapshot).flatten())
                    else:
                        raise RuntimeError("Could not extract snapshot from EZyRB prediction")
                else:
                    raise RuntimeError("EZyRB prediction returned empty database")
            
            return np.array(results).T  # [n_features, n_steps]
        else:
            # Single prediction
            result_db = self.rom.predict([mu_tuple])
            # EZyRB predict returns a Database object
            # Extract the snapshot values from _pairs
            if hasattr(result_db, '_pairs') and len(result_db._pairs) > 0:
                # Get snapshot from first pair (pair is (parameter, snapshot))
                pair = result_db._pairs[0]
                if len(pair) >= 2:
                    snapshot = pair[1]  # Second element is the Snapshot
                    if hasattr(snapshot, 'values'):
                        return np.array(snapshot.values).flatten()
                    elif hasattr(snapshot, 'array'):
                        return np.array(snapshot.array).flatten()
                    elif isinstance(snapshot, np.ndarray):
                        return snapshot.flatten()
            
            # Fallback: try to access as list
            if len(result_db) > 0:
                item = result_db[0]
                if hasattr(item, '_pairs') and len(item._pairs) > 0:
                    pair = item._pairs[0]
                    if len(pair) >= 2:
                        snapshot = pair[1]
                        if hasattr(snapshot, 'values'):
                            return np.array(snapshot.values).flatten()
            
            raise RuntimeError("Could not extract snapshot values from EZyRB prediction")
    
    def metadata(self) -> Dict[str, Any]:
        """Get EZyRB ROM metadata."""
        pod_rank = self.pod_rank
        if pod_rank is None and self.pod is not None:
            # Get actual rank from POD
            try:
                pod_rank = self.pod.rank if hasattr(self.pod, 'rank') else None
            except:
                pass
        
        return {
            "id": self.model_id,
            "name": self.name,
            "type": self.model_type,
            "method": "EZyRB",
            "pod_rank": pod_rank,
            "interpolator_type": self.interpolator_type,
            "input_shape": self._input_shape,
            "output_shape": self._output_shape,
            "parameter_shape": self._parameter_shape,
            "ezyrb_available": EZYRB_AVAILABLE,
        }
