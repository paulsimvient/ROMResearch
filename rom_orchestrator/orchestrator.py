"""Threaded orchestration tool for automatic ROM generation and comparison."""
import threading
import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime


@dataclass
class OrchestrationStatus:
    """Status of orchestration process."""
    status: str  # "pending", "running", "completed", "error"
    progress: float  # 0.0 to 1.0
    current_step: str
    fom_id: str
    rom_results: Dict[str, Any]
    rom_progress: Dict[str, Dict[str, Any]] = None  # Per-ROM progress tracking
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def __post_init__(self):
        """Initialize rom_progress if not provided."""
        if self.rom_progress is None:
            self.rom_progress = {}
    
    def _clean_nan(self, obj):
        """Recursively clean NaN values from dictionary/list structures."""
        if isinstance(obj, dict):
            return {k: self._clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_nan(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            # Convert numpy array to list and clean NaN
            arr_list = obj.tolist()
            return self._clean_nan(arr_list)
        elif isinstance(obj, (np.floating, np.integer)):
            # Convert numpy scalar to Python type
            val = float(obj) if isinstance(obj, np.floating) else int(obj)
            return None if (isinstance(val, float) and (np.isnan(val) or np.isinf(val))) else val
        elif isinstance(obj, float):
            # Handle Python float NaN/Inf
            return None if (np.isnan(obj) or np.isinf(obj)) else obj
        else:
            return obj
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert numpy arrays to lists and clean NaN values
        for rom_id, rom_data in result.get("rom_results", {}).items():
            if "snapshots" in rom_data:
                if isinstance(rom_data["snapshots"], np.ndarray):
                    rom_data["snapshots"] = rom_data["snapshots"].tolist()
            if "predictions" in rom_data:
                if isinstance(rom_data["predictions"], np.ndarray):
                    rom_data["predictions"] = rom_data["predictions"].tolist()
            if "errors" in rom_data:
                if isinstance(rom_data["errors"], np.ndarray):
                    rom_data["errors"] = rom_data["errors"].tolist()
        
        # Clean NaN values recursively
        result = self._clean_nan(result)
        return result


class ROMOrchestrator:
    """Orchestrates automatic ROM generation, fitting, and comparison."""
    
    def __init__(self, registry):
        """
        Initialize orchestrator.
        
        Args:
            registry: Model registry instance
        """
        self.registry = registry
        self.active_jobs: Dict[str, OrchestrationStatus] = {}
        self._lock = threading.Lock()
    
    def start_orchestration(
        self,
        fom_id: str,
        n_snapshots: int = 200,
        t_end: float = 10.0,
        rom_types: Optional[List[str]] = None,
        test_n_steps: int = 50
    ) -> str:
        """
        Start orchestration process in background thread.
        
        Args:
            fom_id: FOM model ID to use
            n_snapshots: Number of training snapshots
            t_end: End time for simulation
            rom_types: List of ROM types to create (None = all available)
            test_n_steps: Number of test steps for comparison
        
        Returns:
            Job ID
        """
        job_id = f"job_{int(time.time() * 1000)}"
        
        status = OrchestrationStatus(
            status="pending",
            progress=0.0,
            current_step="Initializing",
            fom_id=fom_id,
            rom_results={},
            start_time=time.time()
        )
        
        with self._lock:
            self.active_jobs[job_id] = status
        
        # Start background thread
        thread = threading.Thread(
            target=self._run_orchestration,
            args=(job_id, fom_id, n_snapshots, t_end, rom_types, test_n_steps),
            daemon=True
        )
        thread.start()
        
        return job_id
    
    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get orchestration status."""
        with self._lock:
            status = self.active_jobs.get(job_id)
            if status:
                return status.to_dict()
            return None
    
    def _run_orchestration(
        self,
        job_id: str,
        fom_id: str,
        n_snapshots: int,
        t_end: float,
        rom_types: Optional[List[str]],
        test_n_steps: int
    ):
        """Run orchestration process."""
        try:
            status = self.active_jobs[job_id]
            status.status = "running"
            
            # Step 1: Get FOM
            status.current_step = f"Loading FOM: {fom_id}"
            status.progress = 0.05
            fom = self.registry.get(fom_id)
            if fom is None:
                raise ValueError(f"FOM {fom_id} not found")
            
            if fom.metadata().get("type") != "FOM":
                raise ValueError(f"Model {fom_id} is not a FOM")
            
            # Step 2: Generate training snapshots
            status.current_step = f"Generating {n_snapshots} training snapshots"
            status.progress = 0.10
            training_snapshots = self._generate_snapshots(fom, n_snapshots, t_end)
            
            # Step 3: Get available ROM types
            status.current_step = "Identifying ROM types"
            status.progress = 0.15
            available_rom_types = rom_types or list(self.registry.rom_types.list_rom_types().keys())
            
            # Initialize progress for all ROM types
            for rom_type in available_rom_types:
                status.rom_progress[rom_type] = {
                    "status": "pending",
                    "progress": 0.0,
                    "step": "Waiting..."
                }
            
            # Step 4: Create and fit ROMs
            # Get snapshot dimensions for ROM creation
            n_features, n_samples = training_snapshots.shape
            
            total_roms = len(available_rom_types)
            rom_results = {}
            
            for idx, rom_type in enumerate(available_rom_types):
                try:
                    # Initialize ROM progress
                    status.rom_progress[rom_type] = {
                        "status": "pending",
                        "progress": 0.0,
                        "step": "Initializing"
                    }
                    
                    status.current_step = f"Creating and fitting {rom_type} ROM ({idx+1}/{total_roms})"
                    status.progress = 0.15 + (idx / total_roms) * 0.60
                    
                    rom_id = f"{fom_id}_{rom_type.lower()}_{int(time.time())}"
                    rom_name = f"{rom_type} ROM for {fom_id}"
                    
                    # Update ROM progress: Creating
                    status.rom_progress[rom_type] = {
                        "status": "running",
                        "progress": 0.1,
                        "step": "Creating ROM"
                    }
                    
                    # Create ROM with proper dimensions
                    rom = self._create_rom(rom_type, rom_id, rom_name, n_features=n_features)
                    if rom is None:
                        rom_results[rom_type] = {
                            "rom_id": rom_id,
                            "status": "error",
                            "error": f"Failed to create {rom_type} ROM"
                        }
                        status.rom_progress[rom_type] = {
                            "status": "error",
                            "progress": 0.0,
                            "step": "Failed to create"
                        }
                        continue
                    
                    # Check if ROM has fit method
                    if not hasattr(rom, 'fit'):
                        rom_results[rom_type] = {
                            "rom_id": rom_id,
                            "status": "skipped",
                            "error": f"{rom_type} ROM does not support automatic fitting"
                        }
                        status.rom_progress[rom_type] = {
                            "status": "skipped",
                            "progress": 0.0,
                            "step": "Skipped (no fit method)"
                        }
                        continue
                    
                    # Update ROM progress: Fitting
                    status.rom_progress[rom_type] = {
                        "status": "running",
                        "progress": 0.3,
                        "step": "Fitting/Training ROM"
                    }
                    
                    # Fit ROM (with training for neural network ROMs)
                    try:
                        self._fit_rom(rom, rom_type, training_snapshots)
                    except Exception as fit_error:
                        # Log the error with shape information
                        error_msg = f"{str(fit_error)} (snapshots shape: {training_snapshots.shape})"
                        raise RuntimeError(error_msg) from fit_error
                    
                    # Update ROM progress: Testing
                    status.rom_progress[rom_type] = {
                        "status": "running",
                        "progress": 0.7,
                        "step": "Testing ROM"
                    }
                    
                    # Step 5: Generate test data and compare
                    status.current_step = f"Testing {rom_type} ROM"
                    try:
                        test_snapshots, rom_predictions, errors = self._compare_fom_rom(
                            fom, rom, test_n_steps, t_end
                        )
                    except Exception as test_error:
                        # If testing fails, still mark ROM as completed but with error info
                        raise RuntimeError(f"Testing failed: {str(test_error)}") from test_error
                    
                    # Calculate metrics
                    mse = float(np.mean(errors ** 2))
                    max_error = float(np.max(np.abs(errors)))
                    mean_error = float(np.mean(np.abs(errors)))
                    
                    rom_results[rom_type] = {
                        "rom_id": rom_id,
                        "status": "completed",
                        "mse": mse,
                        "max_error": max_error,
                        "mean_error": mean_error,
                        "snapshots": test_snapshots[:100].tolist() if len(test_snapshots) > 100 else test_snapshots.tolist(),  # Limit for JSON
                        "predictions": rom_predictions[:100].tolist() if len(rom_predictions) > 100 else rom_predictions.tolist(),
                        "errors": errors[:100].tolist() if len(errors) > 100 else errors.tolist()
                    }
                    
                    # Update ROM progress: Completed
                    status.rom_progress[rom_type] = {
                        "status": "completed",
                        "progress": 1.0,
                        "step": "Completed"
                    }
                    
                except Exception as e:
                    rom_results[rom_type] = {
                        "rom_id": rom_id if 'rom_id' in locals() else "unknown",
                        "status": "error",
                        "error": str(e)
                    }
                    status.rom_progress[rom_type] = {
                        "status": "error",
                        "progress": 0.0,
                        "step": f"Error: {str(e)[:50]}"
                    }
            
            # Step 6: Finalize
            status.current_step = "Orchestration completed"
            status.progress = 1.0
            status.status = "completed"
            status.rom_results = rom_results
            status.end_time = time.time()
            
        except Exception as e:
            with self._lock:
                status = self.active_jobs.get(job_id)
                if status:
                    status.status = "error"
                    status.error = str(e)
                    status.end_time = time.time()
    
    def _generate_snapshots(self, fom, n_snapshots: int, t_end: float) -> np.ndarray:
        """Generate training snapshots from FOM.
        
        Returns:
            Snapshot matrix in [n_features, n_samples] format (standard for DMD/Koopman)
        """
        t = np.linspace(0, t_end, n_snapshots)
        
        # Get initial condition
        meta = fom.metadata()
        n_dof = meta.get("n_dof", 100)
        
        # Generate random initial condition
        u0 = np.random.randn(n_dof) * 0.1
        
        # Run simulation
        result = fom.simulate({
            "u0": u0,
            "t": t,
            "n_steps": n_snapshots
        })
        
        # Ensure result is 2D
        if result.ndim == 1:
            # Single time step, reshape to [n_features, 1]
            result = result.reshape(-1, 1)
        elif result.ndim == 2:
            # Check if we need to transpose
            # FOM typically returns [n_time_steps, n_features] or [n_features, n_time_steps]
            # We want [n_features, n_time_steps] format
            if result.shape[0] < result.shape[1] and result.shape[1] == n_snapshots:
                # Likely [n_features, n_snapshots] - good
                pass
            elif result.shape[0] == n_snapshots and result.shape[1] < result.shape[0]:
                # Likely [n_snapshots, n_features] - transpose
                result = result.T
            # If shape is ambiguous, assume first dim is features if it matches n_dof
            elif result.shape[0] == n_dof:
                # Already [n_features, n_snapshots]
                pass
            elif result.shape[1] == n_dof:
                # [n_snapshots, n_features] - transpose
                result = result.T
        
        # Final check: ensure we have [n_features, n_samples] format
        # where n_features should match n_dof
        if result.shape[0] != n_dof and result.shape[1] == n_dof:
            result = result.T
        
        return result
    
    def _create_rom(self, rom_type: str, rom_id: str, rom_name: str, n_features: int = None):
        """Create a ROM instance.
        
        Args:
            rom_type: Type of ROM to create
            rom_id: Unique identifier
            rom_name: Human-readable name
            n_features: Number of features (for neural network ROMs)
        """
        try:
            # Get default parameters for each ROM type
            params = {}
            if rom_type == "DMD":
                params = {"rank": 10, "implementation": "numpy"}
            elif rom_type == "Koopman":
                params = {"n_modes": 15}
            elif rom_type == "Autoencoder":
                params = {"latent_dim": 20}
                if n_features:
                    params["input_shape"] = (n_features,)
                    params["output_shape"] = (n_features,)
            elif rom_type == "EZyRB":
                params = {"pod_rank": 10, "interpolator_type": "RBF"}
            elif rom_type == "NeuralNetwork":
                params = {}
                if n_features:
                    params["input_shape"] = (n_features,)
                    params["output_shape"] = (n_features,)
            
            rom = self.registry.rom_types.create_rom(
                rom_type=rom_type,
                model_id=rom_id,
                name=rom_name,
                **params
            )
            
            self.registry.register(rom)
            return rom
        except Exception as e:
            print(f"Error creating {rom_type} ROM: {e}")
            return None
    
    def _fit_rom(self, rom, rom_type: str, training_snapshots: np.ndarray):
        """Fit a ROM with training data.
        
        Args:
            rom: ROM instance to fit
            rom_type: ROM type name
            training_snapshots: Snapshot matrix in [n_features, n_samples] format (from _generate_snapshots)
        """
        meta = rom.metadata()
        method = meta.get("method")
        
        # training_snapshots comes in as [n_features, n_samples] from _generate_snapshots
        n_features, n_samples = training_snapshots.shape
        
        if method == "EZyRB":
            # EZyRB requires parameters and expects [n_features, n_samples]
            # Create dummy parameters: [n_samples] format (1 scalar parameter per snapshot)
            parameters = np.array([0.1 + i * 0.01 for i in range(n_samples)])  # Shape: [n_samples]
            rom.fit(training_snapshots, parameters)
        elif method in ["Autoencoder", "NeuralNetwork"]:
            # Neural network ROMs expect [n_samples, n_features] format
            # Transpose: [n_features, n_samples] -> [n_samples, n_features]
            training_snapshots_transposed = training_snapshots.T.copy()  # Use copy to avoid issues
            
            # Verify shape
            if training_snapshots_transposed.shape[1] != n_features:
                raise ValueError(
                    f"Shape mismatch: transposed data has {training_snapshots_transposed.shape[1]} features, "
                    f"expected {n_features}. Original shape: {training_snapshots.shape}"
                )
            
            # Train with reduced epochs for faster orchestration
            # Use fewer epochs for faster completion
            epochs = 20 if rom_type == "Autoencoder" else 30  # Autoencoder takes longer
            rom.fit(training_snapshots_transposed, epochs=epochs, batch_size=32, learning_rate=0.001)
        else:
            # DMD, Koopman expect [n_features, n_samples] format (already correct)
            rom.fit(training_snapshots)
    
    def _compare_fom_rom(self, fom, rom, n_steps: int, t_end: float):
        """Compare FOM and ROM predictions."""
        # Generate test initial condition
        meta = fom.metadata()
        n_dof = meta.get("n_dof", 100)
        u0_test = np.random.randn(n_dof) * 0.1
        
        # Generate test time points
        t_test = np.linspace(0, t_end, n_steps)
        
        # Run FOM simulation
        fom_result = fom.simulate({
            "u0": u0_test,
            "t": t_test,
            "n_steps": n_steps
        })
        
        # Ensure 2D
        if fom_result.ndim == 1:
            fom_result = fom_result.reshape(-1, 1)
        elif fom_result.ndim == 2 and fom_result.shape[0] < fom_result.shape[1]:
            fom_result = fom_result.T
        
        # Run ROM simulation
        rom_meta = rom.metadata()
        if rom_meta.get("method") == "EZyRB":
            # EZyRB uses parameters
            mu = np.array([0.5])  # Test parameter
            rom_result = rom.simulate({"mu": mu})
        else:
            # Use first snapshot as initial condition
            x0 = fom_result[:, 0] if fom_result.shape[0] > fom_result.shape[1] else fom_result[0, :]
            rom_result = rom.simulate({"x0": x0, "t": t_test, "n_steps": n_steps})
        
        # Ensure ROM result is 2D
        if rom_result.ndim == 1:
            rom_result = rom_result.reshape(-1, 1)
        elif rom_result.ndim == 2 and rom_result.shape[0] < rom_result.shape[1]:
            rom_result = rom_result.T
        
        # Align dimensions for comparison
        min_dof = min(fom_result.shape[0], rom_result.shape[0])
        min_steps = min(fom_result.shape[1], rom_result.shape[1])
        
        fom_aligned = fom_result[:min_dof, :min_steps]
        rom_aligned = rom_result[:min_dof, :min_steps]
        
        # Calculate errors
        errors = fom_aligned - rom_aligned
        
        return fom_aligned, rom_aligned, errors
