"""Full Order Model (FOM) implementations for testing ROMs."""
from typing import Dict, Any, Optional
import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from .base import ModelAdapter


class HeatEquationFOM(ModelAdapter):
    """2D Heat Equation FOM: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²).
    
    Discretized on a grid, this creates a high-dimensional system.
    """
    
    def __init__(
        self,
        model_id: str,
        name: str,
        nx: int = 50,
        ny: int = 50,
        alpha: float = 0.1,
        Lx: float = 1.0,
        Ly: float = 1.0
    ):
        """
        Initialize Heat Equation FOM.
        
        Args:
            model_id: Unique identifier
            name: Human-readable name
            nx: Number of grid points in x direction
            ny: Number of grid points in y direction
            alpha: Thermal diffusivity
            Lx: Domain length in x
            Ly: Domain length in y
        """
        super().__init__(model_id, name, "FOM")
        self.nx = nx
        self.ny = ny
        self.alpha = alpha
        self.Lx = Lx
        self.Ly = Ly
        self.n_dof = nx * ny
        
        # Grid spacing
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        
        # Create Laplacian matrix (sparse)
        self._build_laplacian()
    
    def _build_laplacian(self):
        """Build the discrete Laplacian operator."""
        n = self.n_dof
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        
        # 5-point stencil for 2D Laplacian
        # Use simpler approach: build matrix directly
        
        # Initialize with zeros
        from scipy.sparse import lil_matrix
        L = lil_matrix((n, n))
        
        for i in range(ny):
            for j in range(nx):
                idx = i * nx + j
                
                # Diagonal: -4/dx² (assuming dx=dy for simplicity)
                L[idx, idx] = -2.0 / (dx*dx) - 2.0 / (dy*dy)
                
                # X-direction neighbors
                if j > 0:
                    L[idx, idx - 1] = 1.0 / (dx*dx)
                if j < nx - 1:
                    L[idx, idx + 1] = 1.0 / (dx*dx)
                
                # Y-direction neighbors
                if i > 0:
                    L[idx, idx - nx] = 1.0 / (dy*dy)
                if i < ny - 1:
                    L[idx, idx + nx] = 1.0 / (dy*dy)
        
        self.L = L.tocsr()
    
    def _rhs(self, t, u):
        """Right-hand side of ODE: du/dt = alpha * L * u"""
        return (self.alpha * self.L.dot(u)).flatten()
    
    def simulate(self, input_params: Dict[str, Any]) -> np.ndarray:
        """
        Run heat equation simulation.
        
        Args:
            input_params: Dictionary with:
                - "u0": Initial condition [nx*ny] or [nx, ny] (optional, defaults to random)
                - "t": Time array
                - "n_steps": Number of time steps (if t not provided)
                - "t_end": End time (default 1.0)
        
        Returns:
            Solution array [nx*ny, n_steps] or [nx, ny, n_steps]
        """
        # Get time array
        t = input_params.get("t")
        if t is None:
            n_steps = input_params.get("n_steps", 100)
            t_end = input_params.get("t_end", 1.0)
            t = np.linspace(0, t_end, n_steps)
        else:
            t = np.array(t)
        
        # Get initial condition
        u0 = input_params.get("u0")
        if u0 is None:
            # Default: Gaussian initial condition
            x = np.linspace(0, self.Lx, self.nx)
            y = np.linspace(0, self.Ly, self.ny)
            X, Y = np.meshgrid(x, y)
            u0 = np.exp(-((X - 0.3)**2 + (Y - 0.3)**2) / 0.1)
            u0 = u0.flatten()
        else:
            u0 = np.array(u0).flatten()
            if len(u0) != self.n_dof:
                raise ValueError(f"Initial condition must have {self.n_dof} elements")
        
        # Solve ODE system
        sol = solve_ivp(self._rhs, [t[0], t[-1]], u0, t_eval=t, method='RK45')
        
        # Return as [n_dof, n_steps]
        return sol.y
    
    def metadata(self) -> Dict[str, Any]:
        """Get FOM metadata."""
        return {
            "id": self.model_id,
            "name": self.name,
            "type": self.model_type,
            "method": "HeatEquation",
            "nx": self.nx,
            "ny": self.ny,
            "n_dof": self.n_dof,
            "alpha": self.alpha,
            "input_shape": (self.n_dof,),
            "output_shape": (self.n_dof,),
        }


class CoupledOscillatorsFOM(ModelAdapter):
    """Coupled nonlinear oscillators FOM.
    
    System of N coupled Duffing oscillators with nonlinear coupling.
    """
    
    def __init__(
        self,
        model_id: str,
        name: str,
        n_oscillators: int = 100,
        coupling_strength: float = 0.1,
        nonlinearity: float = 0.5
    ):
        """
        Initialize Coupled Oscillators FOM.
        
        Args:
            model_id: Unique identifier
            name: Human-readable name
            n_oscillators: Number of coupled oscillators
            coupling_strength: Strength of coupling between oscillators
            nonlinearity: Nonlinearity parameter
        """
        super().__init__(model_id, name, "FOM")
        self.n_oscillators = n_oscillators
        self.coupling = coupling_strength
        self.nonlinearity = nonlinearity
        self.n_dof = 2 * n_oscillators  # position and velocity for each
    
    def _rhs(self, t, y):
        """Right-hand side: coupled Duffing oscillators."""
        n = self.n_oscillators
        q = y[:n]  # positions
        p = y[n:]  # momenta/velocities
        
        dqdt = p
        
        # Nonlinear restoring force + coupling
        dpdt = np.zeros(n)
        for i in range(n):
            # Duffing oscillator: -q - nonlinearity*q³
            dpdt[i] = -q[i] - self.nonlinearity * q[i]**3
            
            # Coupling to neighbors
            if i > 0:
                dpdt[i] += self.coupling * (q[i-1] - q[i])
            if i < n - 1:
                dpdt[i] += self.coupling * (q[i+1] - q[i])
        
        return np.concatenate([dqdt, dpdt])
    
    def simulate(self, input_params: Dict[str, Any]) -> np.ndarray:
        """
        Run coupled oscillators simulation.
        
        Args:
            input_params: Dictionary with:
                - "y0": Initial condition [2*n_oscillators] (optional)
                - "t": Time array
                - "n_steps": Number of time steps (if t not provided)
                - "t_end": End time (default 10.0)
        
        Returns:
            Solution array [2*n_oscillators, n_steps]
        """
        # Get time array
        t = input_params.get("t")
        if t is None:
            n_steps = input_params.get("n_steps", 200)
            t_end = input_params.get("t_end", 10.0)
            t = np.linspace(0, t_end, n_steps)
        else:
            t = np.array(t)
        
        # Get initial condition
        y0 = input_params.get("y0")
        if y0 is None:
            # Default: random initial conditions
            y0 = np.random.randn(self.n_dof) * 0.1
        else:
            y0 = np.array(y0).flatten()
            if len(y0) != self.n_dof:
                raise ValueError(f"Initial condition must have {self.n_dof} elements")
        
        # Solve ODE system
        sol = solve_ivp(self._rhs, [t[0], t[-1]], y0, t_eval=t, method='RK45')
        
        # Return as [n_dof, n_steps]
        return sol.y
    
    def metadata(self) -> Dict[str, Any]:
        """Get FOM metadata."""
        return {
            "id": self.model_id,
            "name": self.name,
            "type": self.model_type,
            "method": "CoupledOscillators",
            "n_oscillators": self.n_oscillators,
            "n_dof": self.n_dof,
            "coupling_strength": self.coupling,
            "nonlinearity": self.nonlinearity,
            "input_shape": (self.n_dof,),
            "output_shape": (self.n_dof,),
        }


class BurgersEquationFOM(ModelAdapter):
    """1D Burgers' Equation FOM: ∂u/∂t + u(∂u/∂x) = ν(∂²u/∂x²).
    
    Nonlinear PDE that exhibits shock formation.
    """
    
    def __init__(
        self,
        model_id: str,
        name: str,
        nx: int = 200,
        nu: float = 0.01,
        L: float = 2.0 * np.pi
    ):
        """
        Initialize Burgers' Equation FOM.
        
        Args:
            model_id: Unique identifier
            name: Human-readable name
            nx: Number of grid points
            nu: Viscosity coefficient
            L: Domain length
        """
        super().__init__(model_id, name, "FOM")
        self.nx = nx
        self.nu = nu
        self.L = L
        self.n_dof = nx
        
        self.dx = L / (nx - 1)
        self.x = np.linspace(0, L, nx)
        
        # Build differentiation matrices
        self._build_derivatives()
    
    def _build_derivatives(self):
        """Build first and second derivative matrices."""
        n = self.nx
        dx = self.dx
        
        # First derivative (central difference, periodic BCs)
        diag = np.zeros(n)
        upper = np.ones(n - 1) / (2 * dx)
        lower = -np.ones(n - 1) / (2 * dx)
        self.D1 = diags([lower, diag, upper], [-1, 0, 1], shape=(n, n), format='csr')
        
        # Periodic boundary conditions
        self.D1[0, n-1] = -1 / (2 * dx)
        self.D1[n-1, 0] = 1 / (2 * dx)
        
        # Second derivative (central difference, periodic BCs)
        diag = -2 * np.ones(n) / (dx * dx)
        upper = np.ones(n - 1) / (dx * dx)
        lower = np.ones(n - 1) / (dx * dx)
        self.D2 = diags([lower, diag, upper], [-1, 0, 1], shape=(n, n), format='csr')
        
        # Periodic boundary conditions
        self.D2[0, n-1] = 1 / (dx * dx)
        self.D2[n-1, 0] = 1 / (dx * dx)
    
    def _rhs(self, t, u):
        """Right-hand side: -u*du/dx + nu*d²u/dx²"""
        ux = self.D1.dot(u)
        uxx = self.D2.dot(u)
        return -u * ux + self.nu * uxx
    
    def simulate(self, input_params: Dict[str, Any]) -> np.ndarray:
        """
        Run Burgers' equation simulation.
        
        Args:
            input_params: Dictionary with:
                - "u0": Initial condition [nx] (optional)
                - "t": Time array
                - "n_steps": Number of time steps (if t not provided)
                - "t_end": End time (default 2.0)
        
        Returns:
            Solution array [nx, n_steps]
        """
        # Get time array
        t = input_params.get("t")
        if t is None:
            n_steps = input_params.get("n_steps", 200)
            t_end = input_params.get("t_end", 2.0)
            t = np.linspace(0, t_end, n_steps)
        else:
            t = np.array(t)
        
        # Get initial condition
        u0 = input_params.get("u0")
        if u0 is None:
            # Default: sinusoidal initial condition
            u0 = np.sin(self.x)
        else:
            u0 = np.array(u0).flatten()
            if len(u0) != self.n_dof:
                raise ValueError(f"Initial condition must have {self.n_dof} elements")
        
        # Solve ODE system
        sol = solve_ivp(self._rhs, [t[0], t[-1]], u0, t_eval=t, method='RK45')
        
        # Return as [n_dof, n_steps]
        return sol.y
    
    def metadata(self) -> Dict[str, Any]:
        """Get FOM metadata."""
        return {
            "id": self.model_id,
            "name": self.name,
            "type": self.model_type,
            "method": "BurgersEquation",
            "nx": self.nx,
            "n_dof": self.n_dof,
            "nu": self.nu,
            "input_shape": (self.n_dof,),
            "output_shape": (self.n_dof,),
        }
