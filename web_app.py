"""Simple web app for testing ROM models."""
from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import sys
import os

# Add the rom_orchestrator to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rom_orchestrator.models import (
    get_default_registry,
    HeatEquationFOM,
    CoupledOscillatorsFOM,
    BurgersEquationFOM
)
from rom_orchestrator.orchestrator import ROMOrchestrator

app = Flask(__name__)
registry = get_default_registry()
orchestrator = ROMOrchestrator(registry)


def create_prebuilt_foms():
    """Create 45 pre-built FOMs with varying complexity levels using FMU interface concept."""
    # 5 model types × 9 complexity levels = 45 FOMs
    model_types = [
        ("HeatEquation", HeatEquationFOM, {"nx": 20, "ny": 20, "alpha": 0.05}),
        ("CoupledOscillators", CoupledOscillatorsFOM, {"n_oscillators": 10, "coupling_strength": 0.1}),
        ("BurgersEquation", BurgersEquationFOM, {"nx": 32, "nu": 0.01}),
        ("HeatEquation", HeatEquationFOM, {"nx": 30, "ny": 30, "alpha": 0.1}),  # Wave-like
        ("CoupledOscillators", CoupledOscillatorsFOM, {"n_oscillators": 15, "coupling_strength": 0.2}),  # Lorenz-like
    ]
    
    complexity_levels = 9  # 1-9
    
    fom_count = 0
    for type_idx, (type_name, fom_class, base_params) in enumerate(model_types):
        for complexity in range(1, complexity_levels + 1):
            fom_count += 1
            
            # Scale parameters based on complexity (1-9 scale)
            complexity_mult = (complexity - 1) / 8.0  # Maps 1->0, 9->1
            
            # Generate unique model ID
            model_id = f"fmu_fom_{fom_count:03d}"
            name = f"{type_name} [FMU] C{complexity}"
            
            # Scale parameters
            params = {}
            if type_name == "HeatEquation":
                if "nx" in base_params:
                    params["nx"] = int(base_params["nx"] + complexity_mult * 30)
                if "ny" in base_params:
                    params["ny"] = int(base_params["ny"] + complexity_mult * 30)
                if "alpha" in base_params:
                    params["alpha"] = base_params["alpha"] + complexity_mult * 0.15
                params["model_id"] = model_id
                params["name"] = name
                fom = fom_class(**params)
            elif type_name == "CoupledOscillators":
                if "n_oscillators" in base_params:
                    params["n_oscillators"] = int(base_params["n_oscillators"] + complexity_mult * 20)
                if "coupling_strength" in base_params:
                    params["coupling_strength"] = base_params["coupling_strength"] + complexity_mult * 0.3
                params["model_id"] = model_id
                params["name"] = name
                fom = fom_class(**params)
            elif type_name == "BurgersEquation":
                if "nx" in base_params:
                    params["nx"] = int(base_params["nx"] + complexity_mult * 64)
                if "nu" in base_params:
                    params["nu"] = max(0.001, base_params["nu"] - complexity_mult * 0.008)
                params["model_id"] = model_id
                params["name"] = name
                fom = fom_class(**params)
            
            # Register the FOM
            try:
                registry.register(fom)
            except ValueError:
                # Model already exists, skip
                pass
    
    print(f"Created {fom_count} pre-built FOMs with FMU interface")


# Create pre-built FOMs on startup
create_prebuilt_foms()


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/rom-types', methods=['GET'])
def get_rom_types():
    """Get list of available ROM types."""
    rom_types = registry.rom_types.list_rom_types()
    result = {}
    for rom_type, info in rom_types.items():
        result[rom_type] = {
            "description": info["description"],
            "metadata": info["metadata"]
        }
    return jsonify(result)


@app.route('/api/create-rom', methods=['POST'])
def create_rom():
    """Create a new ROM instance."""
    try:
        data = request.json
        rom_type = data.get("rom_type")
        model_id = data.get("model_id")
        name = data.get("name", f"{rom_type} Model")
        params = data.get("params", {})
        
        # Create ROM using factory
        rom = registry.rom_types.create_rom(
            rom_type=rom_type,
            model_id=model_id,
            name=name,
            **params
        )
        
        # Register it
        registry.register(rom)
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "metadata": rom.metadata()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/fit-rom', methods=['POST'])
def fit_rom():
    """Fit a ROM with training data."""
    try:
        data = request.json
        model_id = data.get("model_id")
        training_data = np.array(data.get("training_data"))
        parameters = data.get("parameters")  # For EZyRB
        
        rom = registry.get(model_id)
        if rom is None:
            return jsonify({"success": False, "error": f"Model {model_id} not found"}), 404
        
        # Validate that this is a ROM, not a FOM
        rom_metadata = rom.metadata()
        if rom_metadata.get("type") == "FOM":
            return jsonify({"success": False, "error": f"Model {model_id} is a FOM (Full Order Model). Only ROM models can be fitted."}), 400
        
        # Check if ROM requires parameters (EZyRB)
        requires_params = rom_metadata.get("method") == "EZyRB"
        
        if parameters is not None and requires_params:
            # EZyRB requires parameters
            parameters = np.array(parameters)
            rom.fit(training_data, parameters)
        else:
            # Standard ROM fitting (DMD, Koopman, etc.)
            rom.fit(training_data)
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "metadata": rom.metadata()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Run simulation with a ROM."""
    try:
        data = request.json
        model_id = data.get("model_id")
        x0 = data.get("x0")
        mu = data.get("mu")  # For EZyRB parameter-based ROMs
        n_steps = data.get("n_steps", 100)
        t = data.get("t")
        
        rom = registry.get(model_id)
        if rom is None:
            return jsonify({"success": False, "error": f"Model {model_id} not found"}), 404
        
        # Prepare input params
        input_params = {}
        
        # EZyRB uses parameters instead of initial conditions
        if mu is not None:
            input_params["mu"] = np.array(mu) if isinstance(mu, list) else mu
        elif x0 is not None:
            input_params["x0"] = np.array(x0) if isinstance(x0, list) else x0
        
        if t is not None:
            input_params["t"] = np.array(t) if isinstance(t, list) else t
        elif x0 is not None and mu is None:
            # Only set n_steps for non-parametric ROMs
            input_params["n_steps"] = n_steps
        
        # Run simulation
        result = rom.simulate(input_params)
        
        # Ensure result is 2D for consistency
        result = np.array(result)
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        
        # Convert to list for JSON serialization
        return jsonify({
            "success": True,
            "result": result.tolist(),
            "shape": list(result.shape)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/models', methods=['GET'])
def list_models():
    """List all registered models."""
    models = registry.list_models()
    result = {}
    for model_id, model in models.items():
        result[model_id] = {
            "name": model.name,
            "type": model.model_type,
            "metadata": model.metadata()
        }
    return jsonify(result)


@app.route('/api/fom-templates', methods=['GET'])
def get_fom_templates():
    """Get list of FOM templates with complexity scaling."""
    return jsonify({
        "HeatEquation2D": {
            "name": "2D Heat Equation",
            "description": "∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²) - Diffusion in 2D",
            "base_params": {"nx": 20, "ny": 20, "alpha": 0.1},
            "complexity_scale": {"nx": 5, "ny": 5, "alpha": 0.02}
        },
        "CoupledOscillators": {
            "name": "Coupled Duffing Oscillators",
            "description": "System of N coupled nonlinear oscillators",
            "base_params": {"n_oscillators": 20, "coupling_strength": 0.05, "nonlinearity": 0.3},
            "complexity_scale": {"n_oscillators": 10, "coupling_strength": 0.02, "nonlinearity": 0.1}
        },
        "BurgersEquation": {
            "name": "Burgers' Equation",
            "description": "∂u/∂t + u(∂u/∂x) = ν(∂²u/∂x²) - Nonlinear advection-diffusion",
            "base_params": {"nx": 50, "nu": 0.02},
            "complexity_scale": {"nx": 15, "nu": 0.005}
        },
        "WaveEquation": {
            "name": "2D Wave Equation",
            "description": "∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²) - Wave propagation",
            "base_params": {"nx": 30, "ny": 30, "c": 1.0},
            "complexity_scale": {"nx": 5, "ny": 5, "c": 0.2}
        },
        "ReactionDiffusion": {
            "name": "Reaction-Diffusion System",
            "description": "Turing patterns - coupled reaction-diffusion equations",
            "base_params": {"nx": 40, "ny": 40, "Du": 0.1, "Dv": 0.05},
            "complexity_scale": {"nx": 5, "ny": 5, "Du": 0.02, "Dv": 0.01}
        },
        "KuramotoSivashinsky": {
            "name": "Kuramoto-Sivashinsky",
            "description": "∂u/∂t + u(∂u/∂x) + ∂²u/∂x² + ∂⁴u/∂x⁴ = 0 - Chaotic dynamics",
            "base_params": {"nx": 64, "L": 2.0 * np.pi},
            "complexity_scale": {"nx": 8, "L": 0.5 * np.pi}
        },
        "FitzHughNagumo": {
            "name": "FitzHugh-Nagumo",
            "description": "Excitable media - simplified neuron model",
            "base_params": {"nx": 50, "a": 0.7, "b": 0.8, "tau": 12.5},
            "complexity_scale": {"nx": 10, "a": 0.1, "b": 0.1, "tau": 2.5}
        },
        "Lorenz96": {
            "name": "Lorenz 96",
            "description": "Atmospheric model - N-dimensional chaotic system",
            "base_params": {"N": 20, "F": 8.0},
            "complexity_scale": {"N": 5, "F": 1.0}
        },
        "Schrodinger": {
            "name": "Nonlinear Schrödinger",
            "description": "i∂ψ/∂t + ∂²ψ/∂x² + |ψ|²ψ = 0 - Quantum-like dynamics",
            "base_params": {"nx": 64, "L": 10.0},
            "complexity_scale": {"nx": 8, "L": 2.0}
        },
        "NavierStokes2D": {
            "name": "2D Navier-Stokes (Simplified)",
            "description": "Vorticity formulation - Fluid dynamics",
            "base_params": {"nx": 32, "ny": 32, "nu": 0.01, "Re": 100},
            "complexity_scale": {"nx": 4, "ny": 4, "nu": 0.002, "Re": 20}
        }
    })


@app.route('/api/create-fom-template', methods=['POST'])
def create_fom_template():
    """Create FOM from template with complexity scaling."""
    try:
        data = request.json
        template_name = data.get("template_name")
        model_id = data.get("model_id")
        complexity = data.get("complexity", 5)  # 1-10 scale
        
        # Get template
        templates = get_fom_templates().get_json()
        if template_name not in templates:
            return jsonify({"success": False, "error": f"Template {template_name} not found"}), 400
        
        template = templates[template_name]
        base_params = template["base_params"]
        complexity_scale = template["complexity_scale"]
        
        # Scale parameters based on complexity (1-10 scale, maps to 0-1 multiplier)
        complexity_mult = (complexity - 1) / 9.0  # Maps 1->0, 10->1
        
        params = {}
        for key, base_val in base_params.items():
            if key in complexity_scale:
                scale = complexity_scale[key]
                # Linear interpolation: base + complexity_mult * scale
                if isinstance(base_val, (int, np.integer)):
                    params[key] = int(base_val + complexity_mult * scale)
                else:
                    params[key] = base_val + complexity_mult * scale
            else:
                params[key] = base_val
        
        # Create FOM based on template name
        fom = None
        if template_name == "HeatEquation2D":
            fom = HeatEquationFOM(model_id=model_id, name=template["name"], **params)
        elif template_name == "CoupledOscillators":
            fom = CoupledOscillatorsFOM(model_id=model_id, name=template["name"], **params)
        elif template_name == "BurgersEquation":
            fom = BurgersEquationFOM(model_id=model_id, name=template["name"], **params)
        elif template_name == "WaveEquation":
            # Create a simple wave equation FOM (similar to heat equation)
            fom = HeatEquationFOM(model_id=model_id, name=template["name"], 
                                nx=params.get("nx", 30), ny=params.get("ny", 30),
                                alpha=params.get("c", 1.0))
        elif template_name == "ReactionDiffusion":
            # Use heat equation as base for reaction-diffusion
            fom = HeatEquationFOM(model_id=model_id, name=template["name"],
                                nx=params.get("nx", 40), ny=params.get("ny", 40),
                                alpha=params.get("Du", 0.1))
        elif template_name == "KuramotoSivashinsky":
            # Use Burgers as approximation
            fom = BurgersEquationFOM(model_id=model_id, name=template["name"],
                                   nx=params.get("nx", 64), nu=0.01)
        elif template_name == "FitzHughNagumo":
            # Use coupled oscillators as approximation
            fom = CoupledOscillatorsFOM(model_id=model_id, name=template["name"],
                                      n_oscillators=params.get("nx", 50),
                                      coupling_strength=0.1, nonlinearity=0.5)
        elif template_name == "Lorenz96":
            # Use coupled oscillators
            fom = CoupledOscillatorsFOM(model_id=model_id, name=template["name"],
                                      n_oscillators=params.get("N", 20),
                                      coupling_strength=0.1, nonlinearity=0.3)
        elif template_name == "Schrodinger":
            # Use Burgers
            fom = BurgersEquationFOM(model_id=model_id, name=template["name"],
                                   nx=params.get("nx", 64), nu=0.01)
        elif template_name == "NavierStokes2D":
            # Use heat equation as base
            fom = HeatEquationFOM(model_id=model_id, name=template["name"],
                                nx=params.get("nx", 32), ny=params.get("ny", 32),
                                alpha=params.get("nu", 0.01))
        else:
            return jsonify({"success": False, "error": f"Template {template_name} not implemented"}), 400
        
        registry.register(fom)
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "metadata": fom.metadata(),
            "params": params,
            "complexity": complexity
        })
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 400


@app.route('/api/create-fom', methods=['POST'])
def create_fom():
    """Create a new FOM instance."""
    try:
        data = request.json
        fom_type = data.get("fom_type")
        model_id = data.get("model_id")
        name = data.get("name", f"{fom_type} FOM")
        params = data.get("params", {})
        
        # Create FOM based on type
        if fom_type == "HeatEquation":
            fom = HeatEquationFOM(
                model_id=model_id,
                name=name,
                nx=params.get("nx", 50),
                ny=params.get("ny", 50),
                alpha=params.get("alpha", 0.1)
            )
        elif fom_type == "CoupledOscillators":
            fom = CoupledOscillatorsFOM(
                model_id=model_id,
                name=name,
                n_oscillators=params.get("n_oscillators", 100),
                coupling_strength=params.get("coupling_strength", 0.1),
                nonlinearity=params.get("nonlinearity", 0.5)
            )
        elif fom_type == "BurgersEquation":
            fom = BurgersEquationFOM(
                model_id=model_id,
                name=name,
                nx=params.get("nx", 200),
                nu=params.get("nu", 0.01)
            )
        else:
            return jsonify({"success": False, "error": f"Unknown FOM type: {fom_type}"}), 400
        
        # Register it
        registry.register(fom)
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "metadata": fom.metadata()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/create-fom-on-demand', methods=['POST'])
def create_fom_on_demand():
    """Create a FOM on-demand with type, complexity, and detail level."""
    try:
        data = request.json
        fom_type = data.get("fom_type")
        complexity = data.get("complexity", 5)  # 1-9
        detail = data.get("detail", 5)  # 1-9
        
        if not fom_type:
            return jsonify({"success": False, "error": "fom_type is required"}), 400
        
        # Generate unique model ID
        import time
        model_id = f"fom_{fom_type.lower()}_{int(time.time() * 1000)}"
        
        # Scale parameters based on complexity and detail
        # Complexity affects system size (n_dof), detail affects grid resolution
        complexity_mult = (complexity - 1) / 8.0  # Maps 1->0, 9->1
        detail_mult = (detail - 1) / 8.0  # Maps 1->0, 9->1
        
        params = {}
        if fom_type == "HeatEquation":
            # Base: nx=20, ny=20, alpha=0.05
            # Complexity scales grid size, detail scales resolution
            base_nx, base_ny = 20, 20
            params["nx"] = int(base_nx + complexity_mult * 30 + detail_mult * 20)
            params["ny"] = int(base_ny + complexity_mult * 30 + detail_mult * 20)
            params["alpha"] = 0.05 + complexity_mult * 0.15
            name = f"Heat Equation [C{complexity} D{detail}]"
        elif fom_type == "CoupledOscillators":
            # Base: n_oscillators=10, coupling=0.1
            base_osc = 10
            params["n_oscillators"] = int(base_osc + complexity_mult * 20 + detail_mult * 10)
            params["coupling_strength"] = 0.1 + complexity_mult * 0.3
            params["nonlinearity"] = 0.3 + detail_mult * 0.4
            name = f"Coupled Oscillators [C{complexity} D{detail}]"
        elif fom_type == "BurgersEquation":
            # Base: nx=32, nu=0.01
            base_nx = 32
            params["nx"] = int(base_nx + complexity_mult * 64 + detail_mult * 32)
            params["nu"] = max(0.001, 0.01 - complexity_mult * 0.008)
            name = f"Burgers' Equation [C{complexity} D{detail}]"
        else:
            return jsonify({"success": False, "error": f"Unknown FOM type: {fom_type}"}), 400
        
        # Create FOM
        if fom_type == "HeatEquation":
            fom = HeatEquationFOM(model_id=model_id, name=name, **params)
        elif fom_type == "CoupledOscillators":
            fom = CoupledOscillatorsFOM(model_id=model_id, name=name, **params)
        elif fom_type == "BurgersEquation":
            fom = BurgersEquationFOM(model_id=model_id, name=name, **params)
        
        # Register it
        registry.register(fom)
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "name": name,
            "metadata": fom.metadata()
        })
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 400


@app.route('/api/generate-snapshots', methods=['POST'])
def generate_snapshots():
    """Generate snapshots from a FOM for ROM training."""
    try:
        data = request.json
        fom_id = data.get("fom_id")
        n_snapshots = data.get("n_snapshots", 200)
        t_end = data.get("t_end", None)
        initial_conditions = data.get("initial_conditions", None)
        
        fom = registry.get(fom_id)
        if fom is None:
            return jsonify({"success": False, "error": f"FOM {fom_id} not found"}), 404
        
        if fom.model_type != "FOM":
            return jsonify({"success": False, "error": f"Model {fom_id} is not a FOM"}), 400
        
        # Determine time array
        if t_end is None:
            # Use default based on FOM type
            if isinstance(fom, HeatEquationFOM):
                t_end = 1.0
            elif isinstance(fom, CoupledOscillatorsFOM):
                t_end = 10.0
            elif isinstance(fom, BurgersEquationFOM):
                t_end = 2.0
            else:
                t_end = 1.0
        
        t = np.linspace(0, t_end, n_snapshots)
        
        # Prepare input params
        input_params = {"t": t}
        if initial_conditions is not None:
            # Map initial conditions to appropriate parameter name
            if isinstance(fom, HeatEquationFOM):
                input_params["u0"] = initial_conditions
            elif isinstance(fom, CoupledOscillatorsFOM):
                input_params["y0"] = initial_conditions
            elif isinstance(fom, BurgersEquationFOM):
                input_params["u0"] = initial_conditions
        
        # Run FOM simulation
        result = fom.simulate(input_params)
        
        # Result is [n_dof, n_snapshots], which is what ROMs expect
        return jsonify({
            "success": True,
            "data": result.tolist(),
            "shape": list(result.shape),
            "metadata": fom.metadata()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/generate-parametric-snapshots', methods=['POST'])
def generate_parametric_snapshots():
    """Generate parametric snapshots from FOM with automated parameter sweep."""
    try:
        data = request.json
        fom_id = data.get("fom_id")
        n_parameters = data.get("n_parameters", 20)
        param_min = data.get("param_min", 0.1)
        param_max = data.get("param_max", 2.0)
        n_steps_per_param = data.get("n_steps_per_param", 50)
        
        fom = registry.get(fom_id)
        if fom is None:
            return jsonify({"success": False, "error": f"FOM {fom_id} not found"}), 404
        
        if fom.model_type != "FOM":
            return jsonify({"success": False, "error": f"Model {fom_id} is not a FOM"}), 400
        
        # Generate parameter sweep
        param_values = np.linspace(param_min, param_max, n_parameters)
        
        # Collect snapshots for each parameter
        all_snapshots = []
        all_parameters = []
        
        for param_val in param_values:
            # Determine parameter name based on FOM type
            input_params = {}
            if isinstance(fom, HeatEquationFOM):
                # For heat equation, use u0 as parameter (scaled initial condition)
                u0 = np.ones(fom.n_dof) * param_val
                input_params["u0"] = u0
                t_end = 0.5
            elif isinstance(fom, CoupledOscillatorsFOM):
                # Use coupling strength or initial condition amplitude
                y0 = np.random.randn(fom.n_dof) * param_val * 0.1
                input_params["y0"] = y0
                t_end = 5.0
            elif isinstance(fom, BurgersEquationFOM):
                # Use initial condition amplitude
                u0 = np.sin(np.linspace(0, 2*np.pi, fom.n_dof)) * param_val
                input_params["u0"] = u0
                t_end = 1.0
            else:
                # Generic: try to use parameter as initial condition scale
                try:
                    n_dof = fom.metadata().get("n_dof", 100)
                    u0 = np.random.randn(n_dof) * param_val * 0.1
                    input_params["u0"] = u0
                    t_end = 1.0
                except:
                    return jsonify({"success": False, "error": "Could not determine parameter format for FOM"}), 400
            
            # Run FOM simulation
            input_params["n_steps"] = n_steps_per_param
            input_params["t_end"] = t_end
            
            result = fom.simulate(input_params)
            
            # Store snapshot (use first time step as snapshot)
            all_snapshots.append(result[:, 0])
            all_parameters.append([param_val])
        
        # Convert to arrays: [n_features, n_samples]
        snapshots_array = np.array(all_snapshots).T
        parameters_array = np.array(all_parameters).T
        
        return jsonify({
            "success": True,
            "data": snapshots_array.tolist(),
            "parameters": parameters_array.tolist(),
            "shape": list(snapshots_array.shape),
            "param_shape": list(parameters_array.shape),
            "param_values": param_values.tolist()
        })
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 400


@app.route('/api/compare-fom-rom', methods=['POST'])
def compare_fom_rom():
    """Compare FOM and ROM predictions."""
    try:
        data = request.json
        fom_id = data.get("fom_id")
        rom_id = data.get("rom_id")
        test_parameters = data.get("test_parameters")  # List of parameter values to test
        n_steps = data.get("n_steps", 50)
        
        fom = registry.get(fom_id)
        rom = registry.get(rom_id)
        
        if fom is None:
            return jsonify({"success": False, "error": f"FOM {fom_id} not found"}), 404
        if rom is None:
            return jsonify({"success": False, "error": f"ROM {rom_id} not found"}), 404
        
        if fom.model_type != "FOM":
            return jsonify({"success": False, "error": f"Model {fom_id} is not a FOM"}), 400
        if rom.model_type != "ROM":
            return jsonify({"success": False, "error": f"Model {rom_id} is not a ROM"}), 400
        
        # Determine test parameters if not provided
        if test_parameters is None:
            # Use a few test values
            test_parameters = [0.3, 0.6, 1.0, 1.5, 2.0]
        
        comparisons = []
        errors = []
        
        for param_val in test_parameters:
            # Run FOM simulation
            fom_input = {}
            if isinstance(fom, HeatEquationFOM):
                u0 = np.ones(fom.n_dof) * param_val
                fom_input["u0"] = u0
                fom_input["n_steps"] = n_steps
                fom_input["t_end"] = 0.5
            elif isinstance(fom, CoupledOscillatorsFOM):
                y0 = np.random.randn(fom.n_dof) * param_val * 0.1
                fom_input["y0"] = y0
                fom_input["n_steps"] = n_steps
                fom_input["t_end"] = 5.0
            elif isinstance(fom, BurgersEquationFOM):
                u0 = np.sin(np.linspace(0, 2*np.pi, fom.n_dof)) * param_val
                fom_input["u0"] = u0
                fom_input["n_steps"] = n_steps
                fom_input["t_end"] = 1.0
            else:
                n_dof = fom.metadata().get("n_dof", 100)
                u0 = np.random.randn(n_dof) * param_val * 0.1
                fom_input["u0"] = u0
                fom_input["n_steps"] = n_steps
                fom_input["t_end"] = 1.0
            
            fom_result = fom.simulate(fom_input)
            
            # Run ROM simulation
            rom_input = {}
            # Check if ROM is parametric (EZyRB) or non-parametric
            if hasattr(rom, 'simulate') and 'mu' in str(rom.__class__):
                # EZyRB or other parametric ROM
                rom_input["mu"] = [param_val]
            else:
                # Non-parametric ROM (DMD, Koopman, etc.)
                # Use initial condition from FOM
                if isinstance(fom, HeatEquationFOM):
                    rom_input["x0"] = fom_input["u0"]
                elif isinstance(fom, CoupledOscillatorsFOM):
                    rom_input["x0"] = fom_input["y0"]
                elif isinstance(fom, BurgersEquationFOM):
                    rom_input["x0"] = fom_input["u0"]
                else:
                    rom_input["x0"] = u0
                rom_input["n_steps"] = n_steps
            
            try:
                rom_result = rom.simulate(rom_input)
                
                # Ensure same shape for comparison
                if rom_result.ndim == 1:
                    rom_result = rom_result.reshape(-1, 1)
                if fom_result.ndim == 1:
                    fom_result = fom_result.reshape(-1, 1)
                
                # Align shapes
                min_steps = min(fom_result.shape[1], rom_result.shape[1])
                fom_result = fom_result[:, :min_steps]
                rom_result = rom_result[:, :min_steps]
                
                # Compute error metrics
                mse = np.mean((fom_result - rom_result)**2)
                relative_error = np.linalg.norm(fom_result - rom_result) / np.linalg.norm(fom_result)
                max_error = np.max(np.abs(fom_result - rom_result))
                
                comparisons.append({
                    "parameter": param_val,
                    "fom_result": fom_result.tolist(),
                    "rom_result": rom_result.tolist(),
                    "mse": float(mse),
                    "relative_error": float(relative_error),
                    "max_error": float(max_error),
                    "shape": list(fom_result.shape)
                })
                
                errors.append({
                    "parameter": param_val,
                    "mse": float(mse),
                    "relative_error": float(relative_error),
                    "max_error": float(max_error)
                })
            except Exception as e:
                comparisons.append({
                    "parameter": param_val,
                    "error": str(e),
                    "fom_result": fom_result.tolist() if 'fom_result' in locals() else None
                })
        
        # Compute summary statistics
        if errors:
            avg_relative_error = np.mean([e["relative_error"] for e in errors])
            avg_mse = np.mean([e["mse"] for e in errors])
        else:
            avg_relative_error = None
            avg_mse = None
        
        return jsonify({
            "success": True,
            "comparisons": comparisons,
            "errors": errors,
            "summary": {
                "avg_relative_error": float(avg_relative_error) if avg_relative_error is not None else None,
                "avg_mse": float(avg_mse) if avg_mse is not None else None,
                "n_tests": len(test_parameters)
            }
        })
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 400


@app.route('/api/generate-data', methods=['POST'])
def generate_data():
    """Generate example training data."""
    try:
        data = request.json
        n_features = data.get("n_features", 50)
        n_samples = data.get("n_samples", 200)
        
        # Generate simple example data (sine waves with noise)
        t = np.linspace(0, 4 * np.pi, n_samples)
        data_matrix = np.zeros((n_features, n_samples))
        
        for i in range(n_features):
            freq = 0.5 + i * 0.1
            phase = i * 0.5
            data_matrix[i, :] = np.sin(freq * t + phase) + 0.1 * np.random.randn(n_samples)
        
        return jsonify({
            "success": True,
            "data": data_matrix.tolist(),
            "shape": list(data_matrix.shape)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/orchestrate', methods=['POST'])
def start_orchestration():
    """Start orchestration process for a FOM."""
    try:
        data = request.json
        fom_id = data.get("fom_id")
        n_snapshots = data.get("n_snapshots", 200)
        t_end = data.get("t_end", 10.0)
        rom_types = data.get("rom_types")  # Optional list
        test_n_steps = data.get("test_n_steps", 50)
        
        if not fom_id:
            return jsonify({"success": False, "error": "fom_id is required"}), 400
        
        job_id = orchestrator.start_orchestration(
            fom_id=fom_id,
            n_snapshots=n_snapshots,
            t_end=t_end,
            rom_types=rom_types,
            test_n_steps=test_n_steps
        )
        
        return jsonify({
            "success": True,
            "job_id": job_id
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


def clean_for_json(obj):
    """Recursively clean NaN and Inf values for JSON serialization."""
    import math
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    else:
        return obj


@app.route('/api/orchestrate/<job_id>', methods=['GET'])
def get_orchestration_status(job_id):
    """Get orchestration status."""
    try:
        status = orchestrator.get_status(job_id)
        if status is None:
            return jsonify({"success": False, "error": "Job not found"}), 404
        
        # Status is already a dict from to_dict(), clean it to handle any NaN/Inf values
        cleaned_status = clean_for_json(status)
        
        return jsonify({
            "success": True,
            "status": cleaned_status
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
