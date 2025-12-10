"""Automated FOM vs ROM comparison with performance graphs."""
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import time

BASE_URL = "http://127.0.0.1:5000"


def create_fom(fom_type, model_id, name, params):
    """Create a FOM model."""
    response = requests.post(f"{BASE_URL}/api/create-fom", json={
        "fom_type": fom_type,
        "model_id": model_id,
        "name": name,
        "params": params
    })
    return response.json()


def generate_parametric_snapshots(fom_id, n_params=20, param_min=0.1, param_max=2.0):
    """Generate parametric snapshots from FOM."""
    print(f"Generating {n_params} parametric snapshots from FOM {fom_id}...")
    response = requests.post(f"{BASE_URL}/api/generate-parametric-snapshots", json={
        "fom_id": fom_id,
        "n_parameters": n_params,
        "param_min": param_min,
        "param_max": param_max,
        "n_steps_per_param": 50
    })
    result = response.json()
    if result["success"]:
        print(f"  Generated {result['shape'][1]} snapshots ({result['shape'][0]} DOF)")
        return result
    else:
        raise Exception(f"Failed to generate snapshots: {result.get('error')}")


def create_rom(rom_type, model_id, name, params):
    """Create a ROM model."""
    response = requests.post(f"{BASE_URL}/api/create-rom", json={
        "rom_type": rom_type,
        "model_id": model_id,
        "name": name,
        "params": params
    })
    return response.json()


def fit_rom(rom_id, snapshots, parameters=None):
    """Fit a ROM with training data."""
    print(f"Fitting ROM {rom_id}...")
    body = {
        "model_id": rom_id,
        "training_data": snapshots
    }
    if parameters is not None:
        body["parameters"] = parameters
    
    response = requests.post(f"{BASE_URL}/api/fit-rom", json=body)
    result = response.json()
    if result["success"]:
        print(f"  ROM fitted successfully")
        return result
    else:
        raise Exception(f"Failed to fit ROM: {result.get('error')}")


def compare_fom_rom(fom_id, rom_id, test_params=None, n_steps=50):
    """Compare FOM vs ROM predictions."""
    print(f"Comparing FOM {fom_id} vs ROM {rom_id}...")
    response = requests.post(f"{BASE_URL}/api/compare-fom-rom", json={
        "fom_id": fom_id,
        "rom_id": rom_id,
        "test_parameters": test_params,
        "n_steps": n_steps
    })
    return response.json()


def plot_comparison_results(comparison_result):
    """Plot FOM vs ROM comparison results."""
    if not comparison_result["success"]:
        print(f"Comparison failed: {comparison_result.get('error')}")
        return
    
    comparisons = comparison_result["comparisons"]
    errors = comparison_result["errors"]
    summary = comparison_result["summary"]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FOM vs ROM Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: FOM vs ROM trajectories for first comparison
    ax1 = axes[0, 0]
    if comparisons and comparisons[0].get("fom_result"):
        first_comp = comparisons[0]
        fom_data = np.array(first_comp["fom_result"])
        rom_data = np.array(first_comp["rom_result"])
        
        # Plot first 3 features
        n_features = min(3, fom_data.shape[0])
        for i in range(n_features):
            ax1.plot(fom_data[i, :], label=f'FOM Feature {i}', linewidth=2, alpha=0.7)
            ax1.plot(rom_data[i, :], '--', label=f'ROM Feature {i}', linewidth=2, alpha=0.7)
        
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Value')
        ax1.set_title(f'Trajectory Comparison (Parameter: {first_comp["parameter"]:.2f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error vs Parameter
    ax2 = axes[0, 1]
    if errors:
        params = [e["parameter"] for e in errors]
        rel_errors = [e["relative_error"] * 100 for e in errors]  # Convert to %
        mse_errors = [e["mse"] for e in errors]
        
        ax2_twin = ax2.twinx()
        line1 = ax2.plot(params, rel_errors, 'o-', color='#ff6b6b', linewidth=2, 
                        markersize=8, label='Relative Error (%)')
        line2 = ax2_twin.plot(params, mse_errors, 's-', color='#4a90e2', linewidth=2, 
                             markersize=8, label='MSE')
        
        ax2.set_xlabel('Parameter Value')
        ax2.set_ylabel('Relative Error (%)', color='#ff6b6b')
        ax2_twin.set_ylabel('MSE', color='#4a90e2')
        ax2.set_title('Error Metrics vs Parameter')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor='#ff6b6b')
        ax2_twin.tick_params(axis='y', labelcolor='#4a90e2')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
    
    # Plot 3: Error distribution
    ax3 = axes[1, 0]
    if errors:
        rel_errors = [e["relative_error"] * 100 for e in errors]
        ax3.hist(rel_errors, bins=min(10, len(rel_errors)), edgecolor='black', alpha=0.7, color='#50c878')
        ax3.set_xlabel('Relative Error (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axvline(summary["avg_relative_error"] * 100, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {summary["avg_relative_error"]*100:.2f}%')
        ax3.legend()
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    Performance Summary
    
    Number of Tests: {summary["n_tests"]}
    Average Relative Error: {summary["avg_relative_error"]*100:.2f}%
    Average MSE: {summary["avg_mse"]:.6f}
    
    Error Range:
    Min: {min([e["relative_error"]*100 for e in errors]):.2f}%
    Max: {max([e["relative_error"]*100 for e in errors]):.2f}%
    
    MSE Range:
    Min: {min([e["mse"] for e in errors]):.6f}
    Max: {max([e["mse"] for e in errors]):.6f}
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('fom_rom_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison graph saved to 'fom_rom_comparison.png'")
    plt.show()


def run_automated_comparison():
    """Run automated FOM vs ROM comparison workflow."""
    print("=" * 60)
    print("Automated FOM vs ROM Comparison")
    print("=" * 60)
    
    # Step 1: Create FOM
    print("\n[1/5] Creating FOM...")
    fom_result = create_fom(
        fom_type="CoupledOscillators",
        model_id="auto_fom",
        name="Automated Test FOM",
        params={"n_oscillators": 50, "coupling_strength": 0.1, "nonlinearity": 0.5}
    )
    if not fom_result["success"]:
        print(f"Failed to create FOM: {fom_result.get('error')}")
        return
    print(f"  Created FOM: {fom_result['model_id']} ({fom_result['metadata']['n_dof']} DOF)")
    
    # Step 2: Generate parametric snapshots
    print("\n[2/5] Generating parametric snapshots...")
    snapshot_result = generate_parametric_snapshots(
        fom_id="auto_fom",
        n_params=20,
        param_min=0.1,
        param_max=2.0
    )
    snapshots = snapshot_result["data"]
    parameters = snapshot_result["parameters"]
    
    # Step 3: Create ROM
    print("\n[3/5] Creating ROM...")
    rom_result = create_rom(
        rom_type="DMD",
        model_id="auto_rom",
        name="Automated Test ROM",
        params={"rank": 15, "implementation": "numpy"}
    )
    if not rom_result["success"]:
        print(f"Failed to create ROM: {rom_result.get('error')}")
        return
    print(f"  Created ROM: {rom_result['model_id']}")
    
    # Step 4: Fit ROM
    print("\n[4/5] Fitting ROM...")
    fit_result = fit_rom(
        rom_id="auto_rom",
        snapshots=snapshots,
        parameters=parameters
    )
    
    # Step 5: Compare FOM vs ROM
    print("\n[5/5] Comparing FOM vs ROM...")
    # Use test parameters within training range for better comparison
    test_params = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
    comparison_result = compare_fom_rom(
        fom_id="auto_fom",
        rom_id="auto_rom",
        test_params=test_params,
        n_steps=50
    )
    
    if comparison_result["success"]:
        summary = comparison_result["summary"]
        print(f"\nComparison Results:")
        print(f"  Average Relative Error: {summary['avg_relative_error']*100:.2f}%")
        print(f"  Average MSE: {summary['avg_mse']:.6f}")
        print(f"  Number of Tests: {summary['n_tests']}")
        
        # Plot results
        print("\nGenerating comparison graphs...")
        plot_comparison_results(comparison_result)
    else:
        print(f"Comparison failed: {comparison_result.get('error')}")
    
    print("\n" + "=" * 60)
    print("Automated comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Wait a bit for server to be ready
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    try:
        run_automated_comparison()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
