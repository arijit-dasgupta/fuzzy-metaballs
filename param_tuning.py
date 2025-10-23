#!/usr/bin/env python3
"""
Comprehensive Parameter Tuning for Fuzzy Metaballs
Systematically tests different parameter combinations and measures reconstruction quality.
"""

import os
import sys
import yaml
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
import subprocess
import shutil

def run_experiment(config_dict, output_dir, experiment_name):
    """Run a single experiment with given configuration"""
    
    # Create experiment directory
    exp_dir = Path(output_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = exp_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    # Run experiment
    start_time = time.time()
    try:
        result = subprocess.run([
            'python', 'run_fmb.py', 
            '--config', str(config_path),
            '--output-dir', str(exp_dir / "output")
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode != 0:
            print(f"❌ Experiment {experiment_name} failed:")
            print(result.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment {experiment_name} timed out")
        return None
    
    end_time = time.time()
    
    # Parse results from stdout
    lines = result.stdout.split('\n')
    final_loss = None
    total_time = None
    avg_forward = None
    avg_backward = None
    
    for line in lines:
        if 'Final loss:' in line:
            final_loss = float(line.split(':')[1].strip())
        elif 'Total time:' in line and 'ms' in line:
            time_str = line.split(':')[1].strip().split()[0]
            total_time = float(time_str)
        elif 'Avg forward time:' in line:
            avg_forward = float(line.split(':')[1].strip().split()[0])
        elif 'Avg backward time:' in line:
            avg_backward = float(line.split(':')[1].strip().split()[0])
    
    # Calculate depth quality metrics if available
    depth_quality = None
    depth_comparison_path = exp_dir / "output" / "09_depth_comparison.png"
    if depth_comparison_path.exists():
        # For now, we'll use a simple heuristic based on file size and loss
        # In a real implementation, you'd load and analyze the depth images
        depth_quality = 1.0 / (final_loss + 1e-6) if final_loss else 0.0
    
    return {
        'experiment_name': experiment_name,
        'config': config_dict,
        'final_loss': final_loss,
        'total_time': total_time,
        'avg_forward': avg_forward,
        'avg_backward': avg_backward,
        'depth_quality': depth_quality,
        'success': final_loss is not None
    }

def create_parameter_grid():
    """Define parameter combinations to test"""
    
    # Model parameters
    model_params = {
        'num_mixtures': [20, 40, 60, 80],
        'gmm_init_scale': [0.5, 1.0, 2.0, 4.0],
        'rand_sphere_size': [10, 20, 30, 50]
    }
    
    # Rendering parameters  
    rendering_params = {
        'num_views': [10, 20, 30],
        'image_width': [32, 64, 128],
        'image_height': [32, 64, 128],
        'vfov_degrees': [30, 45, 60]
    }
    
    # Optimization parameters
    optimization_params = {
        'num_epochs': [5, 10, 20],
        'batch_size': [400, 800, 1600],
        'initial_lr': [0.01, 0.1, 0.5],
        'opt_shape_scale': [1.5, 2.2, 3.0]
    }
    
    # Hyperparameters
    hyperparams = {
        'beta2_exp': [1.0, 2.0, 3.0],
        'beta3_exp': [0.1, 0.25, 0.5]
    }
    
    return {
        'model': model_params,
        'rendering': rendering_params, 
        'optimization': optimization_params,
        'hyperparams': hyperparams
    }

def run_parameter_sweep(param_category, param_name, values, base_config, output_dir):
    """Run experiments for a single parameter sweep"""
    
    results = []
    
    for i, value in enumerate(values):
        # Create modified config
        config = base_config.copy()
        config[param_category][param_name] = value
        
        # Create experiment name
        exp_name = f"{param_category}_{param_name}_{value}"
        
        print(f"\n🧪 Testing {param_name} = {value} ({i+1}/{len(values)})")
        
        # Run experiment
        result = run_experiment(config, output_dir, exp_name)
        
        if result:
            results.append(result)
            print(f"✅ Loss: {result['final_loss']:.6f}, Time: {result['total_time']:.1f}ms")
        else:
            print(f"❌ Failed")
    
    return results

def analyze_results(results, param_name):
    """Analyze and visualize results for a parameter"""
    
    if not results:
        return
    
    # Extract data
    values = [r['config'][param_name.split('_')[0]][param_name.split('_')[1]] for r in results]
    losses = [r['final_loss'] for r in results if r['final_loss']]
    times = [r['total_time'] for r in results if r['total_time']]
    
    if not losses:
        return
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(values[:len(losses)], losses, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Final Loss')
    ax1.set_title(f'Loss vs {param_name}')
    ax1.grid(True, alpha=0.3)
    
    # Time plot
    if times:
        ax2.plot(values[:len(times)], times, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel(param_name)
        ax2.set_ylabel('Total Time (ms)')
        ax2.set_title(f'Time vs {param_name}')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """Main parameter tuning function"""
    
    print("🚀 FUZZY METABALLS PARAMETER TUNING")
    print("=" * 60)
    
    # Load base configuration
    with open('config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create results directory
    results_dir = Path("tuning_results")
    results_dir.mkdir(exist_ok=True)
    
    # Get parameter grid
    param_grid = create_parameter_grid()
    
    # Store all results
    all_results = {}
    
    # Test each parameter category
    for category, params in param_grid.items():
        print(f"\n📊 TESTING {category.upper()} PARAMETERS")
        print("-" * 40)
        
        category_results = {}
        
        for param_name, values in params.items():
            print(f"\n🔍 Testing {param_name}: {values}")
            
            # Run parameter sweep
            results = run_parameter_sweep(category, param_name, values, base_config, results_dir)
            
            if results:
                category_results[param_name] = results
                
                # Find best result
                best_result = min(results, key=lambda x: x['final_loss'] if x['final_loss'] else float('inf'))
                print(f"🏆 Best {param_name}: {best_result['config'][category][param_name]} (loss: {best_result['final_loss']:.6f})")
                
                # Create visualization
                fig = analyze_results(results, f"{category}_{param_name}")
                if fig:
                    fig.savefig(results_dir / f"{category}_{param_name}_analysis.png", dpi=150, bbox_inches='tight')
                    plt.close(fig)
        
        all_results[category] = category_results
    
    # Save comprehensive results
    with open(results_dir / "all_results.json", 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(all_results, f, indent=2, default=convert_types)
    
    # Find overall best configuration
    print("\n🏆 OVERALL BEST CONFIGURATION")
    print("=" * 60)
    
    best_config = base_config.copy()
    best_loss = float('inf')
    
    for category, params in all_results.items():
        for param_name, results in params.items():
            if results:
                best_result = min(results, key=lambda x: x['final_loss'] if x['final_loss'] else float('inf'))
                if best_result['final_loss'] and best_result['final_loss'] < best_loss:
                    best_loss = best_result['final_loss']
                    # Update best config
                    for key, value in best_result['config'][category].items():
                        best_config[category][key] = value
    
    # Save best configuration
    with open(results_dir / "best_config.yaml", 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    print(f"Best configuration saved to: {results_dir / 'best_config.yaml'}")
    print(f"Best loss: {best_loss:.6f}")
    
    print("\n✨ PARAMETER TUNING COMPLETE!")
    print(f"Results saved to: {results_dir}")

if __name__ == '__main__':
    main()
