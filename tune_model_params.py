#!/usr/bin/env python3
"""
Focused Model Parameter Tuning for Fuzzy Metaballs
Tests different model parameters systematically.
"""

import os
import sys
import yaml
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess

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
    
    return {
        'experiment_name': experiment_name,
        'config': config_dict,
        'final_loss': final_loss,
        'total_time': total_time,
        'avg_forward': avg_forward,
        'avg_backward': avg_backward,
        'success': final_loss is not None
    }

def main():
    """Main parameter tuning function"""
    
    print("🚀 FUZZY METABALLS MODEL PARAMETER TUNING")
    print("=" * 60)
    
    # Load base configuration
    with open('config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create results directory
    results_dir = Path("tuning_results")
    results_dir.mkdir(exist_ok=True)
    
    # Model parameters to test
    model_tests = [
        # Test different numbers of mixtures
        {'num_mixtures': 20, 'gmm_init_scale': 1.0, 'rand_sphere_size': 30},
        {'num_mixtures': 40, 'gmm_init_scale': 1.0, 'rand_sphere_size': 30},  # baseline
        {'num_mixtures': 60, 'gmm_init_scale': 1.0, 'rand_sphere_size': 30},
        {'num_mixtures': 80, 'gmm_init_scale': 1.0, 'rand_sphere_size': 30},
        
        # Test different GMM init scales
        {'num_mixtures': 40, 'gmm_init_scale': 0.5, 'rand_sphere_size': 30},
        {'num_mixtures': 40, 'gmm_init_scale': 2.0, 'rand_sphere_size': 30},
        {'num_mixtures': 40, 'gmm_init_scale': 4.0, 'rand_sphere_size': 30},
        
        # Test different sphere sizes
        {'num_mixtures': 40, 'gmm_init_scale': 1.0, 'rand_sphere_size': 10},
        {'num_mixtures': 40, 'gmm_init_scale': 1.0, 'rand_sphere_size': 50},
        {'num_mixtures': 40, 'gmm_init_scale': 1.0, 'rand_sphere_size': 100},
    ]
    
    results = []
    
    for i, test_params in enumerate(model_tests):
        # Create modified config
        config = base_config.copy()
        config['model'].update(test_params)
        
        # Create experiment name
        exp_name = f"model_{i:02d}_m{test_params['num_mixtures']}_g{test_params['gmm_init_scale']}_s{test_params['rand_sphere_size']}"
        
        print(f"\n🧪 Test {i+1}/{len(model_tests)}: {exp_name}")
        print(f"   Mixtures: {test_params['num_mixtures']}, GMM scale: {test_params['gmm_init_scale']}, Sphere size: {test_params['rand_sphere_size']}")
        
        # Run experiment
        result = run_experiment(config, results_dir, exp_name)
        
        if result and result['success']:
            results.append(result)
            print(f"✅ Loss: {result['final_loss']:.6f}, Time: {result['total_time']:.1f}ms")
        else:
            print(f"❌ Failed")
    
    # Analyze results
    if results:
        print(f"\n📊 ANALYSIS OF {len(results)} SUCCESSFUL EXPERIMENTS")
        print("=" * 60)
        
        # Sort by loss
        results.sort(key=lambda x: x['final_loss'])
        
        print("\n🏆 TOP 5 CONFIGURATIONS BY LOSS:")
        for i, result in enumerate(results[:5]):
            config = result['config']['model']
            print(f"{i+1}. Loss: {result['final_loss']:.6f} | "
                  f"Mixtures: {config['num_mixtures']}, "
                  f"GMM scale: {config['gmm_init_scale']}, "
                  f"Sphere size: {config['rand_sphere_size']}")
        
        # Find best configuration
        best_result = results[0]
        best_config = base_config.copy()
        best_config['model'] = best_result['config']['model']
        
        # Save best configuration
        with open(results_dir / "best_model_config.yaml", 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False)
        
        print(f"\n💾 Best model configuration saved to: {results_dir / 'best_model_config.yaml'}")
        print(f"Best loss: {best_result['final_loss']:.6f}")
        
        # Create visualization
        mixtures = [r['config']['model']['num_mixtures'] for r in results]
        losses = [r['final_loss'] for r in results]
        times = [r['total_time'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss vs mixtures
        ax1.scatter(mixtures, losses, alpha=0.7, s=100)
        ax1.set_xlabel('Number of Mixtures')
        ax1.set_ylabel('Final Loss')
        ax1.set_title('Loss vs Number of Mixtures')
        ax1.grid(True, alpha=0.3)
        
        # Time vs mixtures
        ax2.scatter(mixtures, times, alpha=0.7, s=100, color='red')
        ax2.set_xlabel('Number of Mixtures')
        ax2.set_ylabel('Total Time (ms)')
        ax2.set_title('Time vs Number of Mixtures')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / "model_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Analysis plot saved to: {results_dir / 'model_analysis.png'}")
    
    print("\n✨ MODEL PARAMETER TUNING COMPLETE!")

if __name__ == '__main__':
    main()
