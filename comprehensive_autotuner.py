#!/usr/bin/env python3
"""
Comprehensive Fuzzy Metaballs Autotuner
Intelligent parameter search with continuous optimization.
"""

import os
import sys
import json
import time
import random
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import subprocess
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt

class FuzzyMetaballsAutotuner:
    def __init__(self, base_config_path='config.yaml', results_dir='autotune_results'):
        self.base_config_path = base_config_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Load base configuration
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Define parameter search space (much wider ranges)
        self.param_ranges = {
            # Model parameters
            'num_mixtures': (10, 200),           # Much wider range
            'gmm_init_scale': (0.1, 10.0),      # Much wider range  
            'rand_sphere_size': (1, 200),        # Much wider range
            
            # Rendering parameters
            'num_views': (5, 50),               # Much wider range
            'image_width': (32, 256),            # Much wider range
            'image_height': (32, 256),           # Much wider range
            'vfov_degrees': (20, 80),            # Much wider range
            
            # Optimization parameters
            'num_epochs': (3, 50),              # Much wider range
            'batch_size': (100, 4000),          # Much wider range
            'initial_lr': (0.001, 2.0),         # Much wider range
            'opt_shape_scale': (0.5, 5.0),      # Much wider range
            
            # Learning rate schedule
            'lr_decay_p_thresh': (0.1, 0.9),
            'lr_decay_window': (5, 50),
            'lr_decay_p_window': (3, 20),
            'lr_decay_slope_less': (-1e-2, -1e-6),
            'lr_decay_max_drops': (2, 30),
            
            # Hyperparameters
            'beta2_exp': (0.5, 5.0),           # Much wider range
            'beta3_exp': (0.01, 1.0),          # Much wider range
        }
        
        # Results storage
        self.results = []
        self.best_result = None
        self.best_score = float('inf')
        
        # Bayesian optimization setup
        self.gp_model = None
        self.param_names = list(self.param_ranges.keys())
        
        print(f"🚀 Fuzzy Metaballs Autotuner Initialized")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"🎯 Parameter space: {len(self.param_ranges)} parameters")
        print(f"🔍 Search ranges: {self.param_ranges}")
    
    def sample_parameters(self, method='random'):
        """Sample parameters from the search space"""
        print(f"🎲 Sampling parameters using method: {method}")
        
        if method == 'random':
            print("   📊 Using random sampling for exploration")
            params = {}
            for name, (min_val, max_val) in self.param_ranges.items():
                if name in ['num_mixtures', 'num_views', 'image_width', 'image_height', 
                           'num_epochs', 'batch_size', 'lr_decay_window', 'lr_decay_p_window', 'lr_decay_max_drops']:
                    # Integer parameters
                    params[name] = random.randint(int(min_val), int(max_val))
                    print(f"   🔢 {name}: {params[name]} (int, range: {int(min_val)}-{int(max_val)})")
                else:
                    # Float parameters
                    if name in ['gmm_init_scale', 'initial_lr']:
                        # Log scale for these parameters
                        params[name] = np.exp(random.uniform(np.log(min_val), np.log(max_val)))
                        print(f"   📈 {name}: {params[name]:.4f} (log scale, range: {min_val:.4f}-{max_val:.4f})")
                    else:
                        params[name] = random.uniform(min_val, max_val)
                        print(f"   📊 {name}: {params[name]:.4f} (linear, range: {min_val:.4f}-{max_val:.4f})")
            return params
        
        elif method == 'bayesian' and len(self.results) >= 5:
            print(f"   🧠 Using Bayesian optimization (have {len(self.results)} previous results)")
            # Use Gaussian Process for intelligent sampling
            return self._bayesian_sample()
        
        else:
            print(f"   ⚠️  Fallback to random sampling (method: {method}, results: {len(self.results)})")
            # Fallback to random
            return self.sample_parameters('random')
    
    def _bayesian_sample(self):
        """Use Bayesian optimization to sample next parameters"""
        print(f"   🧠 Bayesian optimization with {len(self.results)} previous results")
        
        if len(self.results) < 5:
            print("   ⚠️  Not enough results for Bayesian optimization, falling back to random")
            return self.sample_parameters('random')
        
        # Prepare training data
        print("   📊 Preparing training data for Gaussian Process...")
        X = []
        y = []
        for result in self.results:
            if result['score'] is not None:
                params = []
                for name in self.param_names:
                    val = result['config'][name]
                    # Normalize to [0, 1]
                    min_val, max_val = self.param_ranges[name]
                    if name in ['gmm_init_scale', 'initial_lr']:
                        # Log scale normalization
                        norm_val = (np.log(val) - np.log(min_val)) / (np.log(max_val) - np.log(min_val))
                    else:
                        norm_val = (val - min_val) / (max_val - min_val)
                    params.append(norm_val)
                X.append(params)
                y.append(result['score'])
        
        print(f"   📈 Training data: {len(X)} samples, score range: {min(y):.4f} - {max(y):.4f}")
        
        if len(X) < 5:
            print("   ⚠️  Not enough valid training data, falling back to random")
            return self.sample_parameters('random')
        
        # Fit Gaussian Process
        print("   🔧 Fitting Gaussian Process model...")
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp_model = GaussianProcessRegressor(kernel=kernel, random_state=42)
        self.gp_model.fit(X, y)
        print("   ✅ Gaussian Process model fitted successfully")
        
        # Acquisition function (Expected Improvement)
        def acquisition(x):
            mean, std = self.gp_model.predict([x], return_std=True)
            mean, std = mean[0], std[0]
            best_y = min(y)
            z = (best_y - mean) / (std + 1e-9)
            ei = (best_y - mean) * self._normal_cdf(z) + std * self._normal_pdf(z)
            return ei
        
        # Optimize acquisition function
        print("   🎯 Optimizing acquisition function (Expected Improvement)...")
        best_x = None
        best_acq = -np.inf
        
        for i in range(100):  # Random restarts
            x0 = [random.uniform(0, 1) for _ in self.param_names]
            try:
                result = minimize(lambda x: -acquisition(x), x0, bounds=[(0, 1)] * len(self.param_names))
                if result.success and -result.fun > best_acq:
                    best_acq = -result.fun
                    best_x = result.x
                    if i % 20 == 0:  # Print progress every 20 attempts
                        print(f"   🔍 Optimization attempt {i+1}/100, best acquisition: {best_acq:.6f}")
            except:
                continue
        
        if best_x is None:
            print("   ❌ Acquisition optimization failed, falling back to random")
            return self.sample_parameters('random')
        
        print(f"   🏆 Best acquisition value: {best_acq:.6f}")
        
        # Convert back to parameter space
        print("   🔄 Converting optimized parameters back to original space...")
        params = {}
        for i, name in enumerate(self.param_names):
            norm_val = best_x[i]
            min_val, max_val = self.param_ranges[name]
            
            if name in ['num_mixtures', 'num_views', 'image_width', 'image_height', 
                       'num_epochs', 'batch_size', 'lr_decay_window', 'lr_decay_p_window', 'lr_decay_max_drops']:
                params[name] = int(min_val + norm_val * (max_val - min_val))
                print(f"   🔢 {name}: {params[name]} (int, range: {int(min_val)}-{int(max_val)})")
            else:
                if name in ['gmm_init_scale', 'initial_lr']:
                    params[name] = np.exp(np.log(min_val) + norm_val * (np.log(max_val) - np.log(min_val)))
                    print(f"   📈 {name}: {params[name]:.4f} (log scale, range: {min_val:.4f}-{max_val:.4f})")
                else:
                    params[name] = min_val + norm_val * (max_val - min_val)
                    print(f"   📊 {name}: {params[name]:.4f} (linear, range: {min_val:.4f}-{max_val:.4f})")
        
        print("   ✅ Bayesian optimization complete!")
        return params
    
    def _normal_cdf(self, x):
        """Normal CDF approximation"""
        return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def _normal_pdf(self, x):
        """Normal PDF"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def create_config(self, params):
        """Create configuration from parameters"""
        config = self.base_config.copy()
        
        # Update model parameters
        if 'num_mixtures' in params:
            config['model']['num_mixtures'] = params['num_mixtures']
        if 'gmm_init_scale' in params:
            config['model']['gmm_init_scale'] = params['gmm_init_scale']
        if 'rand_sphere_size' in params:
            config['model']['rand_sphere_size'] = params['rand_sphere_size']
        
        # Update rendering parameters
        if 'num_views' in params:
            config['rendering']['num_views'] = params['num_views']
        if 'image_width' in params:
            config['rendering']['image_width'] = params['image_width']
        if 'image_height' in params:
            config['rendering']['image_height'] = params['image_height']
        if 'vfov_degrees' in params:
            config['rendering']['vfov_degrees'] = params['vfov_degrees']
        
        # Update optimization parameters
        if 'num_epochs' in params:
            config['optimization']['num_epochs'] = params['num_epochs']
        if 'batch_size' in params:
            config['optimization']['batch_size'] = params['batch_size']
        if 'initial_lr' in params:
            config['optimization']['initial_lr'] = params['initial_lr']
        if 'opt_shape_scale' in params:
            config['optimization']['opt_shape_scale'] = params['opt_shape_scale']
        
        # Update learning rate schedule
        if 'lr_decay_p_thresh' in params:
            config['optimization']['lr_decay_p_thresh'] = params['lr_decay_p_thresh']
        if 'lr_decay_window' in params:
            config['optimization']['lr_decay_window'] = params['lr_decay_window']
        if 'lr_decay_p_window' in params:
            config['optimization']['lr_decay_p_window'] = params['lr_decay_p_window']
        if 'lr_decay_slope_less' in params:
            config['optimization']['lr_decay_slope_less'] = params['lr_decay_slope_less']
        if 'lr_decay_max_drops' in params:
            config['optimization']['lr_decay_max_drops'] = params['lr_decay_max_drops']
        
        # Update hyperparameters
        if 'beta2_exp' in params:
            config['hyperparams']['beta2_exp'] = params['beta2_exp']
        if 'beta3_exp' in params:
            config['hyperparams']['beta3_exp'] = params['beta3_exp']
        
        return config
    
    def run_experiment(self, params, experiment_id):
        """Run a single experiment"""
        print(f"🔧 Creating configuration for experiment {experiment_id}")
        config = self.create_config(params)
        
        # Create experiment directory
        exp_dir = self.results_dir / f"exp_{experiment_id:04d}"
        exp_dir.mkdir(exist_ok=True)
        print(f"📁 Experiment directory: {exp_dir}")
        
        # Save config
        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"💾 Configuration saved to: {config_path}")
        
        # Set tune mode environment variable
        env = os.environ.copy()
        env['TUNE_MODE'] = 'true'
        print("🎯 Running experiment in TUNE_MODE (suppressed output)")
        
        # Run experiment
        start_time = time.time()
        print("🚀 Starting Fuzzy Metaballs experiment...")
        try:
            result = subprocess.run([
                'python', 'run_fmb.py', 
                '--config', str(config_path),
                '--output-dir', str(exp_dir / "output")
            ], capture_output=True, text=True, timeout=600, env=env)  # 10 minute timeout
            
            if result.returncode != 0:
                print(f"❌ Experiment {experiment_id} failed with return code {result.returncode}")
                print(f"   Error: {result.stderr[:200]}...")
                return None
            else:
                print(f"✅ Experiment {experiment_id} completed successfully")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Experiment {experiment_id} timed out after 10 minutes")
            return None
        
        end_time = time.time()
        runtime = end_time - start_time
        print(f"⏱️  Experiment runtime: {runtime:.1f} seconds")
        
        # Parse JSON output
        print("📊 Parsing experiment metrics...")
        try:
            metrics = json.loads(result.stdout)
            print(f"✅ Successfully parsed metrics: {list(metrics.keys())}")
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse metrics for experiment {experiment_id}: {e}")
            print(f"   Raw output: {result.stdout[:200]}...")
            return None
        
        # Calculate composite score
        print("🧮 Calculating composite quality score...")
        score = self.calculate_score(metrics)
        print(f"📈 Composite score: {score:.6f}")
        
        result_data = {
            'experiment_id': experiment_id,
            'params': params,
            'config': config,
            'metrics': metrics,
            'score': score,
            'runtime': runtime,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"💾 Experiment {experiment_id} data prepared successfully")
        return result_data
    
    def calculate_score(self, metrics):
        """Calculate composite quality score"""
        print("   🧮 Calculating composite quality score...")
        
        # Weight different metrics
        weights = {
            'final_loss': 0.4,           # Primary optimization target
            'depth_quality': 0.4,       # Depth reconstruction quality
            'efficiency': 0.2           # Computational efficiency
        }
        print(f"   📊 Score weights: Loss={weights['final_loss']}, Depth={weights['depth_quality']}, Efficiency={weights['efficiency']}")
        
        # Final loss component (lower is better)
        final_loss = metrics['final_loss']
        loss_score = 1.0 / (final_loss + 1e-6)
        print(f"   📉 Final loss: {final_loss:.6f} → Loss score: {loss_score:.6f}")
        
        # Depth quality component
        depth_score = 0.0
        if 'depth_quality' in metrics and 'aggregate' in metrics['depth_quality']:
            agg = metrics['depth_quality']['aggregate']
            print(f"   🎯 Depth quality metrics: SSIM={agg.get('mean_ssim', 0):.4f}, Corr={agg.get('mean_correlation', 0):.4f}, MAE={agg.get('mean_mae', 1):.4f}, RMSE={agg.get('mean_rmse', 1):.4f}")
            # Higher SSIM and correlation are better, lower MAE/RMSE are better
            depth_score = (
                agg.get('mean_ssim', 0) * 0.3 +
                agg.get('mean_correlation', 0) * 0.3 +
                (1 - agg.get('mean_mae', 1)) * 0.2 +
                (1 - agg.get('mean_rmse', 1)) * 0.2
            )
            print(f"   🎯 Depth score: {depth_score:.6f}")
        else:
            print("   ⚠️  No depth quality metrics available")
        
        # Efficiency component (faster is better)
        total_time = metrics['total_time']
        efficiency_score = 1.0 / (total_time + 1e-6)
        print(f"   ⏱️  Total time: {total_time:.1f}ms → Efficiency score: {efficiency_score:.6f}")
        
        # Normalize scores to [0, 1]
        loss_score = min(loss_score / 10.0, 1.0)  # Assume good loss is < 0.1
        depth_score = max(0, min(depth_score, 1.0))
        efficiency_score = min(efficiency_score / 1000.0, 1.0)  # Assume good time is < 1000ms
        
        print(f"   📊 Normalized scores: Loss={loss_score:.6f}, Depth={depth_score:.6f}, Efficiency={efficiency_score:.6f}")
        
        # Composite score
        composite_score = (
            weights['final_loss'] * loss_score +
            weights['depth_quality'] * depth_score +
            weights['efficiency'] * efficiency_score
        )
        
        print(f"   🏆 Final composite score: {composite_score:.6f}")
        return composite_score
    
    def save_results(self):
        """Save all results to file"""
        results_file = self.results_dir / "all_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save best configuration
        if self.best_result:
            best_config_file = self.results_dir / "best_config.yaml"
            with open(best_config_file, 'w') as f:
                yaml.dump(self.best_result['config'], f, default_flow_style=False)
    
    def plot_progress(self):
        """Plot optimization progress"""
        if len(self.results) < 2:
            return
        
        scores = [r['score'] for r in self.results if r['score'] is not None]
        losses = [r['metrics']['final_loss'] for r in self.results if r['score'] is not None]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Score progression
        ax1.plot(scores, 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Composite Score')
        ax1.set_title('Optimization Progress')
        ax1.grid(True, alpha=0.3)
        
        # Loss progression
        ax2.plot(losses, 'r-o', linewidth=2, markersize=4)
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('Final Loss')
        ax2.set_title('Loss Progression')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "optimization_progress.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_autotuning(self, max_experiments=1000, save_interval=10):
        """Run continuous autotuning"""
        print(f"🚀 Starting autotuning for up to {max_experiments} experiments")
        print(f"💾 Saving results every {save_interval} experiments")
        print("=" * 80)
        
        for experiment_id in range(max_experiments):
            print(f"\n{'='*60}")
            print(f"🧪 EXPERIMENT {experiment_id + 1}/{max_experiments}")
            print(f"{'='*60}")
            
            # Choose sampling method
            if experiment_id < 20:
                method = 'random'  # Start with random exploration
                print(f"🎲 Phase: Random exploration (experiments 1-20)")
            else:
                method = 'bayesian'  # Use intelligent sampling
                print(f"🧠 Phase: Bayesian optimization (experiments 21+)")
            
            # Sample parameters
            print(f"\n📋 Sampling parameters using {method} method...")
            params = self.sample_parameters(method)
            print(f"✅ Sampled parameters: {params}")
            
            # Run experiment
            print(f"\n🔬 Running experiment...")
            result = self.run_experiment(params, experiment_id)
            
            if result:
                self.results.append(result)
                print(f"\n📊 EXPERIMENT RESULTS:")
                print(f"   🏆 Score: {result['score']:.6f}")
                print(f"   📉 Final Loss: {result['metrics']['final_loss']:.6f}")
                print(f"   ⏱️  Runtime: {result['runtime']:.1f}s")
                print(f"   🎯 Total experiments completed: {len(self.results)}")
                
                # Update best result
                if result['score'] < self.best_score:
                    self.best_score = result['score']
                    self.best_result = result
                    print(f"\n🎉 NEW BEST RESULT! 🎉")
                    print(f"   🏆 New best score: {result['score']:.6f}")
                    print(f"   📉 New best loss: {result['metrics']['final_loss']:.6f}")
                    print(f"   🎯 Best parameters: {result['params']}")
                else:
                    print(f"   📈 Current best score: {self.best_score:.6f} (no improvement)")
            else:
                print(f"\n❌ EXPERIMENT FAILED")
                print(f"   🎯 Total successful experiments: {len(self.results)}")
            
            # Save results periodically
            if (experiment_id + 1) % save_interval == 0:
                print(f"\n💾 SAVING RESULTS (checkpoint {experiment_id + 1})")
                self.save_results()
                self.plot_progress()
                print(f"   ✅ Results saved to: {self.results_dir}")
                print(f"   📊 Progress plot updated")
            
            # Print summary
            print(f"\n📈 OPTIMIZATION SUMMARY:")
            if self.best_result:
                print(f"   🏆 Best score so far: {self.best_score:.6f}")
                print(f"   📉 Best loss so far: {self.best_result['metrics']['final_loss']:.6f}")
                print(f"   🎯 Best parameters: {self.best_result['params']}")
            else:
                print(f"   ⚠️  No successful experiments yet")
            
            print(f"   📊 Total experiments: {experiment_id + 1}")
            print(f"   ✅ Successful experiments: {len(self.results)}")
            print(f"   📈 Success rate: {len(self.results)/(experiment_id + 1)*100:.1f}%")
        
        # Final save
        print(f"\n🏁 AUTOTUNING COMPLETE!")
        print(f"💾 Saving final results...")
        self.save_results()
        self.plot_progress()
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   🎯 Total experiments: {len(self.results)}")
        print(f"   🏆 Best score: {self.best_score:.6f}")
        if self.best_result:
            print(f"   📉 Best loss: {self.best_result['metrics']['final_loss']:.6f}")
            print(f"   🎯 Best parameters: {self.best_result['params']}")
            print(f"   ⏱️  Best runtime: {self.best_result['runtime']:.1f}s")
        print(f"   📁 Results saved to: {self.results_dir}")
        print(f"   📊 Progress plot: {self.results_dir}/optimization_progress.png")


def main():
    """Main autotuning function"""
    print("🚀 COMPREHENSIVE FUZZY METABALLS AUTOTUNER")
    print("=" * 80)
    print("This will run continuously and intelligently search parameter space.")
    print("You can go to sleep - it will run automatically!")
    print("=" * 80)
    
    # Create autotuner
    autotuner = FuzzyMetaballsAutotuner()
    
    # Run autotuning
    autotuner.run_autotuning(max_experiments=1000, save_interval=10)


if __name__ == '__main__':
    main()
