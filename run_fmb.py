#!/usr/bin/env python3
"""
Fuzzy Metaballs: Shape from Silhouette Reconstruction

This script demonstrates 3D shape reconstruction from multiple 2D silhouette views
using the Fuzzy Metaballs differentiable renderer. Given a set of camera poses and
silhouette images, the algorithm optimizes a Gaussian Mixture Model (GMM) representation
to match the observed silhouettes across all views.

Method:
  - Input: 20 camera views with silhouette masks (binary images)
  - Representation: 40 Gaussian mixture components (520 parameters)
  - Loss: Binary cross-entropy between rendered and target silhouettes
  - Optimization: Adam with adaptive learning rate scheduling

Based on: "Approximate Differentiable Rendering with Algebraic Surfaces"
          Keselman & Hebert, ECCV 2022
          https://arxiv.org/abs/2207.10606
"""

import os
import sys

# Check for --cpu flag BEFORE any other imports (especially JAX)
if '--cpu' in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
import argparse
import yaml
from pathlib import Path
from collections import defaultdict

# Set matplotlib to non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import trimesh
import pyrender
import transforms3d
from tqdm import tqdm

# Import local utilities
from util import image_grid, DegradeLR


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Fuzzy Metaballs Shape from Silhouette Reconstruction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # General options
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--cpu', action='store_true',
                        help='Force JAX to use CPU instead of GPU')
    
    # Model parameters
    parser.add_argument('--num-mixtures', type=int, metavar='N',
                        help='Number of Gaussian mixture components')
    parser.add_argument('--gmm-init-scale', type=float, metavar='F',
                        help='Initial scale for GMM weights')
    parser.add_argument('--rand-sphere-size', type=float, metavar='F',
                        help='Initial sphere size for random initialization')
    
    # Rendering parameters
    parser.add_argument('--num-views', type=int, metavar='N',
                        help='Number of camera views')
    parser.add_argument('--image-width', type=int, metavar='W',
                        help='Image width in pixels')
    parser.add_argument('--image-height', type=int, metavar='H',
                        help='Image height in pixels')
    parser.add_argument('--vfov-degrees', type=float, metavar='F',
                        help='Vertical field of view in degrees')
    
    # Optimization parameters
    parser.add_argument('--num-epochs', type=int, metavar='N',
                        help='Maximum number of training epochs')
    parser.add_argument('--batch-size', type=int, metavar='N',
                        help='Number of rays per batch')
    parser.add_argument('--initial-lr', type=float, metavar='F',
                        help='Initial learning rate')
    parser.add_argument('--opt-shape-scale', type=float, metavar='F',
                        help='Shape scale multiplier for optimization')
    
    # I/O parameters
    parser.add_argument('--mesh-file', type=str, metavar='PATH',
                        help='Path to input mesh file')
    parser.add_argument('--output-dir', type=str, metavar='DIR',
                        help='Directory for output files')
    
    # Random seed
    parser.add_argument('--random-seed', type=int, metavar='N',
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def merge_config_and_args(config, args):
    """Merge configuration from YAML and command-line arguments"""
    # Command-line arguments override YAML config
    if args.num_mixtures is not None:
        config['model']['num_mixtures'] = args.num_mixtures
    if args.gmm_init_scale is not None:
        config['model']['gmm_init_scale'] = args.gmm_init_scale
    if args.rand_sphere_size is not None:
        config['model']['rand_sphere_size'] = args.rand_sphere_size
    
    if args.num_views is not None:
        config['rendering']['num_views'] = args.num_views
    if args.image_width is not None:
        config['rendering']['image_width'] = args.image_width
    if args.image_height is not None:
        config['rendering']['image_height'] = args.image_height
    if args.vfov_degrees is not None:
        config['rendering']['vfov_degrees'] = args.vfov_degrees
    
    if args.num_epochs is not None:
        config['optimization']['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        config['optimization']['batch_size'] = args.batch_size
    if args.initial_lr is not None:
        config['optimization']['initial_lr'] = args.initial_lr
    if args.opt_shape_scale is not None:
        config['optimization']['opt_shape_scale'] = args.opt_shape_scale
    
    if args.mesh_file is not None:
        config['io']['mesh_file'] = args.mesh_file
    if args.output_dir is not None:
        config['io']['output_dir'] = args.output_dir
    
    if args.random_seed is not None:
        config['random_seed'] = args.random_seed
    
    return config


def benchmark(name, timings):
    """Context manager for timing code sections"""
    class Timer:
        def __init__(self, name, timings):
            self.name = name
            self.timings = timings
            
        def __enter__(self):
            self.start = time.perf_counter()
            return self
            
        def __exit__(self, *args):
            elapsed = (time.perf_counter() - self.start) * 1000  # Convert to ms
            self.timings[self.name].append(elapsed)
    
    return Timer(name, timings)


def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    config = merge_config_and_args(config, args)
    
    # Extract config values
    NUM_MIXTURE = config['model']['num_mixtures']
    gmm_init_scale = config['model']['gmm_init_scale']
    rand_sphere_size = config['model']['rand_sphere_size']
    
    num_views = config['rendering']['num_views']
    image_size = (config['rendering']['image_height'], config['rendering']['image_width'])
    vfov_degrees = config['rendering']['vfov_degrees']
    
    Nepochs = config['optimization']['num_epochs']
    batch_size = config['optimization']['batch_size']
    initial_lr = config['optimization']['initial_lr']
    opt_shape_scale = config['optimization']['opt_shape_scale']
    clip_alpha = config['optimization']['clip_alpha']
    
    mesh_file = config['io']['mesh_file']
    output_dir = config['io']['output_dir']
    
    random_seed = config['random_seed']
    
    # Timing dictionary
    timings = defaultdict(list)
    
    # ============================================================================
    # BANNER
    # ============================================================================
    print("=" * 80)
    print(" " * 20 + "FUZZY METABALLS DIFFERENTIABLE RENDERER")
    print(" " * 15 + "Shape from Silhouette Reconstruction Demo")
    print("=" * 80)
    print()
    print("📄 Paper: Keselman & Hebert, ECCV 2022")
    print("🔗 Project: https://leonidk.github.io/fuzzy-metaballs")
    print()
    print("⚡ Performance claims from paper:")
    print("   • Forward passes:  ~5x faster than mesh renderers")
    print("   • Backward passes: ~30x faster than mesh renderers")
    print("=" * 80)
    print()
    
    # ============================================================================
    # SETUP
    # ============================================================================
    
    print("🔧 INITIALIZATION")
    print("-" * 80)
    
    if args.cpu:
        print("🖥️  JAX forced to CPU mode")
    
    # Start virtual display for headless rendering
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(1400, 900))
    display.start()
    print("✓ Virtual display started (1400x900)")
    
    # Create output directory for plots
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory: ./{output_dir}/")
    print()
    
    # ============================================================================
    # LOAD MESH AND SETUP CAMERAS
    # ============================================================================
    
    print("📦 LOADING 3D MODEL")
    print("-" * 80)
    mesh_tri = trimesh.load(mesh_file)
    
    # Mesh statistics
    num_vertices = len(mesh_tri.vertices)
    num_faces = len(mesh_tri.faces)
    shape_scale = float(mesh_tri.vertices.std(0).mean())*3
    center = np.array(mesh_tri.vertices.mean(0))
    shape_scale_mul = opt_shape_scale/shape_scale
    
    print(f"Model file:     {mesh_file}")
    print(f"Vertices:       {num_vertices:,}")
    print(f"Faces:          {num_faces:,}")
    print(f"Scale factor:   {shape_scale:.4f} ({shape_scale/1.18:.2f}x reference cow)")
    print(f"Center:         [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print()
    
    # Setup rendering parameters
    print("📷 CAMERA CONFIGURATION")
    print("-" * 80)
    
    focal_length = 0.5*image_size[0]/np.tan((np.pi/180.0)*vfov_degrees/2)
    cx = (image_size[1]-1)/2
    cy = (image_size[0]-1)/2
    
    print(f"Number of views:    {num_views}")
    print(f"Image resolution:   {image_size[1]}×{image_size[0]} pixels")
    print(f"Vertical FOV:       {vfov_degrees}°")
    print(f"Focal length:       {focal_length:.2f} pixels")
    print(f"Principal point:    ({cx:.1f}, {cy:.1f})")
    print()
    
    # Generate random camera poses
    np.random.seed(random_seed)
    rand_quats = np.random.randn(num_views, 4)
    rand_quats = rand_quats/np.linalg.norm(rand_quats, axis=1, keepdims=True)
    
    # ============================================================================
    # RENDER REFERENCE VIEWS
    # ============================================================================
    
    print("🎨 RENDERING REFERENCE VIEWS")
    print("-" * 80)
    mesh = pyrender.Mesh.from_trimesh(mesh_tri)
    ref_colors = []
    ref_depths = []
    scene = pyrender.Scene()
    scene.add(mesh)
    
    trans = []
    render_start = time.perf_counter()
    
    for i, quat in enumerate(tqdm(rand_quats, desc="Rendering", unit="view")):
        R = transforms3d.quaternions.quat2mat(quat)
        loc = np.array([0, 0, 3*shape_scale]) @ R + center
        trans.append(loc)
        pose = np.vstack([np.vstack([R, loc]).T, np.array([0, 0, 0, 1])])
    
        light = pyrender.SpotLight(color=np.ones(3), intensity=50.0,
                                    innerConeAngle=np.pi/16.0,
                                    outerConeAngle=np.pi/6.0)
        scene.add(light, pose=pose)
    
        camera = pyrender.IntrinsicsCamera(focal_length, focal_length, cx, cy,
                                           znear=0.1*shape_scale, zfar=100*shape_scale)
        scene.add(camera, pose=pose)
    
        r = pyrender.OffscreenRenderer(image_size[1], image_size[0])
        color, target_depth = r.render(scene)
        target_depth[target_depth == 0] = np.nan
        ref_colors.append(color)
        ref_depths.append(target_depth)
        
        for node in list(scene.light_nodes):
            scene.remove_node(node)
            time.sleep(0.01)
        for node in list(scene.camera_nodes):
            scene.remove_node(node)
            time.sleep(0.01)
    
    render_time = (time.perf_counter() - render_start) * 1000
    print(f"\n✓ Rendered {num_views} views in {render_time:.1f} ms")
    print(f"  Average per view: {render_time/num_views:.2f} ms")
    print()
    
    # Save reference renders
    target_sil = (~np.isnan(ref_depths)).astype(np.float32)
    
    image_grid(ref_colors, rows=4, cols=5, rgb=True)
    plt.savefig(f'{output_dir}/01_reference_colors.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    image_grid(ref_depths, rows=4, cols=5, rgb=False)
    plt.savefig(f'{output_dir}/02_reference_depths.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    image_grid(target_sil, rows=4, cols=5, rgb=False, cmap='Greys')
    plt.gcf().subplots_adjust(top=0.92)
    plt.suptitle('Reference Masks')
    plt.savefig(f'{output_dir}/03_reference_masks.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("💾 Saved reference renders:")
    print(f"   • {output_dir}/01_reference_colors.png")
    print(f"   • {output_dir}/02_reference_depths.png")
    print(f"   • {output_dir}/03_reference_masks.png")
    print()
    
    # ============================================================================
    # SETUP FUZZY METABALLS RENDERER
    # ============================================================================
    
    print("⚙️  FUZZY METABALLS RENDERER SETUP")
    print("-" * 80)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    import jax
    import jax.numpy as jnp
    import fm_render
    
    # Check JAX configuration
    print(f"JAX version:        {jax.__version__}")
    print(f"JAX backend:        {jax.default_backend().upper()}")
    print(f"JAX devices:        {jax.devices()}")
    print()
    
    hyperparams = fm_render.hyperparams
    beta2 = jnp.float32(np.exp(hyperparams[0]))
    beta3 = jnp.float32(np.exp(hyperparams[1]))
    
    print(f"Mixture components: {NUM_MIXTURE}")
    print(f"Hyperparameter β₂:  {float(beta2):.4f}")
    print(f"Hyperparameter β₃:  {float(beta3):.4f}")
    print(f"Parameters total:   {NUM_MIXTURE * 13} (40 mixtures × 13 params each)")
    print(f"  • Means:          {NUM_MIXTURE * 3} parameters (3D positions)")
    print(f"  • Precisions:     {NUM_MIXTURE * 9} parameters (3×3 matrices)")
    print(f"  • Weights:        {NUM_MIXTURE * 1} parameters (log weights)")
    print()
    
    render_jit = jax.jit(fm_render.render_func_rays)
    
    # ============================================================================
    # INITIALIZE FUZZY METABALLS
    # ============================================================================
    
    print("🎲 INITIALIZING FUZZY METABALLS (Random Gaussian Cloud)")
    print("-" * 80)
    
    rand_mean = center + np.random.multivariate_normal(
        mean=[0, 0, 0], cov=1e-2*np.identity(3)*shape_scale, size=NUM_MIXTURE)
    rand_weight_log = jnp.log(np.ones(NUM_MIXTURE)/NUM_MIXTURE) + jnp.log(gmm_init_scale)
    rand_prec = jnp.array([np.identity(3)*rand_sphere_size/shape_scale 
                           for _ in range(NUM_MIXTURE)])
    
    print(f"Initialization:     Random Gaussians near center")
    print(f"Mean position:      {center} ± {np.sqrt(1e-2*shape_scale):.4f}")
    print(f"Sphere radius:      {1.0/np.sqrt(rand_sphere_size/shape_scale):.4f}")
    print()
    
    # Setup camera rays
    height, width = image_size
    K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
    pixel_list = (np.array(np.meshgrid(np.arange(width), height-np.arange(height)-1, [0]))[:, :, :, 0]).reshape((3, -1)).T
    camera_rays = (pixel_list - K[:, 2])/np.diag(K)
    camera_rays[:, -1] = -1
    cameras_list = []
    for tran, quat in zip(trans, rand_quats):
        R = transforms3d.quaternions.quat2mat(quat)
        camera_rays2 = camera_rays @ R
        t = np.tile(tran[None], (camera_rays2.shape[0], 1))
        rays_trans = np.stack([camera_rays2, t], 1)
        cameras_list.append(rays_trans)
    
    print(f"Camera rays:        {len(cameras_list)} views × {camera_rays.shape[0]:,} rays/view")
    print(f"Total rays:         {len(cameras_list) * camera_rays.shape[0]:,}")
    print()
    
    # ============================================================================
    # BENCHMARK INITIAL FORWARD PASS (WITH PROPER WARMUP)
    # ============================================================================
    
    print("⏱️  BENCHMARKING FORWARD PASSES")
    print("-" * 80)
    
    # Warmup JIT compilation (run multiple times)
    print("Warming up forward pass JIT compilation...")
    for _ in range(3):
        _ = render_jit(rand_mean, rand_prec, rand_weight_log, cameras_list[0], 
                       beta2/shape_scale, beta3)
    print("✓ Forward pass JIT warmup complete")
    print()
    
    # Benchmark forward passes
    print(f"Running {len(cameras_list)} forward passes...")
    alpha_results_rand = []
    alpha_results_rand_depth = []
    forward_times = []
    
    for i, camera_rays in enumerate(cameras_list):
        t_start = time.perf_counter()
        est_depth, est_alpha, est_norm, est_w = render_jit(
            rand_mean, rand_prec, rand_weight_log, camera_rays, 
            beta2/shape_scale, beta3)
        est_alpha.block_until_ready()  # Ensure computation completes
        t_elapsed = (time.perf_counter() - t_start) * 1000
        forward_times.append(t_elapsed)
        
        alpha_results_rand.append(est_alpha.reshape(image_size))
        est_depth = np.array(est_depth)
        est_depth[est_alpha < 0.5] = np.nan
        alpha_results_rand_depth.append(est_depth.reshape(image_size))
    
    avg_forward = np.mean(forward_times)
    std_forward = np.std(forward_times)
    min_forward = np.min(forward_times)
    max_forward = np.max(forward_times)
    
    print()
    print(f"📊 Forward Pass Statistics ({num_views} renders):")
    print(f"   Mean:   {avg_forward:.3f} ms/frame ± {std_forward:.3f} ms")
    print(f"   Min:    {min_forward:.3f} ms/frame")
    print(f"   Max:    {max_forward:.3f} ms/frame")
    print(f"   Total:  {sum(forward_times):.1f} ms for {num_views} frames")
    print(f"   Pixels: {image_size[0]*image_size[1]:,} pixels/frame → {(image_size[0]*image_size[1]*1000/avg_forward)/1e6:.2f} Mpix/sec")
    print()
    
    # Save initial renderings
    image_grid(alpha_results_rand, rows=4, cols=5, rgb=False, cmap='Greys')
    plt.gcf().subplots_adjust(top=0.92)
    plt.suptitle('Random Init Masks')
    plt.savefig(f'{output_dir}/04_random_init_masks.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    image_grid(alpha_results_rand_depth, rows=4, cols=5, rgb=False)
    plt.gcf().subplots_adjust(top=0.92)
    plt.suptitle('SFS Fuzzy Metaball Initialization')
    plt.savefig(f'{output_dir}/05_random_init_depth.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("💾 Saved initial renderings:")
    print(f"   • {output_dir}/04_random_init_masks.png")
    print(f"   • {output_dir}/05_random_init_depth.png")
    print()
    
    # ============================================================================
    # SETUP OPTIMIZATION
    # ============================================================================
    
    print("🎯 OPTIMIZATION SETUP")
    print("-" * 80)
    
    def objective(params, true_alpha):
        means, prec, weights_log, camera_rays, beta2, beta3 = params
        render_res = render_jit(means, prec, weights_log, camera_rays, beta2, beta3)
    
        est_alpha = render_res[1]
        est_alpha = jnp.clip(est_alpha, clip_alpha, 1-clip_alpha)
        mask_loss = -((true_alpha * jnp.log(est_alpha)) + (1-true_alpha)*jnp.log(1-est_alpha))
        return mask_loss.mean()
    
    grad_render3 = jax.jit(jax.value_and_grad(objective))
    
    from jax.example_libraries import optimizers
    
    def irc(x): 
        return int(round(x))
    
    all_cameras = jnp.array(cameras_list).reshape((-1, 2, 3))
    all_sils = jnp.array(target_sil.ravel()).astype(jnp.float32)
    
    Niter_epoch = int(np.ceil(len(all_cameras)/batch_size))
    
    print(f"Optimizer:          Adam with adaptive learning rate")
    print(f"Initial LR:         {initial_lr}")
    print(f"Loss function:      Binary cross-entropy (silhouette)")
    print(f"Epochs:             {Nepochs}")
    print(f"Batch size:         {batch_size} rays")
    print(f"Iterations/epoch:   {Niter_epoch}")
    print(f"Total iterations:   {Nepochs * Niter_epoch}")
    print(f"Total rays:         {len(all_cameras):,}")
    print()
    
    vecM = jnp.array([[1, 1, 1], [shape_scale_mul, shape_scale_mul, shape_scale_mul]])[None]
    
    adjust_lr = DegradeLR(initial_lr, 
                         config['optimization']['lr_decay_p_thresh'],
                         irc(Niter_epoch*0.4), 
                         irc(config['optimization']['lr_decay_p_window']), 
                         config['optimization']['lr_decay_slope_less'],
                         config['optimization']['lr_decay_max_drops'])
    opt_init, opt_update, opt_params = optimizers.adam(adjust_lr.step_func)
    tmp = [rand_mean*shape_scale_mul, rand_prec/shape_scale_mul, rand_weight_log]
    opt_state = opt_init(tmp)
    
    # Warmup gradient computation (WITH PROPER WARMUP)
    print("Warming up backward pass JIT compilation...")
    p = opt_params(opt_state)
    idx_sample = jnp.array(list(range(min(batch_size, len(all_cameras)))))
    
    # Run multiple warmup iterations
    for _ in range(3):
        val, g = grad_render3([p[0], p[1], p[2], vecM*all_cameras[idx_sample], 
                              beta2/opt_shape_scale, beta3], all_sils[idx_sample])
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), g)
    
    print("✓ Backward pass JIT warmup complete")
    print()
    
    # ============================================================================
    # RUN OPTIMIZATION WITH BENCHMARKING
    # ============================================================================
    
    print("🚀 RUNNING OPTIMIZATION")
    print("=" * 80)
    print()
    
    rand_idx = np.arange(len(all_cameras))
    losses = []
    done = False
    iteration_count = 0
    
    # Benchmark backward passes
    backward_times = []
    
    opt_start_time = time.perf_counter()
    
    for i in range(Nepochs):
        np.random.shuffle(rand_idx)
        rand_idx_jnp = jnp.array(rand_idx)
        
        epoch_start = time.perf_counter()
        
        for j in range(Niter_epoch):
            p = opt_params(opt_state)
            idx = jax.lax.dynamic_slice(rand_idx_jnp, [j*batch_size], [batch_size])
            
            # Benchmark forward + backward pass
            t_start = time.perf_counter()
            val, g = grad_render3([p[0], p[1], p[2], vecM*all_cameras[idx], 
                                  beta2/opt_shape_scale, beta3], all_sils[idx])
            # Ensure computation completes
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), g)
            t_elapsed = (time.perf_counter() - t_start) * 1000
            backward_times.append(t_elapsed)
            
            opt_state = opt_update(i, g[:3], opt_state)
    
            val = float(val)
            losses.append(val)
            iteration_count += 1
            
            if adjust_lr.add(val):
                done = True
                break
        
        epoch_time = (time.perf_counter() - epoch_start) * 1000
        print(f"Epoch {i+1:2d}/{Nepochs} | Loss: {losses[-1]:.4f} | Time: {epoch_time:6.1f} ms")
        
        if done:
            print()
            print("✓ Early stopping triggered (learning rate schedule complete)")
            break
    
    opt_total_time = (time.perf_counter() - opt_start_time) * 1000
    
    print()
    print("=" * 80)
    print("📊 OPTIMIZATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Total time:         {opt_total_time:.1f} ms ({opt_total_time/1000:.2f} seconds)")
    print(f"Total iterations:   {iteration_count}")
    print(f"Average iter time:  {opt_total_time/iteration_count:.2f} ms")
    print(f"Final loss:         {losses[-1]:.6f}")
    print(f"Initial loss:       {losses[0]:.6f}")
    print(f"Loss reduction:     {(1 - losses[-1]/losses[0])*100:.1f}%")
    print()
    
    # ============================================================================
    # BACKWARD PASS STATISTICS
    # ============================================================================
    
    print("📊 BACKWARD PASS STATISTICS (Forward + Gradient)")
    print("-" * 80)
    
    avg_backward = np.mean(backward_times)
    std_backward = np.std(backward_times)
    min_backward = np.min(backward_times)
    max_backward = np.max(backward_times)
    
    print(f"Samples:    {len(backward_times)} backward passes")
    print(f"Mean:       {avg_backward:.3f} ms/batch ± {std_backward:.3f} ms")
    print(f"Min:        {min_backward:.3f} ms/batch")
    print(f"Max:        {max_backward:.3f} ms/batch")
    print(f"Batch size: {batch_size} rays")
    print(f"Per ray:    {avg_backward/batch_size:.6f} ms/ray ({batch_size*1000/avg_backward:.1f} rays/sec)")
    print()
    
    print("⚡ PERFORMANCE COMPARISON vs. Mesh Renderers (Paper Claims)")
    print("-" * 80)
    print(f"Forward pass:   {avg_forward:.3f} ms")
    print(f"Backward pass:  {avg_backward:.3f} ms")
    print(f"Speedup ratio:  {avg_backward/avg_forward:.1f}x (backward vs forward)")
    print()
    print("💡 Note: Paper compared against mesh-based differentiable renderers")
    print("   (PyTorch3D, SoftRasterizer, etc.) on equivalent tasks")
    print()
    
    # Get final parameters
    final_mean, final_prec, final_weight_log = opt_params(opt_state)
    final_mean /= shape_scale_mul
    final_prec *= shape_scale_mul
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.title('Convergence Plot - Shape from Silhouette', fontsize=14, fontweight='bold')
    plt.plot(losses, marker='.', lw=0, ms=5, alpha=0.5, color='#2196F3')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Binary Cross-Entropy Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============================================================================
    # RENDER FINAL RESULTS
    # ============================================================================
    
    print("🎨 RENDERING FINAL RESULTS")
    print("-" * 80)
    
    alpha_results_final = []
    alpha_results_depth = []
    
    for camera_rays in cameras_list:
        est_depth, est_alpha, est_norms, est_w = render_jit(
            final_mean, final_prec, final_weight_log, camera_rays, 
            beta2/shape_scale, beta3)
        alpha_results_final.append(est_alpha.reshape(image_size))
        
        est_depth = np.array(est_depth)
        est_depth[est_alpha < 0.5] = np.nan
        alpha_results_depth.append(est_depth.reshape(image_size))
    
    # Save final results
    image_grid(target_sil, rows=4, cols=5, rgb=False)
    plt.gcf().subplots_adjust(top=0.92)
    plt.suptitle('Reference Masks', fontsize=14, fontweight='bold')
    plt.savefig(f'{output_dir}/07_final_reference_masks.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    image_grid(alpha_results_final, rows=4, cols=5, rgb=False)
    plt.gcf().subplots_adjust(top=0.92)
    plt.suptitle('Final Optimized Masks', fontsize=14, fontweight='bold')
    plt.savefig(f'{output_dir}/08_final_masks.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Depth comparison
    vmin = np.nanmin(np.array(ref_depths))
    vmax = np.nanmax(np.array(ref_depths))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(alpha_results_depth[3], vmin=vmin, vmax=vmax, cmap='viridis')
    ax1.set_title('Estimated Depth (View 3)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(ref_depths[3], vmin=vmin, vmax=vmax, cmap='viridis')
    ax2.set_title('Ground Truth Depth (View 3)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/09_depth_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    image_grid(alpha_results_depth, rows=4, cols=5, rgb=False, vmin=vmin, vmax=vmax)
    plt.gcf().subplots_adjust(top=0.92)
    plt.suptitle('Final Depth Maps - All Views', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/10_all_depth_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Rendered {len(cameras_list)} final views")
    print()
    
    # Save model
    if config['io']['save_model']:
        import pickle
        model_path = f"{output_dir}/{config['io']['model_filename']}"
        with open(model_path, 'wb') as fp:
            pickle.dump([final_mean, final_prec, final_weight_log], fp)
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    
    print("💾 SAVED OUTPUTS")
    print("-" * 80)
    output_files = [
        "01_reference_colors.png      - Ground truth color renders",
        "02_reference_depths.png      - Ground truth depth maps", 
        "03_reference_masks.png       - Ground truth silhouettes",
        "04_random_init_masks.png     - Initial random Gaussian masks",
        "05_random_init_depth.png     - Initial random Gaussian depths",
        "06_convergence.png           - Optimization convergence plot",
        "07_final_reference_masks.png - Final ground truth masks",
        "08_final_masks.png           - Final optimized masks",
        "09_depth_comparison.png      - Depth comparison (estimated vs GT)",
        "10_all_depth_results.png     - All final depth maps",
    ]
    if config['io']['save_model']:
        output_files.append(f"{config['io']['model_filename']}          - Trained model parameters")
    
    for f in output_files:
        print(f"   • {f}")
    print()
    
    print("=" * 80)
    print("✨ RECONSTRUCTION COMPLETE")
    print("=" * 80)
    print()
    print(f"📈 Final Statistics:")
    print(f"   Model:              {NUM_MIXTURE} Gaussian mixtures ({NUM_MIXTURE*13} parameters)")
    print(f"   Training views:     {num_views}")
    print(f"   Image resolution:   {image_size[1]}×{image_size[0]} pixels")
    print(f"   Optimization time:  {opt_total_time/1000:.2f} seconds")
    print(f"   Final loss:         {losses[-1]:.6f}")
    print(f"   Avg forward time:   {avg_forward:.3f} ms/frame")
    print(f"   Avg backward time:  {avg_backward:.3f} ms/batch ({batch_size} rays)")
    print()
    print("🔗 For more details, see:")
    print("   Paper:   https://arxiv.org/abs/2207.10606")
    print("   Project: https://leonidk.github.io/fuzzy-metaballs")
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
