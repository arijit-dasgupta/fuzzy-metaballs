# Fuzzy Metaballs Differentiable Renderer

**Note:** This is a fork of the original [Fuzzy Metaballs repository](https://github.com/leonidk/fuzzy-metaballs) by Leonid Keselman and Martial Hebert (CMU), converted to use [Pixi](https://pixi.sh) for dependency management with exact version pinning and CUDA support.

[Original Project Page](https://leonidk.github.io/fuzzy-metaballs/) | [Paper (ECCV 2022)](https://arxiv.org/abs/2207.10606)

## About

Fuzzy Metaballs is an approximate differentiable renderer for compact, interpretable 3D representations. The method focuses on rendering shapes via depth maps and silhouettes, sacrificing fidelity for utility with fast runtimes and high-quality gradient information.

**Key features:**
- Forward passes ~5x faster than mesh-based renderers
- Backward passes ~30x faster than mesh-based renderers
- Smooth, differentiable depth maps and silhouettes
- 40 Gaussian mixture components (520 parameters)
- GPU-accelerated with JAX

## Quick Start

### 1. Clone and Navigate

```bash
git clone git@github.com:arijit-dasgupta/fuzzy-metaballs.git
cd fuzzy-metaballs
```

### 2. Install Pixi

```bash
# Linux/macOS
curl -fsSL https://pixi.sh/install.sh | bash

# macOS with Homebrew
brew install pixi

# Restart your shell or source profile
source ~/.bashrc  # or ~/.zshrc
```

### 3. Install Dependencies

```bash
# Install all dependencies (Python packages + system libs + CUDA)
pixi install
```

This installs everything with exact versions:
- Python 3.11.14 with scientific computing stack
- JAX 0.8.0 with CUDA12 support
- 3D rendering libraries (PyRender, Trimesh)
- System dependencies (OpenGL, Xvfb) - no sudo needed!

### 4. Run Shape from Silhouette Demo

Using pixi shell (recommended):

```bash
pixi shell
python run_fmb.py
```

Or directly:

```bash
pixi run python run_fmb.py
```

Results saved to `output/` directory with 10 visualization images and trained model.

## Usage Examples

### Default Configuration

Runs with settings from `config.yaml`:

```bash
pixi run python run_fmb.py
```

### CPU Mode

Force JAX to use CPU (useful for debugging or machines without GPU):

```bash
pixi run python run_fmb.py --cpu
```

### Custom Parameters

Override specific parameters:

```bash
# Use 80 mixture components instead of 40
pixi run python run_fmb.py --num-mixtures 80

# Higher resolution images
pixi run python run_fmb.py --image-width 128 --image-height 128

# More training epochs
pixi run python run_fmb.py --num-epochs 20

# Smaller batch size (less memory)
pixi run python run_fmb.py --batch-size 400

# Different learning rate
pixi run python run_fmb.py --initial-lr 0.05
```

### Multiple Parameters

Combine multiple overrides:

```bash
pixi run python run_fmb.py \
    --num-mixtures 60 \
    --num-views 40 \
    --batch-size 1000 \
    --initial-lr 0.2 \
    --cpu
```

### Custom Config File

Use a different configuration file:

```bash
pixi run python run_fmb.py --config my_config.yaml
```

## Configuration

All hyperparameters are in `config.yaml`:

```yaml
model:
  num_mixtures: 40              # Gaussian mixture components
  gmm_init_scale: 1.0
  rand_sphere_size: 30

rendering:
  num_views: 20                 # Camera views
  image_width: 64
  image_height: 64
  vfov_degrees: 45

optimization:
  num_epochs: 10
  batch_size: 800
  initial_lr: 0.1
  opt_shape_scale: 2.2
```

Command-line arguments override YAML settings.

## Using Jupyter Notebooks

Start Jupyter Lab with the Pixi environment:

```bash
pixi run jupyter
```

**Note:** All notebooks automatically use the Pixi environment (`.pixi/envs/default/bin/python`). No kernel registration needed - just open `shape_from_silhouette.ipynb` and all packages (JAX with CUDA, NumPy, PyRender, etc.) are available.
