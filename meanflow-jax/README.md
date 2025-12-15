# MeanFlow for 3D Voxel Generation

Adaptation of [MeanFlow](https://arxiv.org/abs/2505.13447) (Geng et al., 2025) from 2D image generation to 3D voxel generation on the 3D MNIST dataset.

## Overview

This project extends MeanFlow, a one-step generative model based on modeling average velocity fields, to generate 3D voxel data. We use the Diffusion Transformer (DiT) architecture and conduct ablation studies on key hyperparameters.

## Interactive 3D Samples

The following links open **interactive Three.js visualizations**
(rotate, zoom, inspect voxels):

- 🔢 **Sample 3** – Generated digit (Step 156k)  
  https://hedemil.github.io/meanflow-jax/docs/sample_003.html

- 🔢 **Sample 4** – Generated digit (Step 156k)  
  https://hedemil.github.io/meanflow-jax/docs/sample_004.html

- 🔢 **Sample 5** – Generated digit (Step 156k)  
  https://hedemil.github.io/meanflow-jax/docs/sample_005.html


**Key Features:**
- Single-step 3D voxel generation
- Class-conditional generation (10 digit classes)
- Latent-space modeling with VAE compression
- Comprehensive ablation studies

## Installation

### Prerequisites
- Python 3.8+
- JAX with GPU support
- NVIDIA GPU (tested on RTX 3090)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd meanflow-jax

# Install dependencies
pip install -r requirements.txt

# Prepare 3D MNIST data
python data/convert_pt.py  # Downloads and prepares the dataset
```

## Project Structure

```
meanflow-jax/
├── configs/                    # Training configurations
│   ├── train_3d_v1.yml        # Baseline (logit-normal, 75% data, ω=1.0)
│   ├── train_3d_v2.yml        # 100% data proportion
│   ├── train_3d_v3.yml        # Uniform time sampling
│   ├── train_3d_v4.yml        # 50% data proportion
│   └── train_3d_v5.yml        # Strong CFG (ω=3.0)
├── data/                       # Datasets
|   └── MNIST/                 # 3D MNIST
│   └── mnist3d_latents/       # VAE-encoded 3D 
│   └── convert_pt.py          # 
├── utils/
│   ├── evaluation_3d.py       # 3D evaluation metrics
│   └── input_pipeline.py      # Data loading
├── train.py                   # Main training script
├── evaluate.py                # Evaluation script
├── generate_3d_samples.py     # Sample generation
└── meanflow.py                # MeanFlow model implementation
```

## Quick Start

### Training

Train a model with a specific configuration:

```bash
# Train baseline model (V1)
python main.py     --config=configs/load_config.py:train_3d_v1     --workdir=./workdir_3d_v1
```

Training takes approximately 4 hours per configuration on an RTX 3090.

### Evaluation

Evaluate a trained checkpoint:

```bash
# Evaluate baseline model
python evaluate.py --config configs/train_3d_v1.yml --checkpoint workdir_3d_v1

# Evaluate with more samples
python evaluate.py --config configs/train_3d_v1.yml --checkpoint workdir_3d_v1 --num_samples 500
```

**Output (with rotation-invariant metrics):**
```
Chamfer Distance: 0.1202 ± 0.0380
Voxel IoU:        0.2631 ± 0.0976
Mode Coverage:    10/10
```

### Generate Samples

Generate and visualize 3D samples:

```bash
# Generate samples for all digits
python generate_3d_samples.py --config configs/train_3d_v1.yml --checkpoint workdir_3d_v1

# Generate specific digits
python generate_3d_samples.py --config configs/train_3d_v1.yml --checkpoint workdir_3d_v1 --class_labels 0 1 2 3 4
```

### Compare Configurations

Evaluate all configurations at once:

```bash
python eval_all_configs.py
```

This generates a comparison table and saves results to `evaluation_results.json`.

## Configuration Guide

### Config File Structure

```yaml
model:
  cls: DiT_S_4           # Model architecture
  input_size: 16         # Spatial resolution
  in_channels: 16        # Depth as channels

dataset:
  name: mnist3d_latent
  root: data/mnist3d_latents
  num_classes: 10

training:
  batch_size: 64
  num_epochs: 500
  learning_rate: 0.0001
  ema_val: 0.99995

method:
  noise_dist: logit_normal  # Time sampling: logit_normal or unit_normal
  P_mean: -0.4
  P_std: 1.0
  data_proportion: 0.75      # Data vs velocity matching ratio
  class_dropout_prob: 0.1    # CFG dropout probability
  omega: 1.0                 # CFG guidance strength
  kappa: 0.5
  norm_p: 1.0
  norm_eps: 0.01

evaluation:
  chamfer_threshold: -0.5
  iou_threshold: -0.5
  chamfer_enabled: true
  iou_enabled: true

sampling:
  num_steps: 1  # Single-step sampling
```

### Key Hyperparameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `noise_dist` | `logit_normal`, `unit_normal` | Time sampling distribution |
| `data_proportion` | 0.0-1.0 | Ratio of data matching vs velocity matching |
| `omega` | ≥ 0.0 | CFG guidance strength (1.0 = baseline, 3.0 = strong) |
| `class_dropout_prob` | 0.0-1.0 | Probability of dropping class labels (enables CFG) |

## Ablation Study Results

We conducted ablation studies on the 3D MNIST dataset using rotation-invariant metrics:

| Config | Time Sampling | Data Prop. | ω | Chamfer↓ | IoU↑ | Coverage |
|--------|---------------|------------|---|----------|------|----------|
| V1 (Baseline) | Logit-normal | 0.75 | 1.0 | 0.120 ± 0.038 | 0.263 ± 0.098 | 10/10 |
| V2 (100% data) | Logit-normal | **1.00** | 1.0 | 0.125 ± 0.039 | 0.248 ± 0.109 | 10/10 |
| V3 (Uniform) | **Uniform** | 0.75 | 1.0 | 0.118 ± 0.037 | 0.261 ± 0.093 | 10/10 |
| V4 (50% data) | Logit-normal | **0.50** | 1.0 | 0.118 ± 0.040 | **0.269 ± 0.103** | 10/10 |
| V5 (Strong CFG) | Logit-normal | 0.75 | **3.0** | **0.116 ± 0.025** | 0.246 ± 0.090 | 10/10 |

### Key Findings

1. **Time Sampling (V3):** Uniform vs logit-normal has minimal impact (1.6% Chamfer improvement, 0.8% IoU reduction), demonstrating MeanFlow's robustness to time parameterization.

2. **Data/Velocity Ratio (V2, V4):** Non-monotonic behavior observed - 50% data (V4) achieves **best IoU** (0.269), outperforming both 75% (V1) and 100% (V2). This suggests balanced data and velocity matching is optimal for 3D reconstruction.

3. **CFG Strength (V5):** Strong guidance (ω=3.0) achieves **best Chamfer** (0.116) with **lowest variance** (0.025 vs 0.038 for V1), indicating more consistent reconstruction. However, IoU drops to 0.246 (6.5% reduction).

### Quality-Consistency Tradeoff

V5 demonstrates that stronger CFG improves reconstruction consistency (lower Chamfer distance with 34% lower variance) but reduces structural overlap (lower IoU). The low variance suggests more similar samples within each class. For applications requiring diverse outputs, use ω=1.0; for consistent high-quality reconstruction, consider ω=3.0.

### Rotation-Invariant Metrics

Our evaluation uses rotation-invariant matching: each generated sample is rotated 0°, 90°, 180°, 270° around the z-axis, and the best rotation (lowest Chamfer or highest IoU) is selected. This corrects for orientation misalignment in the preprocessed data, improving metric reliability by **13× for Chamfer** (from ~1.8 to ~0.12) and **24% for IoU** (from ~0.21 to ~0.26).

## Evaluation Metrics

### Chamfer Distance
Measures point cloud similarity between generated and real samples. **Lower is better.**
- Normalized point clouds (0-1 range)
- Class-aligned: compares generated digit "5" only with real digit "5"
- Rotation-invariant: tries 4 rotations and selects best match
- Typical range: 0.116-0.125 (with rotation correction)

### Voxel IoU (Intersection over Union)
Measures voxel overlap between generated and real samples. **Higher is better.**
- Threshold at -0.5 for foreground voxels
- Class-aligned comparison
- Rotation-invariant: tries 4 rotations and selects best match
- Typical range: 0.246-0.269 (with rotation correction)

### Coverage
Number of digit classes represented in generated samples. **Should be 10/10.**
- Detects mode collapse
- All our configs achieve 10/10 coverage

### Rotation-Invariant Matching
To address orientation misalignment in the preprocessed 3D MNIST data, each generated sample is rotated 0°, 90°, 180°, and 270° around the z-axis. The rotation with the lowest Chamfer distance (or highest IoU) is selected for evaluation. This improves metric reliability significantly compared to naive comparison.

## Architecture Details

### Model: DiT-S/4
- **Parameters:** ~33M
- **Architecture:** Diffusion Transformer (Small variant, patch size 4)
- **Input:** 16×16 spatial with 16 channels (depth dimension)
- **Output:** 16×16×16 voxel grid

### Data Encoding
- 3D MNIST voxels (16×16×16) encoded with VAE
- Latent space: 16×16×16 → compressed representation
- Values range: [-1, 1]

## Troubleshooting

### IoU Values
**Q:** Why is IoU around 0.25-0.27? Shouldn't it be higher?

**A:** IoU of 0.25-0.27 is expected for *generative* tasks with rotation-invariant matching. We're generating **new** plausible digits, not reconstructing exact copies of validation data. Much higher IoU would indicate overfitting. Without rotation correction, naive IoU would be lower (~0.20).

### Mode Collapse
**Q:** How do I know if my model has mode collapse?

**A:** Check coverage metric. Should be 10/10 for 3D MNIST. Also inspect `class_counts` in evaluation output to ensure balanced generation.

### Out of Memory
**Q:** GPU runs out of memory during training.

**A:** Reduce `batch_size` in config (try 32 or 16). Training will take longer but use less memory.

### Orientation Misalignment (Fixed)
**Q:** Why do we need rotation-invariant metrics?

**A:** The preprocessed 3D MNIST dataset has inconsistent axis orientations between samples. This was discovered by visually comparing real vs generated samples and noticing orientation mismatches. By testing all 4 rotations (90° increments around z-axis) and selecting the best match, we correct for this data artifact and obtain reliable metrics. This is implemented in `utils/evaluation_3d.py` with the `rotation_invariant=True` parameter.

## Citation

If you use this code, please cite the original MeanFlow paper:

```bibtex
@article{geng2025meanflow,
  title={Mean Flows for One-step Generative Modeling},
  author={Geng, Zhongjie and others},
  journal={arXiv preprint arXiv:2505.13447},
  year={2025}
}
```

## Acknowledgments

- Original MeanFlow implementation: [GitHub](https://github.com/zhongjiengeng/meanflow)
- 3D MNIST dataset: [Kaggle](https://www.kaggle.com/datasets/daavoo/3d-mnist)
- DiT architecture: [DiT: Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)

## License

This project adapts the MeanFlow codebase for 3D voxel generation. See original MeanFlow repository for license details.
