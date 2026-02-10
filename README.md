# MeanFlow for 3D Voxel Generation

Final project in DD2610 - Advanced Deep Learning

## Overview

This project adapts [MeanFlow](https://arxiv.org/abs/2505.13447) (Geng et al., 2025), a one-step generative model based on modeling average velocity fields, from 2D image generation to 3D voxel generation. We apply the method to the 3D MNIST dataset and conduct comprehensive ablation studies on key hyperparameters.

**Key Features:**
- Single-step 3D voxel generation
- Class-conditional generation (10 digit classes)
- Latent-space modeling with VAE compression
- Two implementations: JAX and PyTorch

## Implementations

This repository contains two separate implementations of the MeanFlow adaptation:

### [JAX Implementation](./meanflow-jax) (Primary)

**Status:** Complete with full documentation

The JAX implementation is feature-complete and includes:
- Full training pipeline with 5 ablation configurations
- Comprehensive evaluation metrics (Chamfer distance, IoU, coverage)
- Rotation-invariant metric matching
- Sample generation and visualization
- Detailed ablation study results

**[Full Documentation →](./meanflow-jax/README.md)**

**Quick Start:**
```bash
cd meanflow-jax
pip install -r requirements.txt
python data/convert_pt.py
python main.py --config=configs/load_config.py:train_3d_v1 --workdir=./workdir_3d_v1
```

### [PyTorch Implementation](./meanflow-pytorch)

**Status:** In Development

The PyTorch implementation is currently under development.

**[Documentation →](./meanflow-pytorch/README.md)**

## Results Highlights

From our JAX implementation ablation studies on 3D MNIST:

| Config | Key Change | Chamfer↓ | IoU↑ | Coverage |
|--------|-----------|----------|------|----------|
| V1 (Baseline) | Logit-normal, 75% data, ω=1.0 | 0.120 ± 0.038 | 0.263 ± 0.098 | 10/10 |
| V4 (50% data) | 50% data proportion | 0.118 ± 0.040 | **0.269 ± 0.103** | 10/10 |
| V5 (Strong CFG) | ω=3.0 guidance | **0.116 ± 0.025** | 0.246 ± 0.090 | 10/10 |

**Key Findings:**
- Non-monotonic data/velocity ratio: 50% data proportion achieves best IoU
- Strong CFG (ω=3.0) improves reconstruction consistency but reduces structural overlap
- Rotation-invariant metrics essential for reliable 3D evaluation

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

## License

This project is licensed under the terms specified in the [LICENSE](./LICENSE) file.
