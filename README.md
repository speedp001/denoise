# Point Cloud Cleaning

Point cloud denoising and refinement framework for multi-format 3D sensor data.

![Denoising Pipeline](https://i.imgur.com/TrxcBBI.png)

---

# Step 1: Denoising

## PointCleanNet
- Iterative point cloud denoising framework for object-level noise removal.
- Supports `.xyz` point cloud processing.

## Score-Based Point Cloud Denoising
- Score-based diffusion denoising framework for noisy point clouds.
- `.ply ↔ .xyz` conversion utilities implemented.

## Noise2Score3D
- Unsupervised point cloud denoising without clean ground-truth data.
- Can be adapted to domain-specific datasets through training.

## Classical Methods
- SOR
- ROR
- DBSCAN

---

# Step 2: Refinement

## IterativePFN
- Iterative point filtering framework for geometric refinement.
- `.ply → .npy` conversion utilities implemented.

## PointFilter
- Encoder-decoder-based point cloud filtering framework.
- Pretrained models available for object-level denoising.

## P2P-Bridge
- Diffusion bridge framework for point cloud denoising and refinement.
- Supports pretrained inference pipelines.

## BuildAnyPoint
- Unified point cloud reconstruction and refinement framework.
- Official implementation recently released.

## U-CAN
- Unsupervised consistency-aware point cloud denoising framework.
- Preserves structural consistency without clean supervision.

---

# Step 3: Completion / Upsampling (Optional)

## Seen2Scene
- Visibility-guided scene completion framework for realistic 3D environments.
- Generates coherent large-scale scene geometry from partial scans.

## SuperPC
- Unified diffusion framework for completion, denoising, upsampling, and colorization.
- Uses multi-level spatial feature fusion for robust reconstruction.

## NKSR (Neural Kernel Surface Reconstruction)
- Neural implicit surface reconstruction framework for sparse and noisy point clouds.
- Combines kernel interpolation and learned priors for scalable reconstruction.

---

# Utilities

- `.ply ↔ .xyz` conversion
- `.ply ↔ .npy` conversion
- Point cloud preprocessing tools
