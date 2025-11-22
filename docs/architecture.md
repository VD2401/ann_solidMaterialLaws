# 3D U-Net Architecture

## Overview

This document describes the 3D U-Net architecture implemented in `src/ann_solid_materials/model.py` for predicting stress fields from material microstructure.

## Architecture Diagram

```
Input: (batch, 1, 64, 64, 64)
    |
    v
[Encoder Level 1] ──────────────┐ 
    64x64x64 (width1)            │ Skip
    |                             │ Connection
    v (MaxPool 2x2x2)            │
[Encoder Level 2] ─────────────┐│
    32x32x32 (width2)           ││
    |                            ││
    v (MaxPool 2x2x2)           ││
[Encoder Level 3] ────────────┐││
    16x16x16 (width3)          │││
    |                           │││
    v (MaxPool 2x2x2)          │││
[Encoder Level 4] ───────────┐│││
    8x8x8 (width4)            ││││
    |                          ││││
    v (MaxPool 2x2x2)         ││││
[Bottleneck]                  ││││
    4x4x4 (width5)            ││││
    |                          ││││
    v (TransposeConv)         ││││
[Decoder Level 4] <───────────┘│││
    8x8x8 (width4)              │││
    |                            │││
    v (TransposeConv)           │││
[Decoder Level 3] <─────────────┘││
    16x16x16 (width3)             ││
    |                              ││
    v (TransposeConv)             ││
[Decoder Level 2] <───────────────┘│
    32x32x32 (width2)              │
    |                               │
    v (TransposeConv)              │
[Decoder Level 1] <────────────────┘
    64x64x64 (width1)
    |
    v
[Output Conv 1x1x1]
    |
    v
Output: (batch, 1, 64, 64, 64)
```

## Network Configuration

### Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| width1 | 4 | Filters at level 1 (finest) |
| width2 | 8 | Filters at level 2 |
| width3 | 16 | Filters at level 3 |
| width4 | 32 | Filters at level 4 |
| width5 | 64 | Filters at bottleneck |
| Activation | LeakyReLU | Non-linearity |
| Padding | Circular | For periodic BC |
| Pooling | MaxPool 2×2×2 | Downsampling |
| Upsampling | TransposeConv 2×2×2 | Upsampling |

### Parameter Count

For default configuration:
- **Total parameters**: ~140,000
- Breakdown:
  - Encoder: ~45,000
  - Bottleneck: ~50,000
  - Decoder: ~45,000

## Layer-by-Layer Description

### Encoder Path

Each encoder level consists of:
1. **Conv3D** (3×3×3, circular padding)
2. **LeakyReLU** activation
3. **Conv3D** (3×3×3, circular padding)
4. **LeakyReLU** activation
5. **MaxPool3D** (2×2×2, stride 2) - except at bottleneck

**Level 1** (64³ → 32³):
- Input: (batch, 1, 64, 64, 64)
- Conv: 1 → width1, 3×3×3
- Conv: width1 → width1, 3×3×3
- Pool: 2×2×2 stride 2
- Output: (batch, width1, 32, 32, 32)

**Level 2** (32³ → 16³):
- Input: (batch, width1, 32, 32, 32)
- Conv: width1 → width2, 3×3×3
- Conv: width2 → width2, 3×3×3
- Pool: 2×2×2 stride 2
- Output: (batch, width2, 16, 16, 16)

**Level 3** (16³ → 8³):
- Input: (batch, width2, 16, 16, 16)
- Conv: width2 → width3, 3×3×3
- Conv: width3 → width3, 3×3×3
- Pool: 2×2×2 stride 2
- Output: (batch, width3, 8, 8, 8)

**Level 4** (8³ → 4³):
- Input: (batch, width3, 8, 8, 8)
- Conv: width3 → width4, 3×3×3
- Conv: width4 → width4, 3×3×3
- Pool: 2×2×2 stride 2
- Output: (batch, width4, 4, 4, 4)

### Bottleneck (4³ → 4³)

- Input: (batch, width4, 4, 4, 4)
- Conv: width4 → width5, 3×3×3
- Conv: width5 → width5, 3×3×3
- Output: (batch, width5, 4, 4, 4)

### Decoder Path

Each decoder level consists of:
1. **TransposeConv3D** (2×2×2, stride 2) - upsampling
2. **Concatenation** with encoder feature maps (skip connection)
3. **Conv3D** (3×3×3, circular padding)
4. **LeakyReLU** activation
5. **Conv3D** (3×3×3, circular padding)
6. **LeakyReLU** activation

**Level 4** (4³ → 8³):
- Input: (batch, width5, 4, 4, 4)
- TransposeConv: width5 → width4, 2×2×2 stride 2
- After upsampling: (batch, width4, 8, 8, 8)
- Concatenate with encoder level 4 skip: (batch, width5, 8, 8, 8)
- Conv: width5 → width4, 3×3×3
- Conv: width4 → width4, 3×3×3
- Output: (batch, width4, 8, 8, 8)

**Level 3** (8³ → 16³):
- Input: (batch, width4, 8, 8, 8)
- TransposeConv: width4 → width3, 2×2×2 stride 2
- After upsampling: (batch, width3, 16, 16, 16)
- Concatenate with encoder level 3 skip: (batch, width4, 16, 16, 16)
- Conv: width4 → width3, 3×3×3
- Conv: width3 → width3, 3×3×3
- Output: (batch, width3, 16, 16, 16)

**Level 2** (16³ → 32³):
- Input: (batch, width3, 16, 16, 16)
- TransposeConv: width3 → width2, 2×2×2 stride 2
- After upsampling: (batch, width2, 32, 32, 32)
- Concatenate with encoder level 2 skip: (batch, width3, 32, 32, 32)
- Conv: width3 → width2, 3×3×3
- Conv: width2 → width2, 3×3×3
- Output: (batch, width2, 32, 32, 32)

**Level 1** (32³ → 64³):
- Input: (batch, width2, 32, 32, 32)
- TransposeConv: width2 → width1, 2×2×2 stride 2
- After upsampling: (batch, width1, 64, 64, 64)
- Concatenate with encoder level 1 skip: (batch, width2, 64, 64, 64)
- Conv: width2 → width1, 3×3×3
- Conv: width1 → width1, 3×3×3
- Output: (batch, width1, 64, 64, 64)

### Output Layer

- Input: (batch, width1, 64, 64, 64)
- Conv: width1 → 1, 1×1×1 (no padding)
- Output: (batch, 1, 64, 64, 64)
- **No activation** at output (linear layer for regression)

## Design Choices

### Circular Padding

We use **circular padding** instead of zero padding because:
1. The physical problem has **periodic boundary conditions**
2. Avoids edge artifacts
3. Better represents the infinite periodic microstructure

### LeakyReLU Activation

- **LeakyReLU**(negative_slope=0.01) instead of ReLU
- Prevents dying neurons
- Better gradient flow for negative values

### Skip Connections

- Concatenation (not addition) of encoder features to decoder
- Preserves both low-level and high-level features
- Critical for accurate spatial localization

### MaxPooling

- Simple and effective downsampling
- Reduces spatial dimensions by factor of 2
- Reduces computational cost exponentially with depth

## Model Variants

### Narrow Model

```python
model = UNet3D(width1=2, width2=4, width3=8, width4=16, width5=32)
# ~35K parameters - faster training, less capacity
```

### Default Model

```python
model = UNet3D(width1=4, width2=8, width3=16, width4=32, width5=64)
# ~140K parameters - good balance
```

### Wide Model

```python
model = UNet3D(width1=8, width2=16, width3=32, width4=64, width5=128)
# ~560K parameters - higher capacity, slower training
```

### Very Wide Model

```python
model = UNet3D(width1=16, width2=32, width3=64, width4=128, width5=256)
# ~2.2M parameters - maximum capacity
```

## Computational Complexity

### Memory Requirements

For default model with batch size B:

| Stage | Resolution | Channels | Memory (approx) |
|-------|------------|----------|-----------------|
| Input | 64³ | 1 | B × 1 MB |
| Level 1 | 64³ | 4 | B × 4 MB |
| Level 2 | 32³ | 8 | B × 1 MB |
| Level 3 | 16³ | 16 | B × 0.25 MB |
| Level 4 | 8³ | 32 | B × 0.06 MB |
| Bottleneck | 4³ | 64 | B × 0.015 MB |

**Total GPU memory**: ~10-15 MB per sample (forward pass)

During training with gradients and optimizer states: ~50-100 MB per sample

### Computational Cost

- **FLOPs** (forward pass): ~1-2 GFLOPs per sample
- **Training time** (GPU): ~0.5-2 seconds per epoch (128 samples, batch size 8)
- **Inference time** (GPU): ~5-10 ms per sample

## References

1. **Original U-Net**: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In MICCAI.

2. **3D U-Net**: Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). 3D U-Net: learning dense volumetric segmentation from sparse annotation. In MICCAI.

3. **Application to materials**: Various works applying CNNs to computational mechanics and materials science.
