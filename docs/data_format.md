# Data Format Specification

## Overview

This document describes the expected format for the 3D elasticity dataset used to train the neural network.

## File Naming Convention

Data files follow this naming pattern:

```
data_elasticity_3D_128_{file_idx}_L{load}_S{stress}_{type}.pt
```

Where:
- `file_idx`: File index (0, 1, 2, ...) - each file contains 128 samples
- `load`: Loading condition (0-5)
- `stress`: Stress component (0-5)
- `type`: Either `input` or `output`

### Examples

```
data_elasticity_3D_128_0_L0_S0_input.pt   # File 0, Load 0, Stress 0, Inputs
data_elasticity_3D_128_0_L0_S0_output.pt  # File 0, Load 0, Stress 0, Outputs
data_elasticity_3D_128_1_L0_S0_input.pt   # File 1, Load 0, Stress 0, Inputs
data_elasticity_3D_128_1_L0_S0_output.pt  # File 1, Load 0, Stress 0, Outputs
```

## Data Format

### Input Files (`*_input.pt`)

Each input file is a PyTorch `.pt` file containing a dictionary with:

```python
{
    'input': torch.Tensor  # Shape: (128, 1, 64, 64, 64)
}
```

- **Shape**: `(n_samples, n_channels, nx, ny, nz)`
  - `n_samples`: 128 (samples per file)
  - `n_channels`: 1 (single field)
  - `nx, ny, nz`: 64 (spatial resolution)

- **Data type**: `torch.float32`
- **Values**: Young's modulus field
  - Matrix: E ≈ 1.0
  - Inclusions: E ≈ 0.1
  - Range: typically [0.1, 1.0]

### Output Files (`*_output.pt`)

Each output file is a PyTorch `.pt` file containing a dictionary with:

```python
{
    'output': torch.Tensor  # Shape: (128, 1, 64, 64, 64)
}
```

- **Shape**: `(n_samples, n_channels, nx, ny, nz)`
- **Data type**: `torch.float32`
- **Values**: Stress field component
  - Depends on loading and stress component
  - Typically normalized values

## Loading Conditions

The loading index `L{load}` specifies the applied macroscopic strain:

| Index | Loading      | Description                |
|-------|--------------|----------------------------|
| 0     | ε_xx = 1     | Unit strain in x-direction |
| 1     | ε_yy = 1     | Unit strain in y-direction |
| 2     | ε_zz = 1     | Unit strain in z-direction |
| 3     | ε_yz = 1     | Unit shear strain yz       |
| 4     | ε_zx = 1     | Unit shear strain zx       |
| 5     | ε_xy = 1     | Unit shear strain xy       |

All other strain components are zero.

## Stress Components

The stress index `S{stress}` specifies which stress component is stored:

| Index | Component | Description     |
|-------|-----------|-----------------|
| 0     | σ_xx      | Normal stress x |
| 1     | σ_yy      | Normal stress y |
| 2     | σ_zz      | Normal stress z |
| 3     | σ_yz      | Shear stress yz |
| 4     | σ_zx      | Shear stress zx |
| 5     | σ_xy      | Shear stress xy |

## Microstructure Description

Each sample represents a cubic domain (1×1×1) with:

- **Matrix material**:
  - Young's modulus: E_matrix = 1.0
  - Poisson's ratio: ν = 0.3
  
- **Spherical inclusions** (100 per sample):
  - Young's modulus: E_inclusion = 0.1
  - Poisson's ratio: ν = 0.3
  - Diameter: Random uniform in [0.1, 0.2]
  - Position: Random uniform distribution
  
- **Spatial resolution**: 64×64×64 voxels

## Physical Problem

The data represents solutions to the **linear elasticity** problem:

### Governing Equations

1. **Equilibrium** (static):
   ```
   ∇·σ = 0
   ```

2. **Constitutive law** (Hooke's law for isotropic materials):
   ```
   σ = C : ε
   ```
   
   Where `C` is the stiffness tensor depending on Young's modulus and Poisson's ratio.

3. **Boundary conditions**:
   - Periodic boundary conditions
   - Applied macroscopic strain

### Numerical Solution

Solutions are computed using the **Green-FFT method**:

> Sainsot, P., Nelias, D., & Lubrecht, A. A. (2011). *Efficient solution of the dry contact of rough surfaces: a comparison of Fast Fourier Transform and multigrid methods*. Proceedings of the Institution of Mechanical Engineers, Part J: Journal of Engineering Tribology, 225(6), 441-448.

## Data Loading Example

```python
import torch

# Load input file
input_data = torch.load('data_elasticity_3D_128_0_L0_S0_input.pt')
young_modulus = input_data['input']  # Shape: (128, 1, 64, 64, 64)

# Load output file
output_data = torch.load('data_elasticity_3D_128_0_L0_S0_output.pt')
stress_field = output_data['output']  # Shape: (128, 1, 64, 64, 64)

print(f"Young's modulus range: [{young_modulus.min():.3f}, {young_modulus.max():.3f}]")
print(f"Stress range: [{stress_field.min():.3f}, {stress_field.max():.3f}]")

# Visualize a slice
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(young_modulus[0, 0, :, :, 32].T, cmap='copper')
plt.colorbar(label='Young\'s Modulus')
plt.title('Input: Microstructure')

plt.subplot(1, 2, 2)
plt.imshow(stress_field[0, 0, :, :, 32].T, cmap='viridis')
plt.colorbar(label='Stress σ_xx')
plt.title('Output: Stress Field')
plt.tight_layout()
plt.show()
```

## Dataset Statistics

For a typical dataset:

- **Number of samples**: 128-1024 (depending on number of files)
- **File size**: ~135 MB per input/output pair (uncompressed)
- **Total dataset size**: ~1-8 GB for 512-4096 samples
- **Training/Test split**: Typically 75%/25%

## Notes for Users

1. **Memory requirements**: Loading all data into GPU memory requires ~2-4 GB for 512 samples
2. **Data augmentation**: Geometric transformations (rotations, flips) preserve physical properties
3. **Normalization**: Consider normalizing inputs and outputs for better training stability
4. **Multiple components**: To predict all 6 stress components, train 6 separate models or use multi-output architecture
