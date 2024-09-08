# Dataset Elasticity 3D

### Problem

- Domain = cube 1x1x1
- Resolution = 64x64x64
- Microstructure
  - 100 spherical inclusions
  - Random positions
  - Diameters vary from 0.1 to 0.2 according to a uniform random distribution
- Material
  - Isotropic linear elastic
  - Poisson ratio = 0.3
  - Young modulus
    - Matrix = 1
    - Inclusions = 0.1
- Mathematical formulations
  - [Wikipedia Linear Elasticity](https://en.wikipedia.org/wiki/Linear_elasticity)
    - Equilibrium in static = divergence of stress is null $\sigma_{ji,j} = 0$
    - Constitutive equation = generalized Hook's law $\sigma = C : \varepsilon$

### Data

- Dimension sizes
  - Number of samples = `ns = 128`
  - Number of loadings = `nl = 6`
  - Number of elements along `x` = `nx = 64`
  - Number of elements along `y` = `ny = 64`
  - Number of elements along `z` = `nz = 64`
  - Number of stress components = `nc = 6`
- Component id
  - 0 = xx
  - 1 = yy
  - 2 = zz
  - 3 = yz
  - 4 = zx
  - 5 = xy
- Loadings
  - Unit macroscale strain in each components
  - e.g. = loading 2 = input macro strain : e_zz = 1, all other e_ij = 0
- Variables
  - Inputs
    - `"young_modulus"` (full-field = high dimensional = mesoscale)
      - `shape = (ns, nx, ny, nz)`
    - `"fraction"` (statistical = low dimensional = macroscale)
      - `shape = ns`
  - Outputs
    - `"stress"`
      - `shape = (ns, nl, nx, ny, nz, nc)` (full-field = high dimensional = mesoscale)
    - `"equivalent_young_modulus"` (statistical = low dimensional = macroscale)
      - `shape = (ns, nl)`
    - `"equivalent_poisson_ratio"` (statistical = low dimensional = macroscale)
      - `shape = (ns, nl)`
