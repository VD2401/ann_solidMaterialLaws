# Neural Networks for Solid Material Constitutive Laws

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning framework for predicting stress fields in solid materials from microstructure using 3D U-Net architecture. This project demonstrates the application of neural networks to learn constitutive laws in computational solid mechanics.

## 🎯 Overview

This repository implements a **3D U-Net** neural network to predict full-field stress distributions in heterogeneous elastic materials from their Young's modulus distribution. The approach is relevant for:

- **Computational materials science**: Accelerating multi-scale simulations
- **Digital materials design**: Rapid property prediction
- **Homogenization**: Learning effective properties from microstructure

### Key Features

- ✅ **3D U-Net architecture** with skip connections for spatial information preservation
- ✅ **Data augmentation** strategies (rotations, flips) for improved generalization
- ✅ **MLflow integration** for experiment tracking and model versioning
- ✅ **Flexible training** with configurable hyperparameters
- ✅ **Visualization tools** for 3D fields and training progress
- ✅ **Modern Python packaging** with uv for fast dependency management

## 📊 Problem Description

### Physical Problem

We solve the **linear elasticity** problem for heterogeneous materials:

- **Domain**: Unit cube (1×1×1) discretized at 64×64×64 resolution
- **Microstructure**: Random spherical inclusions (100 spheres, diameters 0.1-0.2)
- **Material properties**:
  - Matrix: E = 1.0, ν = 0.3
  - Inclusions: E = 0.1, ν = 0.3
- **Governing equations**:
  - Equilibrium: ∇·σ = 0
  - Constitutive law: σ = C:ε (Hooke's law)

### Data Generation

Training data is generated using the **Green-FFT** method, a Fast Fourier Transform-based computational mechanics solver for heterogeneous materials.

> **Reference**: Sainsot, P., Nelias, D., & Lubrecht, A. A. (2011). *Efficient solution of the dry contact of rough surfaces: a comparison of Fast Fourier Transform and multigrid methods*. Proceedings of the Institution of Mechanical Engineers, Part J: Journal of Engineering Tribology, 225(6), 441-448.

The dataset consists of:
- **Input**: Young's modulus field (64×64×64)
- **Output**: Stress field for specific component and loading (64×64×64)
- **Format**: PyTorch `.pt` files, 128 samples per file

## 🚀 Quick Start with uv

This project uses [**uv**](https://github.com/astral-sh/uv), an extremely fast Python package installer and resolver written in Rust.

### 1. Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/p-devianne/ann_solidMaterialLaws.git
cd ann_solidMaterialLaws

# Create virtual environment and install dependencies (takes ~10 seconds!)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in editable mode with dependencies
uv pip install -e .

# Optional: Install development tools
uv pip install -e ".[dev]"

# Optional: Install Jupyter for notebooks
uv pip install -e ".[notebooks]"
```

### 3. Prepare Your Data

Place your data files in `data/data_files/` following this naming convention:

```
data/data_files/
├── data_elasticity_3D_128_0_L0_S0_input.pt
├── data_elasticity_3D_128_0_L0_S0_output.pt
├── data_elasticity_3D_128_1_L0_S0_input.pt
├── data_elasticity_3D_128_1_L0_S0_output.pt
└── ...
```

Where:
- `0, 1, ...`: File index (128 samples per file)
- `L0`: Loading condition (0-5 for strain components xx, yy, zz, yz, zx, xy)
- `S0`: Stress component (0-5 for σ_xx, σ_yy, σ_zz, σ_yz, σ_zx, σ_xy)

See `docs/data_format.md` for detailed data specifications.

### 4. Run Basic Training

```bash
python examples/basic_training.py
```

This will:
1. Load 128 samples
2. Train a 3D U-Net with default parameters
3. Log results to MLflow
4. Save the trained model

## 📚 Usage Examples

### Basic Training Script

```python
from ann_solid_materials import UNet3D, MaterialDataset, Trainer
import mlflow

# Setup MLflow experiment
mlflow.set_experiment("my_first_experiment")

# Load data
dataset = MaterialDataset(
    data_path='data/data_files/',
    n_samples=256,
    stress_number=0,  # σ_xx component
    load_number=0,     # Unit strain in x-direction
)
dataset.load_data()

# Create model
model = UNet3D(width1=4, width2=8, width3=16, width4=32, width5=64)

# Start MLflow run
with mlflow.start_run(run_name="baseline_model"):
    # Log hyperparameters
    mlflow.log_params({
        "n_samples": 256,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "augmentation_mode": 0,
    })
    
    # Train
    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=8,
        learning_rate=1e-3,
        augmentation_mode=0,
        max_epochs=100,
    )
    trainer.train()
    
    # Save final model
    mlflow.pytorch.log_model(model, "final_model")
```

### Hyperparameter Search

```python
# Try different augmentation strategies
augmentation_modes = {
    0: 'no_augmentation',
    1: 'rotation_x',
    3: 'rotations_xyz',
    4: 'rotations_xyz_flip_yz',
}

for mode, name in augmentation_modes.items():
    with mlflow.start_run(run_name=f"augment_{name}"):
        mlflow.log_param("augmentation_mode", mode)
        
        trainer = Trainer(
            model=UNet3D(),
            dataset=dataset,
            augmentation_mode=mode,
            batch_size=64,
        )
        trainer.train()
```

### Visualization

```python
from ann_solid_materials.visualization import plot_comparison

# Get worst prediction from test set
idx, mae, input_field, target, prediction = trainer.get_worst_prediction()

# Create comparison plot
plot_comparison(
    input_field[0],
    target[0],
    prediction[0],
    slice_idx=32,
    axis=2,
    output_path="worst_prediction.png"
)
```

## 📁 Project Structure

```
ann_solidMaterialLaws/
├── src/ann_solid_materials/    # Main package
│   ├── __init__.py
│   ├── model.py               # 3D U-Net architecture
│   ├── dataset.py             # Data loading
│   ├── training.py            # Trainer class
│   ├── augmentation.py        # Data augmentation
│   └── visualization.py       # Plotting utilities
├── examples/                  # Usage examples
│   ├── basic_training.py
│   ├── hyperparameter_search.py
│   └── notebooks/            # Jupyter notebooks
│       ├── 01_data_exploration.ipynb
│       ├── 02_model_training.ipynb
│       └── 03_results_visualization.ipynb
├── tests/                    # Unit tests
│   ├── test_model.py
│   ├── test_dataset.py
│   ├── test_training.py
│   └── test_augmentation.py
├── docs/                     # Documentation
│   ├── data_format.md
│   └── architecture.md
├── scripts/                  # Utility scripts
├── data/                     # Data directory (not in repo)
│   └── data_files/
├── mlruns/                   # MLflow tracking (generated)
├── pyproject.toml           # Project configuration
├── README.md
└── .gitignore
```

## 🔬 Model Architecture

The 3D U-Net consists of:

- **Encoder**: 5 levels with max pooling (64→32→16→8→4)
- **Bottleneck**: 4×4×4 feature maps
- **Decoder**: 5 levels with transposed convolutions (4→8→16→32→64)
- **Skip connections**: Concatenate encoder features to decoder
- **Activation**: LeakyReLU
- **Padding**: Circular (for periodic boundary conditions)

Default configuration:
- Filters: [4, 8, 16, 32, 64] across levels
- Parameters: ~140K (depending on width configuration)

See `docs/architecture.md` for detailed architecture description.

## 🧪 Experiments with MLflow

### View Experiment Results

```bash
# Start MLflow UI
mlflow ui

# Open browser to http://localhost:5000
```

### Compare Runs

The MLflow UI allows you to:
- Compare metrics across runs
- Visualize training curves
- Download trained models
- Track hyperparameters

### Organize Experiments

```python
# Create organized experiments
mlflow.set_experiment("batch_size_search")
mlflow.set_experiment("learning_rate_search")
mlflow.set_experiment("model_width_search")
mlflow.set_experiment("augmentation_strategy")
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ann_solid_materials --cov-report=html

# Run specific test file
pytest tests/test_model.py -v
```

## 📖 Documentation

- **Data Format**: See `docs/data_format.md` for dataset specifications
- **Architecture**: See `docs/architecture.md` for model details
- **Examples**: Check `examples/` directory for usage patterns
- **Notebooks**: Explore `examples/notebooks/` for interactive tutorials

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@software{ann_solid_materials_2025,
  author = {Devianne, Paul},
  title = {Neural Networks for Solid Material Constitutive Laws},
  year = {2025},
  url = {https://github.com/p-devianne/ann_solidMaterialLaws}
}
```

For the data generation method:

```bibtex
@article{sainsot2011efficient,
  title={Efficient solution of the dry contact of rough surfaces: a comparison of Fast Fourier Transform and multigrid methods},
  author={Sainsot, Philippe and Nelias, Daniel and Lubrecht, Antonius Adrianus},
  journal={Proceedings of the Institution of Mechanical Engineers, Part J: Journal of Engineering Tribology},
  volume={225},
  number={6},
  pages={441--448},
  year={2011},
  publisher={SAGE Publications}
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Data generated using Green-FFT method (Sainsot et al., 2011)
- U-Net architecture inspired by Ronneberger et al. (2015)
- Built with PyTorch, MLflow, and modern Python tooling

## 📧 Contact

**Paul Devianne**  
GitHub: [@p-devianne](https://github.com/p-devianne)

---

**Happy modeling! 🚀**
