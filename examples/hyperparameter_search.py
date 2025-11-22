"""
Hyperparameter Search Example

This script demonstrates how to perform systematic hyperparameter search
for the 3D U-Net model, testing different configurations and logging
results to MLflow for comparison.

Usage:
    python hyperparameter_search.py
"""

import mlflow
import torch
from pathlib import Path
from ann_solid_materials import UNet3D, MaterialDataset, Trainer


def search_augmentation_strategies(dataset: MaterialDataset):
    """Test different data augmentation strategies."""
    
    mlflow.set_experiment("augmentation_strategy_search")
    
    augmentation_modes = {
        0: 'no_augmentation',
        1: 'rotation_x_180',
        2: 'rotations_xy_180',
        3: 'rotations_xyz_180',
        4: 'rotations_xyz_flip_yz',
        5: 'flip_yz_only',
    }
    
    print("\n" + "=" * 70)
    print("Testing Data Augmentation Strategies")
    print("=" * 70 + "\n")
    
    for mode, name in augmentation_modes.items():
        print(f"🔄 Testing: {name} (mode={mode})")
        
        with mlflow.start_run(run_name=f"augment_{name}"):
            mlflow.log_params({
                "augmentation_mode": mode,
                "augmentation_name": name,
                "batch_size": 8,
                "learning_rate": 1e-3,
            })
            
            model = UNet3D()
            trainer = Trainer(
                model=model,
                dataset=dataset,
                batch_size=8,
                augmentation_mode=mode,
                max_epochs=50,
                seed=42,
            )
            trainer.train(verbose=False)
            
            print(f"   Final test MAE: {trainer.testing_losses[-1]:.6f}\n")


def search_batch_sizes(dataset: MaterialDataset):
    """Test different batch sizes."""
    
    mlflow.set_experiment("batch_size_search")
    
    batch_sizes = [4, 8, 16, 32, 64]
    
    print("\n" + "=" * 70)
    print("Testing Batch Sizes")
    print("=" * 70 + "\n")
    
    for bs in batch_sizes:
        print(f"📦 Testing batch size: {bs}")
        
        with mlflow.start_run(run_name=f"batch_size_{bs}"):
            mlflow.log_params({
                "batch_size": bs,
                "learning_rate": 1e-3,
                "augmentation_mode": 0,
            })
            
            model = UNet3D()
            trainer = Trainer(
                model=model,
                dataset=dataset,
                batch_size=bs,
                max_epochs=50,
                seed=42,
            )
            trainer.train(verbose=False)
            
            print(f"   Final test MAE: {trainer.testing_losses[-1]:.6f}\n")


def search_learning_rates(dataset: MaterialDataset):
    """Test different learning rates."""
    
    mlflow.set_experiment("learning_rate_search")
    
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    
    print("\n" + "=" * 70)
    print("Testing Learning Rates")
    print("=" * 70 + "\n")
    
    for lr in learning_rates:
        print(f"📈 Testing learning rate: {lr}")
        
        with mlflow.start_run(run_name=f"lr_{lr:.0e}"):
            mlflow.log_params({
                "learning_rate": lr,
                "batch_size": 8,
                "augmentation_mode": 0,
            })
            
            model = UNet3D()
            trainer = Trainer(
                model=model,
                dataset=dataset,
                learning_rate=lr,
                batch_size=8,
                max_epochs=50,
                seed=42,
            )
            trainer.train(verbose=False)
            
            print(f"   Final test MAE: {trainer.testing_losses[-1]:.6f}\n")


def search_model_widths(dataset: MaterialDataset):
    """Test different model width configurations."""
    
    mlflow.set_experiment("model_width_search")
    
    width_configs = [
        ([2, 4, 8, 16, 32], "narrow"),
        ([4, 8, 16, 32, 64], "default"),
        ([8, 16, 32, 64, 128], "wide"),
        ([16, 32, 64, 128, 256], "very_wide"),
    ]
    
    print("\n" + "=" * 70)
    print("Testing Model Widths")
    print("=" * 70 + "\n")
    
    for widths, name in width_configs:
        print(f"🏗️  Testing {name}: {widths}")
        
        with mlflow.start_run(run_name=f"width_{name}"):
            model = UNet3D(
                width1=widths[0],
                width2=widths[1],
                width3=widths[2],
                width4=widths[3],
                width5=widths[4],
            )
            
            mlflow.log_params({
                "model_width": widths,
                "width_name": name,
                "n_parameters": model.count_parameters(),
                "batch_size": 8,
                "learning_rate": 1e-3,
            })
            
            print(f"   Parameters: {model.count_parameters():,}")
            
            trainer = Trainer(
                model=model,
                dataset=dataset,
                batch_size=8,
                max_epochs=50,
                seed=42,
            )
            trainer.train(verbose=False)
            
            print(f"   Final test MAE: {trainer.testing_losses[-1]:.6f}\n")


def main():
    # Configuration
    DATA_PATH = "data/data_files/"
    N_SAMPLES = 256
    
    print("=" * 70)
    print("Hyperparameter Search for 3D U-Net")
    print("=" * 70)
    
    # Check if data exists
    data_dir = Path(DATA_PATH)
    if not data_dir.exists():
        print(f"\n⚠️  Data directory not found: {DATA_PATH}")
        print("Please create the data directory and add your data files.")
        return
    
    # Load dataset (shared across all experiments)
    print(f"\n📁 Loading dataset ({N_SAMPLES} samples)...")
    dataset = MaterialDataset(
        data_path=DATA_PATH,
        n_samples=N_SAMPLES,
        stress_number=0,
        load_number=0,
    )
    
    try:
        dataset.load_data()
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print(f"✅ Dataset loaded: {len(dataset)} samples\n")
    
    # Run hyperparameter searches
    search_augmentation_strategies(dataset)
    search_batch_sizes(dataset)
    search_learning_rates(dataset)
    search_model_widths(dataset)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ Hyperparameter search completed!")
    print("\n📊 View and compare results in MLflow UI:")
    print("   mlflow ui")
    print("   Then open http://localhost:5000")
    print("\n💡 Tips:")
    print("   - Compare runs within each experiment")
    print("   - Look for the best test MAE")
    print("   - Check training curves for convergence")
    print("   - Consider parameter count vs. performance trade-offs")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
