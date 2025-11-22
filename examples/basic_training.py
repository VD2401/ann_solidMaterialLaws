"""
Basic Training Example

This script demonstrates basic usage of the ann_solid_materials package.
It trains a 3D U-Net model on material microstructure data with default parameters.

Usage:
    python basic_training.py
"""

import mlflow
import torch
from pathlib import Path

from ann_solid_materials import UNet3D, MaterialDataset, Trainer


def main():
    # Configuration
    DATA_PATH = "data/data_files/"
    N_SAMPLES = 128
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    MAX_EPOCHS = 100
    EXPERIMENT_NAME = "basic_training"
    
    print("=" * 70)
    print("3D U-Net Training for Solid Material Constitutive Laws")
    print("=" * 70)
    print()
    
    # Check if data exists
    data_dir = Path(DATA_PATH)
    if not data_dir.exists():
        print(f"⚠️  Data directory not found: {DATA_PATH}")
        print("Please create the data directory and add your data files.")
        print("See docs/data_format.md for data format specifications.")
        return
    
    # Setup MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"📊 MLflow experiment: {EXPERIMENT_NAME}")
    print()
    
    # Load dataset
    print("📁 Loading dataset...")
    dataset = MaterialDataset(
        data_path=DATA_PATH,
        n_samples=N_SAMPLES,
        stress_number=0,  # σ_xx component
        load_number=0,    # Unit strain in x-direction
        resolution=64,
    )
    
    try:
        dataset.load_data()
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        print("Please check that data files exist in the correct format.")
        return
    
    # Print dataset statistics
    stats = dataset.get_statistics()
    print(f"   Samples loaded: {len(dataset)}")
    print(f"   Input range: [{stats['input_min']:.3f}, {stats['input_max']:.3f}]")
    print(f"   Output range: [{stats['output_min']:.3f}, {stats['output_max']:.3f}]")
    print()
    
    # Create model
    print("🏗️  Creating 3D U-Net model...")
    model = UNet3D(
        width1=4,
        width2=8,
        width3=16,
        width4=32,
        width5=64,
    )
    print(f"   Parameters: {model.count_parameters():,}")
    print()
    
    # Start MLflow run
    with mlflow.start_run(run_name="baseline_model"):
        # Log hyperparameters
        params = {
            "n_samples": N_SAMPLES,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "augmentation_mode": 0,
            "stress_component": 0,
            "load_condition": 0,
            "model_width": [4, 8, 16, 32, 64],
            "n_parameters": model.count_parameters(),
        }
        mlflow.log_params(params)
        print("📝 Hyperparameters logged to MLflow")
        print()
        
        # Create trainer
        print("🎯 Initializing trainer...")
        trainer = Trainer(
            model=model,
            dataset=dataset,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            split_ratio=0.75,
            max_epochs=MAX_EPOCHS,
            stop_criteria=0.02,
            augmentation_mode=0,
            seed=42,
        )
        print()
        
        # Train
        print("🚀 Starting training...")
        print("-" * 70)
        trainer.train(verbose=True)
        print("-" * 70)
        print()
        
        # Log final model
        print("💾 Saving final model to MLflow...")
        mlflow.pytorch.log_model(model, "final_model")
        
        # Get and visualize worst prediction
        print("🔍 Analyzing worst prediction...")
        idx, mae, input_field, target, prediction = trainer.get_worst_prediction(
            use_test_set=True
        )
        print(f"   Worst test MAE: {mae:.6f} at sample {idx}")
        
        # Create visualization
        try:
            from ann_solid_materials.visualization import (
                plot_comparison,
                plot_training_history,
            )
            
            # Training history plot
            avg_time = sum(trainer.training_times) / len(trainer.training_times)
            plot_training_history(
                trainer.training_losses,
                trainer.testing_losses,
                avg_time,
                output_path="training_history.png",
            )
            
            # Comparison plot for worst prediction
            plot_comparison(
                input_field[0],
                target[0],
                prediction[0],
                slice_idx=32,
                axis=2,
                output_path="worst_prediction.png",
            )
            
            print("📊 Visualizations saved to MLflow")
        except ImportError:
            print("⚠️  Matplotlib not available, skipping visualizations")
        
        print()
        print("=" * 70)
        print("✅ Training completed successfully!")
        print()
        print("📊 View results in MLflow UI:")
        print("   mlflow ui")
        print("   Then open http://localhost:5000")
        print("=" * 70)


if __name__ == "__main__":
    main()
