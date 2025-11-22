"""
Training module for 3D U-Net on solid material constitutive laws.

This module provides a Trainer class that handles model training, validation,
data augmentation, and MLflow experiment tracking.
"""

import time
from typing import Optional, List, Tuple, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow

from .model import UNet3D
from .dataset import MaterialDataset
from .augmentation import AUGMENTATION_FUNCTIONS


class Trainer:
    """
    Trainer class for 3D U-Net model.
    
    Handles training loop, validation, data splitting, augmentation, and MLflow logging.
    
    Parameters
    ----------
    model : UNet3D
        The 3D U-Net model to train
    dataset : MaterialDataset
        Dataset containing material microstructure data
    learning_rate : float, optional
        Learning rate for Adam optimizer (default: 1e-3)
    batch_size : int, optional
        Batch size for training (default: 8)
    split_ratio : float, optional
        Fraction of data to use for training (default: 0.75)
    max_epochs : int, optional
        Maximum number of training epochs (default: computed based on dataset size)
    stop_criteria : float, optional
        Stop training if test MAE falls below this value (default: 0.02)
    augmentation_mode : int, optional
        Augmentation strategy:
        - 0: No augmentation
        - 1: Add 180° rotation around x-axis
        - 2: Add rotations around x and y axes
        - 3: Add rotations around x, y, and z axes
        - 4: Add rotations + flip along y,z axes
        - 5: Only flip along y,z axes (default: 0)
    seed : int, optional
        Random seed for reproducibility (default: 0)
    device : torch.device, optional
        Device to train on. If None, uses dataset's device
    
    Attributes
    ----------
    training_losses : List[float]
        Training MAE history
    testing_losses : List[float]
        Testing MAE history
    training_times : List[float]
        Time per training epoch
    testing_times : List[float]
        Time per validation epoch
    epochs : int
        Current epoch number
    """
    
    def __init__(
        self,
        model: UNet3D,
        dataset: MaterialDataset,
        learning_rate: float = 1e-3,
        batch_size: int = 8,
        split_ratio: float = 0.75,
        max_epochs: Optional[int] = None,
        stop_criteria: float = 0.02,
        augmentation_mode: int = 0,
        seed: int = 0,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.device = device if device is not None else dataset.device
        self.model.to(self.device)
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.stop_criteria = stop_criteria
        self.augmentation_mode = augmentation_mode
        
        # Adjust max_epochs based on dataset size to keep compute time constant
        if max_epochs is None:
            self.max_epochs = int(500 * 128 / self.dataset.n_samples)
        else:
            self.max_epochs = max_epochs
        
        # Setup optimizer and loss
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training history
        self.training_losses: List[float] = []
        self.testing_losses: List[float] = []
        self.training_times: List[float] = []
        self.testing_times: List[float] = []
        self.epochs = 0
        
        # Set random seed
        self.set_seed(seed)
        
        # Create data loaders
        self.train_loader, self.test_loader = self._create_data_loaders()
        
        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Max epochs: {self.max_epochs}")
        print(f"  Training samples: {len(self.train_loader.dataset)}")
        print(f"  Testing samples: {len(self.test_loader.dataset)}")
    
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to {seed}")
    
    def _apply_augmentation(
        self, 
        inputs: torch.Tensor, 
        outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply data augmentation according to augmentation_mode.
        
        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor (Young's modulus fields)
        outputs : torch.Tensor
            Output tensor (stress fields)
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Augmented inputs and outputs
        """
        if self.augmentation_mode == 0:
            return inputs, outputs
        
        # Special case: only flip along y,z
        if self.augmentation_mode == 5:
            aug_func = AUGMENTATION_FUNCTIONS[4]  # flip_yz
            input_aug, output_aug = aug_func(inputs, outputs)
            return (
                torch.cat([inputs, input_aug], dim=0),
                torch.cat([outputs, output_aug], dim=0)
            )
        
        # Progressive augmentation: add transformations cumulatively
        new_inputs, new_outputs = inputs, outputs
        
        for aug_idx in range(1, min(self.augmentation_mode + 1, 5)):
            aug_func = AUGMENTATION_FUNCTIONS[aug_idx]
            input_aug, output_aug = aug_func(inputs, outputs)
            new_inputs = torch.cat([new_inputs, input_aug], dim=0)
            new_outputs = torch.cat([new_outputs, output_aug], dim=0)
        
        return new_inputs, new_outputs
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and testing data loaders with augmentation."""
        # Split data
        split_idx = int(len(self.dataset) * self.split_ratio)
        
        train_inputs = self.dataset.input[:split_idx]
        train_outputs = self.dataset.output[:split_idx]
        test_inputs = self.dataset.input[split_idx:]
        test_outputs = self.dataset.output[split_idx:]
        
        # Apply augmentation to training data only
        train_inputs, train_outputs = self._apply_augmentation(
            train_inputs, train_outputs
        )
        
        # Create datasets
        train_dataset = TensorDataset(train_inputs, train_outputs)
        test_dataset = TensorDataset(test_inputs, test_outputs)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        return train_loader, test_loader
    
    def train_epoch(self) -> float:
        """
        Execute one training epoch.
        
        Returns
        -------
        float
            Mean Absolute Error over the training set
        """
        self.model.train()
        batch_losses = []
        
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Record batch MAE
            mae = torch.abs(output - target).mean().item()
            batch_losses.append(mae)
            
            # Log to MLflow
            mlflow.log_metric("batch_loss", loss.item(), step=self.epochs)
        
        epoch_mae = sum(batch_losses) / len(batch_losses)
        mlflow.log_metric("training_mae", epoch_mae, step=self.epochs)
        
        return epoch_mae
    
    def validate_epoch(self) -> Tuple[float, float]:
        """
        Execute one validation epoch.
        
        Returns
        -------
        Tuple[float, float]
            Mean Absolute Error and Root Mean Squared Error over test set
        """
        self.model.eval()
        batch_mae = []
        batch_rmse = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                mae = torch.abs(output - target).mean().item()
                rmse = torch.sqrt(torch.mean((output - target) ** 2)).item()
                
                batch_mae.append(mae)
                batch_rmse.append(rmse)
        
        epoch_mae = sum(batch_mae) / len(batch_mae)
        epoch_rmse = sum(batch_rmse) / len(batch_rmse)
        
        mlflow.log_metric("testing_mae", epoch_mae, step=self.epochs)
        mlflow.log_metric("testing_rmse", epoch_rmse, step=self.epochs)
        
        return epoch_mae, epoch_rmse
    
    def train(self, verbose: bool = True) -> None:
        """
        Execute full training loop.
        
        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress (default: True)
        """
        print(f"\nStarting training for up to {self.max_epochs} epochs...")
        print(f"Early stopping at MAE < {self.stop_criteria}")
        
        for epoch in range(self.max_epochs):
            self.epochs = epoch + 1
            
            # Training
            start_time = time.time()
            train_mae = self.train_epoch()
            train_time = time.time() - start_time
            self.training_losses.append(train_mae)
            self.training_times.append(train_time)
            
            # Validation
            start_time = time.time()
            test_mae, test_rmse = self.validate_epoch()
            test_time = time.time() - start_time
            self.testing_losses.append(test_mae)
            self.testing_times.append(test_time)
            
            if verbose:
                print(
                    f"Epoch {self.epochs:3d}/{self.max_epochs} | "
                    f"Train MAE: {train_mae:.6f} | "
                    f"Test MAE: {test_mae:.6f} | "
                    f"Test RMSE: {test_rmse:.6f} | "
                    f"Time: {train_time:.2f}s"
                )
            
            # Save model at checkpoints
            if self.epochs in [
                self.max_epochs // 4,
                self.max_epochs // 2,
                3 * self.max_epochs // 4,
                self.max_epochs
            ]:
                mlflow.pytorch.log_model(self.model, f"model_epoch_{self.epochs}")
                if verbose:
                    print(f"  → Model saved at epoch {self.epochs}")
            
            # Early stopping criteria
            if test_mae < self.stop_criteria:
                print(f"\n✓ Early stopping: Test MAE ({test_mae:.6f}) < {self.stop_criteria}")
                mlflow.pytorch.log_model(self.model, f"model_final_epoch_{self.epochs}")
                break
            
            # Stop if training diverges
            if train_mae > 10:
                print(f"\n✗ Training diverged: MAE = {train_mae:.6f}")
                break
        
        total_time = sum(self.training_times) + sum(self.testing_times)
        print(f"\nTraining completed!")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Final train MAE: {self.training_losses[-1]:.6f}")
        print(f"  Final test MAE: {self.testing_losses[-1]:.6f}")
        
        mlflow.log_metric("final_train_mae", self.training_losses[-1])
        mlflow.log_metric("final_test_mae", self.testing_losses[-1])
        mlflow.log_metric("total_training_time", total_time)
    
    def get_worst_prediction(
        self, use_test_set: bool = True
    ) -> Tuple[int, float, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find the sample with the highest prediction error.
        
        Parameters
        ----------
        use_test_set : bool, optional
            If True, search in test set; otherwise search in training set
        
        Returns
        -------
        Tuple[int, float, torch.Tensor, torch.Tensor, torch.Tensor]
            Index, MAE, input, target, and prediction for worst sample
        """
        loader = self.test_loader if use_test_set else self.train_loader
        
        self.model.eval()
        batch_losses = []
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                mae = torch.abs(output - target).mean(dim=(1, 2, 3, 4))
                batch_losses.extend(mae.cpu().tolist())
        
        # Find worst sample
        worst_idx = batch_losses.index(max(batch_losses))
        worst_mae = batch_losses[worst_idx]
        
        # Get the actual data
        input_sample, target_sample = loader.dataset[worst_idx]
        input_sample = input_sample.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output_sample = self.model(input_sample)
        
        return (
            worst_idx,
            worst_mae,
            input_sample.squeeze(0).cpu(),
            target_sample.cpu(),
            output_sample.squeeze(0).cpu()
        )
