"""
Dataset class for loading 3D elastic material microstructure data.

This module handles loading of Young's modulus fields and corresponding
stress fields generated from FFT-based computational mechanics simulations.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset


def _data_key(file_index: int, load_number: int, stress_number: int) -> str:
    """
    Generate filename for data files.
    
    Parameters
    ----------
    file_index : int
        Index of the data file
    load_number : int
        Loading condition index (0-5 for different strain components)
    stress_number : int
        Stress component index (0-5 for xx, yy, zz, yz, zx, xy)
    
    Returns
    -------
    str
        Filename for the input data
    """
    return f"data_elasticity_3D_128_{file_index}_L{load_number}_S{stress_number}_input.pt"


class MaterialDataset(Dataset):
    """
    PyTorch Dataset for 3D elastic material microstructure data.
    
    This dataset loads Young's modulus fields (inputs) and stress fields (outputs)
    from pre-processed .pt files. Each file contains 128 samples of 64x64x64 voxel
    resolution.
    
    Parameters
    ----------
    data_path : str or Path, optional
        Path to the directory containing data files (default: 'data/data_files/')
    n_samples : int, optional
        Number of samples to load (default: 128)
    stress_number : int, optional
        Stress component to load: 0=xx, 1=yy, 2=zz, 3=yz, 4=zx, 5=xy (default: 0)
    load_number : int, optional
        Loading condition: 0-5 for unit strain in each component (default: 0)
    resolution : int, optional
        Spatial resolution of the 3D data (default: 64)
    device : str or torch.device, optional
        Device to load data to. If None, automatically selects CUDA, MPS, or CPU
    
    Attributes
    ----------
    input : torch.Tensor
        Young's modulus fields of shape (n_samples, 1, resolution, resolution, resolution)
    output : torch.Tensor
        Stress fields of shape (n_samples, 1, resolution, resolution, resolution)
    
    Examples
    --------
    >>> dataset = MaterialDataset(
    ...     data_path='data/data_files/',
    ...     n_samples=256,
    ...     stress_number=0,
    ...     load_number=0
    ... )
    >>> dataset.load_data()
    >>> print(f"Dataset size: {len(dataset)}")
    >>> input_sample, output_sample = dataset[0]
    """
    
    def __init__(
        self,
        data_path: str = 'data/data_files/',
        n_samples: int = 128,
        stress_number: int = 0,
        load_number: int = 0,
        resolution: int = 64,
        device: Optional[torch.device] = None,
    ):
        self.data_path = Path(data_path)
        self.n_samples = n_samples
        self.stress_number = stress_number
        self.load_number = load_number
        self.resolution = resolution
        
        # Automatically select device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        print(f"Device: {self.device}")
        
        # Count available data files (128 samples per file)
        self.number_of_files = self._count_available_files()
        print(f"Found {self.number_of_files} data file(s)")
        
        # Initialize empty tensors (will be filled by load_data())
        self.input = torch.empty(
            (0, 1, resolution, resolution, resolution),
            device=self.device
        )
        self.output = torch.empty(
            (0, 1, resolution, resolution, resolution),
            device=self.device
        )
        self._data_loaded = False

    def _count_available_files(self) -> int:
        """Count how many data files are available."""
        file_index = 0
        while True:
            filename = _data_key(file_index, self.load_number, self.stress_number)
            filepath = self.data_path / filename
            if not filepath.exists():
                break
            file_index += 1
        return file_index

    def load_data(self) -> None:
        """
        Load data from disk into memory.
        
        This method must be called before using the dataset. It reads the
        data files and populates the input and output tensors.
        
        Raises
        ------
        FileNotFoundError
            If required data files are not found
        ValueError
            If data files don't contain expected keys
        """
        if self._data_loaded:
            print("Data already loaded")
            return
        
        # Calculate number of files needed
        num_files_needed = (self.n_samples + 127) // 128  # Ceiling division
        
        if num_files_needed > self.number_of_files:
            raise FileNotFoundError(
                f"Requested {self.n_samples} samples ({num_files_needed} files) "
                f"but only {self.number_of_files} file(s) available"
            )
        
        print(f"Loading {self.n_samples} samples from {num_files_needed} file(s)...")
        
        for i in range(num_files_needed):
            # Generate file paths
            input_filename = _data_key(i, self.load_number, self.stress_number)
            output_filename = input_filename.replace('input', 'output')
            
            input_filepath = self.data_path / input_filename
            output_filepath = self.data_path / output_filename
            
            print(f"  Loading file {i+1}/{num_files_needed}: {input_filename}")
            
            if not input_filepath.exists() or not output_filepath.exists():
                raise FileNotFoundError(
                    f"Data files not found:\n"
                    f"  Input: {input_filepath}\n"
                    f"  Output: {output_filepath}"
                )
            
            # Load the data
            input_data = torch.load(input_filepath, map_location=self.device)
            output_data = torch.load(output_filepath, map_location=self.device)
            
            # Extract tensors (assuming they're stored with 'input' and 'output' keys)
            if 'input' in input_data:
                input_tensor = input_data['input']
            else:
                input_tensor = input_data
            
            if 'output' in output_data:
                output_tensor = output_data['output']
            else:
                output_tensor = output_data
            
            # Concatenate to existing data
            self.input = torch.cat((self.input, input_tensor), dim=0)
            self.output = torch.cat((self.output, output_tensor), dim=0)
        
        # Keep only the requested number of samples
        self.input = self.input[:self.n_samples]
        self.output = self.output[:self.n_samples]
        
        self._data_loaded = True
        print(f"Data loaded successfully. Shape: {self.input.shape}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Parameters
        ----------
        idx : int
            Index of the sample to retrieve
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Input (Young's modulus) and output (stress) tensors
        
        Raises
        ------
        RuntimeError
            If data has not been loaded yet
        IndexError
            If index is out of range
        """
        if not self._data_loaded:
            raise RuntimeError(
                "Data not loaded. Call load_data() before accessing samples."
            )
        
        if idx >= self.n_samples:
            raise IndexError(
                f"Index {idx} out of range for dataset with {self.n_samples} samples"
            )
        
        return self.input[idx], self.output[idx]
    
    def get_statistics(self) -> dict:
        """
        Compute statistics of the dataset.
        
        Returns
        -------
        dict
            Dictionary containing mean and std for inputs and outputs
        """
        if not self._data_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        return {
            'input_mean': self.input.mean().item(),
            'input_std': self.input.std().item(),
            'input_min': self.input.min().item(),
            'input_max': self.input.max().item(),
            'output_mean': self.output.mean().item(),
            'output_std': self.output.std().item(),
            'output_min': self.output.min().item(),
            'output_max': self.output.max().item(),
        }
    
    def __del__(self):
        """Cleanup when dataset object is destroyed."""
        if hasattr(self, 'input') and hasattr(self, 'output'):
            del self.input
            del self.output
