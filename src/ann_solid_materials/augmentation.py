"""
Data augmentation functions for 3D material microstructure data.

These functions provide geometric transformations that preserve the physical
properties of the material while increasing the effective dataset size.
"""

import torch
from typing import Tuple


def rotate_x_180(
    inputs: torch.Tensor, outputs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate inputs and outputs 180° around the x-axis.
    
    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor of shape (batch, channels, x, y, z)
    outputs : torch.Tensor
        Output tensor of shape (batch, channels, x, y, z)
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Rotated inputs and outputs
    """
    return torch.rot90(inputs, k=2, dims=[3, 4]), torch.rot90(outputs, k=2, dims=[3, 4])


def rotate_y_180(
    inputs: torch.Tensor, outputs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate inputs and outputs 180° around the y-axis.
    
    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor of shape (batch, channels, x, y, z)
    outputs : torch.Tensor
        Output tensor of shape (batch, channels, x, y, z)
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Rotated inputs and outputs
    """
    return torch.rot90(inputs, k=2, dims=[2, 4]), torch.rot90(outputs, k=2, dims=[2, 4])


def rotate_z_180(
    inputs: torch.Tensor, outputs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate inputs and outputs 180° around the z-axis.
    
    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor of shape (batch, channels, x, y, z)
    outputs : torch.Tensor
        Output tensor of shape (batch, channels, x, y, z)
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Rotated inputs and outputs
    """
    return torch.rot90(inputs, k=2, dims=[2, 3]), torch.rot90(outputs, k=2, dims=[2, 3])


def flip_yz(
    inputs: torch.Tensor, outputs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flip inputs and outputs along the y and z axes.
    
    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor of shape (batch, channels, x, y, z)
    outputs : torch.Tensor
        Output tensor of shape (batch, channels, x, y, z)
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Flipped inputs and outputs
    """
    return torch.flip(inputs, dims=[3, 4]), torch.flip(outputs, dims=[3, 4])


def flip_xz(
    inputs: torch.Tensor, outputs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flip inputs and outputs along the x and z axes.
    
    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor of shape (batch, channels, x, y, z)
    outputs : torch.Tensor
        Output tensor of shape (batch, channels, x, y, z)
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Flipped inputs and outputs
    """
    return torch.flip(inputs, dims=[2, 4]), torch.flip(outputs, dims=[2, 4])


def flip_xy(
    inputs: torch.Tensor, outputs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flip inputs and outputs along the x and y axes.
    
    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor of shape (batch, channels, x, y, z)
    outputs : torch.Tensor
        Output tensor of shape (batch, channels, x, y, z)
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Flipped inputs and outputs
    """
    return torch.flip(inputs, dims=[2, 3]), torch.flip(outputs, dims=[2, 3])


# Dictionary mapping augmentation mode to function
AUGMENTATION_FUNCTIONS = {
    0: lambda x, y: (x, y),  # No augmentation
    1: rotate_x_180,
    2: rotate_y_180,
    3: rotate_z_180,
    4: flip_yz,
    5: flip_xz,
    6: flip_xy,
}

AUGMENTATION_NAMES = {
    0: "no augmentation",
    1: "rotation around x-axis (180°)",
    2: "rotation around y-axis (180°)",
    3: "rotation around z-axis (180°)",
    4: "flip along y,z axes",
    5: "flip along x,z axes",
    6: "flip along x,y axes",
}
