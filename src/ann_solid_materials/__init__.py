"""
ANN for Solid Material Laws

A neural network framework for learning constitutive laws of solid materials
using 3D U-Net architecture.
"""

__version__ = "0.1.0"

from .model import UNet3D
from .dataset import MaterialDataset
from .training import Trainer
from .augmentation import (
    rotate_x_180,
    rotate_y_180,
    rotate_z_180,
    flip_yz,
    flip_xz,
    flip_xy,
)

__all__ = [
    "UNet3D",
    "MaterialDataset",
    "Trainer",
    "rotate_x_180",
    "rotate_y_180",
    "rotate_z_180",
    "flip_yz",
    "flip_xz",
    "flip_xy",
]
