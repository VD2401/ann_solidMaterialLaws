"""
3D U-Net model for learning solid material constitutive laws.

This module implements a 3D U-Net architecture designed to predict stress fields
from material microstructure (Young's modulus distribution).
"""

import torch
import torch.nn as nn


class UNet3D(nn.Module):
    """
    3D U-Net architecture for full-field stress prediction in solid materials.
    
    The network takes a 3D Young's modulus field as input and predicts the
    corresponding stress field. It uses an encoder-decoder structure with
    skip connections to preserve spatial information.
    
    Parameters
    ----------
    width1 : int, optional
        Number of filters in the first encoder level (default: 4)
    width2 : int, optional
        Number of filters in the second encoder level (default: 8)
    width3 : int, optional
        Number of filters in the third encoder level (default: 16)
    width4 : int, optional
        Number of filters in the fourth encoder level (default: 32)
    width5 : int, optional
        Number of filters in the bottleneck level (default: 64)
    
    Attributes
    ----------
    depth : int
        Depth of the U-Net (number of encoding/decoding levels)
    activation : nn.Module
        Activation function used throughout the network
    
    Notes
    -----
    Input shape: (batch_size, 1, 64, 64, 64)
    Output shape: (batch_size, 1, 64, 64, 64)
    
    The network uses:
    - LeakyReLU activation functions
    - Circular padding to handle periodic boundary conditions
    - Max pooling for downsampling
    - Transposed convolutions for upsampling
    """
    
    def __init__(
        self,
        width1: int = 4,
        width2: int = 8,
        width3: int = 16,
        width4: int = 32,
        width5: int = 64,
    ):
        super(UNet3D, self).__init__()

        # Network configuration
        self.activation = nn.LeakyReLU()
        self.depth = 5
        self.width1 = width1
        self.width2 = width2
        self.width3 = width3
        self.width4 = width4
        self.width5 = width5
        
        # Convolution parameters
        kernel_size_conv = 3
        stride_conv = 1
        padding_conv = "same"
        padding_mode = "circular"
        kernel_size_pool = 2
        stride_pool = 2

        # ==================== Encoder ====================
        # Level 1: 64x64x64 -> 32x32x32
        self.e11 = nn.Conv3d(
            1, width1, kernel_size=kernel_size_conv, stride=stride_conv,
            padding=padding_conv, padding_mode=padding_mode
        )
        self.e12 = nn.Conv3d(
            width1, width1, kernel_size=kernel_size_conv, stride=stride_conv,
            padding=padding_conv, padding_mode=padding_mode
        )
        self.pool1 = nn.MaxPool3d(kernel_size=kernel_size_pool, stride=stride_pool)

        # Level 2: 32x32x32 -> 16x16x16
        self.e21 = nn.Conv3d(
            width1, width2, kernel_size=kernel_size_conv, stride=stride_conv,
            padding=padding_conv, padding_mode=padding_mode
        )
        self.e22 = nn.Conv3d(
            width2, width2, kernel_size=kernel_size_conv, stride=stride_conv,
            padding=padding_conv, padding_mode=padding_mode
        )
        self.pool2 = nn.MaxPool3d(kernel_size=kernel_size_pool, stride=stride_pool)

        # Level 3: 16x16x16 -> 8x8x8
        self.e31 = nn.Conv3d(
            width2, width3, kernel_size=kernel_size_conv, stride=stride_conv,
            padding=padding_conv, padding_mode=padding_mode
        )
        self.e32 = nn.Conv3d(
            width3, width3, kernel_size=kernel_size_conv, stride=stride_conv,
            padding=padding_conv, padding_mode=padding_mode
        )
        self.pool3 = nn.MaxPool3d(kernel_size=kernel_size_pool, stride=stride_pool)

        # Level 4: 8x8x8 -> 4x4x4
        self.e41 = nn.Conv3d(
            width3, width4, kernel_size=kernel_size_conv, stride=stride_conv,
            padding=padding_conv, padding_mode=padding_mode
        )
        self.e42 = nn.Conv3d(
            width4, width4, kernel_size=kernel_size_conv, stride=stride_conv,
            padding=padding_conv, padding_mode=padding_mode
        )
        self.pool4 = nn.MaxPool3d(kernel_size=kernel_size_pool, stride=stride_pool)

        # Bottleneck: 4x4x4 -> 4x4x4
        self.e51 = nn.Conv3d(
            width4, width5, kernel_size=kernel_size_conv, stride=stride_conv,
            padding=padding_conv, padding_mode=padding_mode
        )
        self.e52 = nn.Conv3d(
            width5, width5, kernel_size=kernel_size_conv, stride=stride_conv,
            padding=padding_conv, padding_mode=padding_mode
        )
        
        # ==================== Decoder ====================
        # Level 4: 4x4x4 -> 8x8x8
        self.upconv1 = nn.ConvTranspose3d(
            width5, width4, kernel_size=kernel_size_pool, stride=stride_pool
        )
        self.d11 = nn.Conv3d(
            width5, width4, kernel_size=kernel_size_conv, stride=stride_conv,
            padding=padding_conv, padding_mode=padding_mode
        )
        self.d12 = nn.Conv3d(
            width4, width4, kernel_size=kernel_size_conv, stride=stride_conv,
            padding=padding_conv, padding_mode=padding_mode
        )

        # Level 3: 8x8x8 -> 16x16x16
        self.upconv2 = nn.ConvTranspose3d(
            width4, width3, kernel_size=kernel_size_pool, stride=stride_pool
        )
        self.d21 = nn.Conv3d(
            width4, width3, kernel_size=kernel_size_conv, stride=stride_conv,
            padding=padding_conv, padding_mode=padding_mode
        )
        self.d22 = nn.Conv3d(
            width3, width3, kernel_size=kernel_size_conv, stride=stride_conv,
            padding=padding_conv, padding_mode=padding_mode
        )
        
        # Level 2: 16x16x16 -> 32x32x32
        self.upconv3 = nn.ConvTranspose3d(
            width3, width2, kernel_size=kernel_size_pool, stride=stride_pool
        )
        self.d31 = nn.Conv3d(
            width3, width2, kernel_size=kernel_size_conv, stride=stride_conv,
            padding=padding_conv, padding_mode=padding_mode
        )
        self.d32 = nn.Conv3d(
            width2, width2, kernel_size=kernel_size_conv, stride=stride_conv,
            padding=padding_conv, padding_mode=padding_mode
        )

        # Level 1: 32x32x32 -> 64x64x64
        self.upconv4 = nn.ConvTranspose3d(
            width2, width1, kernel_size=kernel_size_pool, stride=stride_pool
        )
        self.d41 = nn.Conv3d(
            width2, width1, kernel_size=kernel_size_conv, stride=stride_conv,
            padding=padding_conv, padding_mode=padding_mode
        )
        self.d42 = nn.Conv3d(
            width1, width1, kernel_size=kernel_size_conv, stride=stride_conv,
            padding=padding_conv, padding_mode=padding_mode
        )

        # Output layer: 64x64x64 -> 64x64x64
        self.outconv = nn.Conv3d(width1, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the 3D U-Net.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, 64, 64, 64)
            representing the Young's modulus field
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1, 64, 64, 64)
            representing the predicted stress field
        """
        # ==================== Encoder ====================
        xe11 = self.activation(self.e11(x))
        xe12 = self.activation(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = self.activation(self.e21(xp1))
        xe22 = self.activation(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = self.activation(self.e31(xp2))
        xe32 = self.activation(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = self.activation(self.e41(xp3))
        xe42 = self.activation(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = self.activation(self.e51(xp4))
        xe52 = self.activation(self.e52(xe51))

        # ==================== Decoder ====================
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)  # Skip connection
        xd11 = self.activation(self.d11(xu11))
        xd12 = self.activation(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)  # Skip connection
        xd21 = self.activation(self.d21(xu22))
        xd22 = self.activation(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)  # Skip connection
        xd31 = self.activation(self.d31(xu33))
        xd32 = self.activation(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)  # Skip connection
        xd41 = self.activation(self.d41(xu44))
        xd42 = self.activation(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out
    
    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters in the model.
        
        Returns
        -------
        int
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
