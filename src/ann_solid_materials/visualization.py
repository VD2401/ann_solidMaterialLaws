"""
Visualization utilities for 3D material fields and training progress.

This module provides functions to visualize 3D stress and Young's modulus fields,
as well as training progress plots.
"""

from typing import Optional, List
import torch
import matplotlib
import matplotlib.pyplot as plt
import mlflow


def plot_training_history(
    training_losses: List[float],
    testing_losses: List[float],
    avg_epoch_time: float,
    output_path: str = "training_history.png",
    log_scale: bool = True,
    save_to_mlflow: bool = True,
) -> plt.Figure:
    """
    Plot training and testing loss curves.
    
    Parameters
    ----------
    training_losses : List[float]
        Training MAE history
    testing_losses : List[float]
        Testing MAE history
    avg_epoch_time : float
        Average time per epoch in seconds
    output_path : str, optional
        Path to save the figure (default: "training_history.png")
    log_scale : bool, optional
        Use logarithmic scale for y-axis (default: True)
    save_to_mlflow : bool, optional
        Whether to log figure to MLflow (default: True)
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(10, 6), dpi=150)
    
    epochs = range(1, len(testing_losses) + 1)
    
    plt.plot(epochs, training_losses, label='Training MAE', 
             linewidth=2, marker='o', markersize=4, alpha=0.8)
    plt.plot(epochs, testing_losses, label='Testing MAE', 
             linewidth=2, marker='s', markersize=4, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    
    if log_scale:
        plt.yscale('log')
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add epoch time annotation
    plt.text(
        0.02, 0.98, 
        f'Avg. epoch time: {avg_epoch_time:.2f}s',
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if save_to_mlflow:
        mlflow.log_figure(fig, output_path)
    
    return fig


def plot_2d_slice(
    data: torch.Tensor,
    slice_idx: int = 32,
    axis: int = 2,
    title: str = "2D Slice",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis',
    figsize: tuple = (8, 7),
) -> plt.Figure:
    """
    Plot a 2D slice through a 3D volume.
    
    Parameters
    ----------
    data : torch.Tensor
        3D tensor of shape (x, y, z)
    slice_idx : int, optional
        Index of the slice to plot (default: 32)
    axis : int, optional
        Axis along which to slice: 0=x, 1=y, 2=z (default: 2)
    title : str, optional
        Plot title
    vmin, vmax : float, optional
        Color scale limits
    cmap : str, optional
        Colormap name (default: 'viridis')
    figsize : tuple, optional
        Figure size (default: (8, 7))
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    data_cpu = data.cpu().numpy() if torch.is_tensor(data) else data
    
    # Extract slice
    if axis == 0:
        slice_data = data_cpu[slice_idx, :, :]
        xlabel, ylabel = 'Y', 'Z'
    elif axis == 1:
        slice_data = data_cpu[:, slice_idx, :]
        xlabel, ylabel = 'X', 'Z'
    else:  # axis == 2
        slice_data = data_cpu[:, :, slice_idx]
        xlabel, ylabel = 'X', 'Y'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(
        slice_data.T, origin='lower', cmap=cmap,
        vmin=vmin, vmax=vmax, aspect='equal'
    )
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    return fig


def plot_comparison(
    input_field: torch.Tensor,
    target_field: torch.Tensor,
    predicted_field: torch.Tensor,
    slice_idx: int = 32,
    axis: int = 2,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis',
    output_path: str = "comparison.png",
    save_to_mlflow: bool = True,
) -> plt.Figure:
    """
    Create a comparison plot showing input, target, prediction, and error.
    
    Parameters
    ----------
    input_field : torch.Tensor
        Input Young's modulus field (3D)
    target_field : torch.Tensor
        Target stress field (3D)
    predicted_field : torch.Tensor
        Predicted stress field (3D)
    slice_idx : int, optional
        Index of the slice to plot (default: 32)
    axis : int, optional
        Axis along which to slice (default: 2)
    vmin, vmax : float, optional
        Color scale limits for stress fields
    cmap : str, optional
        Colormap name (default: 'viridis')
    output_path : str, optional
        Path to save figure (default: "comparison.png")
    save_to_mlflow : bool, optional
        Whether to log to MLflow (default: True)
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Convert to numpy
    input_cpu = input_field.cpu().numpy() if torch.is_tensor(input_field) else input_field
    target_cpu = target_field.cpu().numpy() if torch.is_tensor(target_field) else target_field
    predicted_cpu = predicted_field.cpu().numpy() if torch.is_tensor(predicted_field) else predicted_field
    
    # Extract slices
    if axis == 0:
        input_slice = input_cpu[slice_idx, :, :]
        target_slice = target_cpu[slice_idx, :, :]
        predicted_slice = predicted_cpu[slice_idx, :, :]
        xlabel, ylabel = 'Y', 'Z'
    elif axis == 1:
        input_slice = input_cpu[:, slice_idx, :]
        target_slice = target_cpu[:, slice_idx, :]
        predicted_slice = predicted_cpu[:, slice_idx, :]
        xlabel, ylabel = 'X', 'Z'
    else:
        input_slice = input_cpu[:, :, slice_idx]
        target_slice = target_cpu[:, :, slice_idx]
        predicted_slice = predicted_cpu[:, :, slice_idx]
        xlabel, ylabel = 'X', 'Y'
    
    # Compute error
    error_slice = abs(target_slice - predicted_slice)
    
    # Determine color limits
    if vmin is None or vmax is None:
        stress_vmin = min(target_slice.min(), predicted_slice.min())
        stress_vmax = max(target_slice.max(), predicted_slice.max())
    else:
        stress_vmin, stress_vmax = vmin, vmax
    
    input_vmin, input_vmax = input_slice.min(), input_slice.max()
    error_vmin, error_vmax = 0, error_slice.max()
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Input (Young's modulus)
    im0 = axes[0, 0].imshow(
        input_slice.T, origin='lower', cmap='copper',
        vmin=input_vmin, vmax=input_vmax, aspect='equal'
    )
    axes[0, 0].set_title('Input: Young\'s Modulus', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel(xlabel)
    axes[0, 0].set_ylabel(ylabel)
    fig.colorbar(im0, ax=axes[0, 0])
    
    # Target (ground truth stress)
    im1 = axes[0, 1].imshow(
        target_slice.T, origin='lower', cmap=cmap,
        vmin=stress_vmin, vmax=stress_vmax, aspect='equal'
    )
    axes[0, 1].set_title('Target: Ground Truth Stress', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel(xlabel)
    axes[0, 1].set_ylabel(ylabel)
    fig.colorbar(im1, ax=axes[0, 1])
    
    # Predicted stress
    im2 = axes[1, 0].imshow(
        predicted_slice.T, origin='lower', cmap=cmap,
        vmin=stress_vmin, vmax=stress_vmax, aspect='equal'
    )
    axes[1, 0].set_title('Predicted: Neural Network Output', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel(xlabel)
    axes[1, 0].set_ylabel(ylabel)
    fig.colorbar(im2, ax=axes[1, 0])
    
    # Error
    im3 = axes[1, 1].imshow(
        error_slice.T, origin='lower', cmap='hot',
        vmin=error_vmin, vmax=error_vmax, aspect='equal'
    )
    axes[1, 1].set_title(f'Error (MAE: {error_slice.mean():.6f})', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel(xlabel)
    axes[1, 1].set_ylabel(ylabel)
    fig.colorbar(im3, ax=axes[1, 1])
    
    plt.suptitle(f'Prediction Comparison (Slice {slice_idx}, Axis {axis})',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_to_mlflow:
        mlflow.log_figure(fig, output_path)
    
    return fig


def plot_3d_volume(
    data: torch.Tensor,
    title: str = "3D Volume",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis',
    elev: int = 15,
    azim: int = -105,
    output_path: Optional[str] = None,
    save_to_mlflow: bool = True,
) -> plt.Figure:
    """
    Create a 3D visualization of a volume showing three orthogonal faces.
    
    Parameters
    ----------
    data : torch.Tensor
        3D tensor of shape (x, y, z)
    title : str, optional
        Plot title
    vmin, vmax : float, optional
        Color scale limits
    cmap : str, optional
        Colormap name (default: 'viridis')
    elev : int, optional
        Elevation viewing angle (default: 15)
    azim : int, optional
        Azimuth viewing angle (default: -105)
    output_path : str, optional
        Path to save figure
    save_to_mlflow : bool, optional
        Whether to log to MLflow (default: True)
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    data_cpu = data.cpu() if torch.is_tensor(data) else torch.tensor(data)
    
    nx, ny, nz = data_cpu.shape
    x = torch.arange(nx + 1)
    y = torch.arange(ny + 1)
    z = torch.arange(nz + 1)
    
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    fig = plt.figure(figsize=(10, 8), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize data for colormapping
    if vmin is None:
        vmin = data_cpu.min().item()
    if vmax is None:
        vmax = data_cpu.max().item()
    
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    data_normalized = norm(data_cpu.numpy())
    
    cmap_obj = plt.get_cmap(cmap)
    
    # Plot arguments
    plot_args = {
        'linewidth': 0,
        'antialiased': False,
        'rstride': 1,
        'cstride': 1,
        'shade': False
    }
    
    # Plot three faces
    # Top face (z = max)
    colors = cmap_obj(data_normalized[:, :, -1])
    ax.plot_surface(
        X[:, :, -1].numpy(), Y[:, :, -1].numpy(), Z[:, :, -1].numpy(),
        facecolors=colors, **plot_args
    )
    
    # Front face (y = 0)
    colors = cmap_obj(data_normalized[:, 0, :])
    ax.plot_surface(
        X[:, 0, :].numpy(), Y[:, 0, :].numpy(), Z[:, 0, :].numpy(),
        facecolors=colors, **plot_args
    )
    
    # Left face (x = 0)
    colors = cmap_obj(data_normalized[0, :, :])
    ax.plot_surface(
        X[0, :, :].numpy(), Y[0, :, :].numpy(), Z[0, :, :].numpy(),
        facecolors=colors, **plot_args
    )
    
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_zlabel('Z', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type('ortho')
    ax.set_box_aspect([1, 1, 1])
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5)
    
    plt.tight_layout()
    
    if output_path and save_to_mlflow:
        mlflow.log_figure(fig, output_path)
    
    return fig
