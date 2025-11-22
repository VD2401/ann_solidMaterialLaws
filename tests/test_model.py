"""
Unit tests for the UNet3D model.
"""

import pytest
import torch
from ann_solid_materials.model import UNet3D


def test_model_initialization():
    """Test that model can be initialized with default parameters."""
    model = UNet3D()
    assert model is not None
    assert model.depth == 5


def test_model_custom_widths():
    """Test model initialization with custom width parameters."""
    model = UNet3D(width1=2, width2=4, width3=8, width4=16, width5=32)
    assert model.width1 == 2
    assert model.width2 == 4
    assert model.width3 == 8
    assert model.width4 == 16
    assert model.width5 == 32


def test_model_forward_pass():
    """Test that model can perform forward pass."""
    model = UNet3D()
    model.eval()
    
    # Create dummy input (batch_size=2, channels=1, 64x64x64)
    x = torch.randn(2, 1, 64, 64, 64)
    
    with torch.no_grad():
        output = model(x)
    
    # Check output shape
    assert output.shape == (2, 1, 64, 64, 64)


def test_model_output_shape_single_sample():
    """Test output shape for single sample."""
    model = UNet3D()
    model.eval()
    
    x = torch.randn(1, 1, 64, 64, 64)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (1, 1, 64, 64, 64)


def test_model_output_shape_batch():
    """Test output shape for batch of samples."""
    model = UNet3D()
    model.eval()
    
    batch_sizes = [1, 4, 8, 16]
    
    for bs in batch_sizes:
        x = torch.randn(bs, 1, 64, 64, 64)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (bs, 1, 64, 64, 64)


def test_model_parameter_count():
    """Test that parameter counting works."""
    model = UNet3D()
    n_params = model.count_parameters()
    
    assert n_params > 0
    assert isinstance(n_params, int)
    
    # Should be around 350K parameters for default config
    assert 300_000 < n_params < 400_000


def test_model_different_widths_parameter_count():
    """Test that wider models have more parameters."""
    model_narrow = UNet3D(width1=2, width2=4, width3=8, width4=16, width5=32)
    model_default = UNet3D(width1=4, width2=8, width3=16, width4=32, width5=64)
    model_wide = UNet3D(width1=8, width2=16, width3=32, width4=64, width5=128)
    
    n_params_narrow = model_narrow.count_parameters()
    n_params_default = model_default.count_parameters()
    n_params_wide = model_wide.count_parameters()
    
    assert n_params_narrow < n_params_default < n_params_wide


def test_model_gradient_flow():
    """Test that gradients can flow through the model."""
    model = UNet3D()
    model.train()
    
    x = torch.randn(2, 1, 64, 64, 64, requires_grad=True)
    output = model(x)
    
    # Compute loss and backward
    loss = output.sum()
    loss.backward()
    
    # Check that input has gradients
    assert x.grad is not None
    assert not torch.all(x.grad == 0)


def test_model_device_placement():
    """Test that model can be moved to different devices."""
    model = UNet3D()
    
    # CPU test
    model_cpu = model.to('cpu')
    x_cpu = torch.randn(1, 1, 64, 64, 64)
    output_cpu = model_cpu(x_cpu)
    assert output_cpu.device.type == 'cpu'
    
    # MPS test (if available on macOS)
    if torch.backends.mps.is_available():
        model_mps = model.to('mps')
        x_mps = torch.randn(1, 1, 64, 64, 64, device='mps')
        output_mps = model_mps(x_mps)
        assert output_mps.device.type == 'mps'


def test_model_eval_mode():
    """Test that model can be set to evaluation mode."""
    model = UNet3D()
    
    model.eval()
    assert not model.training
    
    model.train()
    assert model.training


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_model_various_batch_sizes(batch_size):
    """Test model with various batch sizes."""
    model = UNet3D()
    model.eval()
    
    x = torch.randn(batch_size, 1, 64, 64, 64)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (batch_size, 1, 64, 64, 64)


def test_model_deterministic_output():
    """Test that model gives same output for same input in eval mode."""
    model = UNet3D()
    model.eval()
    
    torch.manual_seed(42)
    x = torch.randn(1, 1, 64, 64, 64)
    
    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)
    
    assert torch.allclose(output1, output2)
