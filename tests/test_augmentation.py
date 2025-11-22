"""
Unit tests for data augmentation functions.
"""

import pytest
import torch
from ann_solid_materials.augmentation import (
    rotate_x_180,
    rotate_y_180,
    rotate_z_180,
    flip_yz,
    flip_xz,
    flip_xy,
    AUGMENTATION_FUNCTIONS,
    AUGMENTATION_NAMES,
)


def test_augmentation_functions_dict():
    """Test that augmentation function dictionaries are properly defined."""
    assert len(AUGMENTATION_FUNCTIONS) == 7  # 0-6
    assert len(AUGMENTATION_NAMES) == 7


def test_rotate_x_180_shape():
    """Test that rotation preserves tensor shape."""
    inputs = torch.randn(4, 1, 64, 64, 64)
    outputs = torch.randn(4, 1, 64, 64, 64)
    
    aug_inputs, aug_outputs = rotate_x_180(inputs, outputs)
    
    assert aug_inputs.shape == inputs.shape
    assert aug_outputs.shape == outputs.shape


def test_rotate_y_180_shape():
    """Test that rotation preserves tensor shape."""
    inputs = torch.randn(4, 1, 64, 64, 64)
    outputs = torch.randn(4, 1, 64, 64, 64)
    
    aug_inputs, aug_outputs = rotate_y_180(inputs, outputs)
    
    assert aug_inputs.shape == inputs.shape
    assert aug_outputs.shape == outputs.shape


def test_rotate_z_180_shape():
    """Test that rotation preserves tensor shape."""
    inputs = torch.randn(4, 1, 64, 64, 64)
    outputs = torch.randn(4, 1, 64, 64, 64)
    
    aug_inputs, aug_outputs = rotate_z_180(inputs, outputs)
    
    assert aug_inputs.shape == inputs.shape
    assert aug_outputs.shape == outputs.shape


def test_flip_yz_shape():
    """Test that flip preserves tensor shape."""
    inputs = torch.randn(4, 1, 64, 64, 64)
    outputs = torch.randn(4, 1, 64, 64, 64)
    
    aug_inputs, aug_outputs = flip_yz(inputs, outputs)
    
    assert aug_inputs.shape == inputs.shape
    assert aug_outputs.shape == outputs.shape


def test_flip_xz_shape():
    """Test that flip preserves tensor shape."""
    inputs = torch.randn(4, 1, 64, 64, 64)
    outputs = torch.randn(4, 1, 64, 64, 64)
    
    aug_inputs, aug_outputs = flip_xz(inputs, outputs)
    
    assert aug_inputs.shape == inputs.shape
    assert aug_outputs.shape == outputs.shape


def test_flip_xy_shape():
    """Test that flip preserves tensor shape."""
    inputs = torch.randn(4, 1, 64, 64, 64)
    outputs = torch.randn(4, 1, 64, 64, 64)
    
    aug_inputs, aug_outputs = flip_xy(inputs, outputs)
    
    assert aug_inputs.shape == inputs.shape
    assert aug_outputs.shape == outputs.shape


def test_rotate_x_180_twice_is_identity():
    """Test that rotating 180° twice returns to original."""
    torch.manual_seed(42)
    inputs = torch.randn(2, 1, 64, 64, 64)
    outputs = torch.randn(2, 1, 64, 64, 64)
    
    aug1_inputs, aug1_outputs = rotate_x_180(inputs, outputs)
    aug2_inputs, aug2_outputs = rotate_x_180(aug1_inputs, aug1_outputs)
    
    assert torch.allclose(aug2_inputs, inputs)
    assert torch.allclose(aug2_outputs, outputs)


def test_rotate_y_180_twice_is_identity():
    """Test that rotating 180° twice returns to original."""
    torch.manual_seed(42)
    inputs = torch.randn(2, 1, 64, 64, 64)
    outputs = torch.randn(2, 1, 64, 64, 64)
    
    aug1_inputs, aug1_outputs = rotate_y_180(inputs, outputs)
    aug2_inputs, aug2_outputs = rotate_y_180(aug1_inputs, aug1_outputs)
    
    assert torch.allclose(aug2_inputs, inputs)
    assert torch.allclose(aug2_outputs, outputs)


def test_rotate_z_180_twice_is_identity():
    """Test that rotating 180° twice returns to original."""
    torch.manual_seed(42)
    inputs = torch.randn(2, 1, 64, 64, 64)
    outputs = torch.randn(2, 1, 64, 64, 64)
    
    aug1_inputs, aug1_outputs = rotate_z_180(inputs, outputs)
    aug2_inputs, aug2_outputs = rotate_z_180(aug1_inputs, aug1_outputs)
    
    assert torch.allclose(aug2_inputs, inputs)
    assert torch.allclose(aug2_outputs, outputs)


def test_flip_twice_is_identity():
    """Test that flipping twice returns to original."""
    torch.manual_seed(42)
    inputs = torch.randn(2, 1, 64, 64, 64)
    outputs = torch.randn(2, 1, 64, 64, 64)
    
    # Test all flip operations
    for flip_func in [flip_yz, flip_xz, flip_xy]:
        aug1_inputs, aug1_outputs = flip_func(inputs, outputs)
        aug2_inputs, aug2_outputs = flip_func(aug1_inputs, aug1_outputs)
        
        assert torch.allclose(aug2_inputs, inputs)
        assert torch.allclose(aug2_outputs, outputs)


def test_augmentation_changes_data():
    """Test that augmentation actually changes the data."""
    torch.manual_seed(42)
    inputs = torch.randn(2, 1, 64, 64, 64)
    outputs = torch.randn(2, 1, 64, 64, 64)
    
    # Create a non-uniform input to ensure change is detectable
    inputs[0, 0, 0, 0, 0] = 999.0
    inputs[0, 0, -1, -1, -1] = -999.0
    
    aug_inputs, aug_outputs = rotate_x_180(inputs, outputs)
    
    # Data should be different
    assert not torch.allclose(aug_inputs, inputs)
    assert not torch.allclose(aug_outputs, outputs)


def test_augmentation_preserves_statistics():
    """Test that augmentation preserves mean and std (approximately)."""
    torch.manual_seed(42)
    inputs = torch.randn(10, 1, 64, 64, 64)
    outputs = torch.randn(10, 1, 64, 64, 64)
    
    orig_input_mean = inputs.mean()
    orig_input_std = inputs.std()
    orig_output_mean = outputs.mean()
    orig_output_std = outputs.std()
    
    aug_inputs, aug_outputs = rotate_x_180(inputs, outputs)
    
    # Mean and std should be approximately preserved
    assert torch.abs(aug_inputs.mean() - orig_input_mean) < 0.1
    assert torch.abs(aug_inputs.std() - orig_input_std) < 0.1
    assert torch.abs(aug_outputs.mean() - orig_output_mean) < 0.1
    assert torch.abs(aug_outputs.std() - orig_output_std) < 0.1


@pytest.mark.parametrize("aug_mode", [0, 1, 2, 3, 4, 5, 6])
def test_augmentation_functions_callable(aug_mode):
    """Test that all augmentation functions are callable."""
    inputs = torch.randn(2, 1, 64, 64, 64)
    outputs = torch.randn(2, 1, 64, 64, 64)
    
    aug_func = AUGMENTATION_FUNCTIONS[aug_mode]
    aug_inputs, aug_outputs = aug_func(inputs, outputs)
    
    assert aug_inputs.shape == inputs.shape
    assert aug_outputs.shape == outputs.shape


def test_no_augmentation_is_identity():
    """Test that augmentation mode 0 returns original data."""
    torch.manual_seed(42)
    inputs = torch.randn(2, 1, 64, 64, 64)
    outputs = torch.randn(2, 1, 64, 64, 64)
    
    aug_func = AUGMENTATION_FUNCTIONS[0]
    aug_inputs, aug_outputs = aug_func(inputs, outputs)
    
    assert torch.equal(aug_inputs, inputs)
    assert torch.equal(aug_outputs, outputs)


def test_augmentation_batch_independence():
    """Test that augmentation operates independently on batch samples."""
    torch.manual_seed(42)
    inputs = torch.randn(4, 1, 64, 64, 64)
    outputs = torch.randn(4, 1, 64, 64, 64)
    
    # Apply augmentation to full batch
    aug_full_inputs, aug_full_outputs = rotate_x_180(inputs, outputs)
    
    # Apply augmentation to individual samples
    aug_ind_inputs = []
    aug_ind_outputs = []
    for i in range(4):
        aug_in, aug_out = rotate_x_180(inputs[i:i+1], outputs[i:i+1])
        aug_ind_inputs.append(aug_in)
        aug_ind_outputs.append(aug_out)
    
    aug_ind_inputs = torch.cat(aug_ind_inputs, dim=0)
    aug_ind_outputs = torch.cat(aug_ind_outputs, dim=0)
    
    # Should be the same
    assert torch.allclose(aug_full_inputs, aug_ind_inputs)
    assert torch.allclose(aug_full_outputs, aug_ind_outputs)
