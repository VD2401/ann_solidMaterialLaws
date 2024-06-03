import torch

# Rotate along x-axis 180°
def augment1(inputs, outputs):
  return torch.rot90(inputs, k=2, dims=[3,4]), torch.rot90(outputs, k=2, dims=[3,4])

# Rotate along y-axis 180°
def augment2(inputs, outputs):
  return torch.rot90(inputs, k=2, dims=[2,4]), torch.rot90(outputs, k=2, dims=[2,4])

# Rotate along z-axis 180°
def augment3(inputs, outputs):
  return torch.rot90(inputs, k=2, dims=[2,3]), torch.rot90(outputs, k=2, dims=[2,3])

# flip along y,z axis
def augment4(inputs, outputs):
  return torch.flip(inputs, dims=[3,4]), torch.flip(outputs, dims=[3,4])

# flip along x,z axis
def augment5(inputs, outputs):
  return torch.flip(inputs, dims=[2,4]), torch.flip(outputs, dims=[2,4])

# flip along x,y axis
def augment6(inputs, outputs):
  return torch.flip(inputs, dims=[2,3]), torch.flip(outputs, dims=[2,3])
