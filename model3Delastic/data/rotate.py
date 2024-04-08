import torch 
from numpy import floor

def rotate180(inputs, axis):
    if inputs.size(0) < 32:
        return torch.flip(inputs, [1 + (axis+1)%3, 1 + (axis+2)%3])
    else:
        n32 = floor(inputs.size(0)/32)
        n32 = int(n32)
        for i in range(n32):
            inputs[i*32:(i+1)*32] = torch.flip(inputs[i*32:(i+1)*32], [1 + (axis+1)%3, 1 + (axis+2)%3])
        inputs[n32*32:] = torch.flip(inputs[n32*32:], [1 + (axis+1)%3, 1 + (axis+2)%3])
        return inputs

def mirror_xx(inputs):
    return rotate180(rotate180(inputs, 1), 2)

def periodic_shift(inputs, shift=8):
    if inputs.size(0) < 32:
    # Tensor of size N x nx x ny x nz
    # We shift periodically on nx
        return torch.cat((inputs[:, -shift:, ...], inputs[:,:-shift, ...]), 1)
    else:
        n32 = floor(inputs.size(0)/32)
        n32 = int(n32)
        return torch.cat((inputs[:, -shift:, ...], inputs[:, :n32*32-shift, ...]), 1)

