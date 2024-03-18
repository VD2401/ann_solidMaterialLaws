import torch
import model2Delastic.plot as plot

data = torch.load('data_elasticity_3D.pt')

for key,value in data.items():
	print(key)
	print(value.shape)

plot.volume(torch.arange(0, 64), torch.arange(0, 64), torch.arange(0, 64), data['young_modulus'].select(0, 2))
plot.volume(torch.arange(0, 64), torch.arange(0, 64), torch.arange(0, 64), 
			data['stress'].select(5, 0).select(1, 0).select(0, 0))