# The syntax for the storage of model is the following
# "model_N + n_samples '_stress' + str(stress_number) + '_ep' + str(n_epochs)"

import torch

# define a UNet3D_2 model less deep than the original UNet3D
class UNet3D_2(torch.nn.Module):
    def __init__(self):
        super(UNet3D_2, self).__init__()
        
        self.activation = torch.nn.LeakyReLU()

        # Encoder
        # input: 64x64x64, output: 32x32x32
        self.e11 = torch.nn.Conv3d(1, 4, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.pool1 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        
        # input: 32x32x32, output: 16x16x16
        self.e21 = torch.nn.Conv3d(4, 8, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.pool2 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        
        # input: 16x16x16, output: 8x8x8
        self.e31 = torch.nn.Conv3d(8, 16, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.pool3 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        
        # input: 8x8x8, output: 4x4x4
        self.e41 = torch.nn.Conv3d(16, 32, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.pool4 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        
        # input: 4x4x4, output: 1x1x1
        self.e51 = torch.nn.Conv3d(32, 64, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        
        # Decoder
        # input 1x1x1, output: 4x4x4
        self.upconv1 = torch.nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.d11 = torch.nn.Conv3d(64, 32, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        
        # input 4x4x4, output: 8x8x8
        self.upconv2 = torch.nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.d21 = torch.nn.Conv3d(32, 16, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        
        # input 8x8x8, output: 16x16x16
        self.upconv3 = torch.nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2)
        self.d31 = torch.nn.Conv3d(16, 8, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        
        # input 16x16x16, output: 32x32x32
        self.upconv4 = torch.nn.ConvTranspose3d(8, 4, kernel_size=2, stride=2)
        self.d41 = torch.nn.Conv3d(8, 4, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        
        # Output layer 32x32x32
        self.outconv = torch.nn.Conv3d(4, 1, kernel_size=1)
        
        
    def forward(self, x):
        # Encoder
        xe11 = self.activation(self.e11(x))
        xp1 = self.pool1(xe11)
        
        xe21 = self.activation(self.e21(xp1))
        xp2 = self.pool2(xe21)
        
        xe31 = self.activation(self.e31(xp2))
        xp3 = self.pool3(xe31)
        
        xe41 = self.activation(self.e41(xp3))
        xp4 = self.pool4(xe41)
        
        xe51 = self.activation(self.e51(xp4))
        
        # Decoder
        xu1 = self.upconv1(xe51)
        xu11 = torch.cat([xu1, xe41], dim=1)
        xd11 = self.activation(self.d11(xu11))
        
        xu2 = self.upconv2(xd11)
        xu22 = torch.cat([xu2, xe31], dim=1)
        xd21 = self.activation(self.d21(xu22))
        
        xu3 = self.upconv3(xd21)
        xu33 = torch.cat([xu3, xe21], dim=1)
        xd31 = self.activation(self.d31(xu33))
        
        xu4 = self.upconv4(xd31)
        xu44 = torch.cat([xu4, xe11], dim=1)
        xd41 = self.activation(self.d41(xu44))
        
        # Output layer
        out = self.outconv(xd41)
        
        return out
        
class UNet3D(torch.nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        
        self.activation = torch.nn.LeakyReLU()

        # input: 64x64x64, output: 32x32x32
        self.e11 = torch.nn.Conv3d(1, 4, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.e12 = torch.nn.Conv3d(4, 4, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.pool1 = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        # input: 32x32x32, output: 16x16x16
        self.e21 = torch.nn.Conv3d(4, 8, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.e22 = torch.nn.Conv3d(8, 8, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.pool2 = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        # input: 16x16x16, output: 8x8x8
        self.e31 = torch.nn.Conv3d(8, 16, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.e32 = torch.nn.Conv3d(16, 16, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.pool3 = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        # input: 8x8x8, output: 4x4x4
        self.e41 = torch.nn.Conv3d(16, 32, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.e42 = torch.nn.Conv3d(32, 32, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.pool4 = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        # input: 4x4x4, output: 1x1x1
        self.e51 = torch.nn.Conv3d(32, 64, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.e52 = torch.nn.Conv3d(64, 64, kernel_size=3, stride=1, padding="same", padding_mode="circular")

        # Decoder
        # input: 4x4x4, output: 8x8x8
        self.upconv1 = torch.nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.d11 = torch.nn.Conv3d(64, 32, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.d12 = torch.nn.Conv3d(32, 32, kernel_size=3, stride=1, padding="same", padding_mode="circular")

        # input: 8x8x8, output: 16x16x16
        self.upconv2 = torch.nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.d21 = torch.nn.Conv3d(32, 16, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.d22 = torch.nn.Conv3d(16, 16, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        
        # input: 16x16x16, output: 32x32x32
        self.upconv3 = torch.nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2)
        self.d31 = torch.nn.Conv3d(16, 8, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.d32 = torch.nn.Conv3d(8, 8, kernel_size=3, stride=1, padding="same", padding_mode="circular")

        # input: 32x32x32, output: 64x64x64
        self.upconv4 = torch.nn.ConvTranspose3d(8, 4, kernel_size=2, stride=2)
        self.d41 = torch.nn.Conv3d(8, 4, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.d42 = torch.nn.Conv3d(4, 4, kernel_size=3, stride=1, padding="same", padding_mode="circular")

        # Output layer 64x64x64
        self.outconv = torch.nn.Conv3d(4, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
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

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = self.activation(self.d11(xu11))
        xd12 = self.activation(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = self.activation(self.d21(xu22))
        xd22 = self.activation(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = self.activation(self.d31(xu33))
        xd32 = self.activation(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = self.activation(self.d41(xu44))
        xd42 = self.activation(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out
    
def save_model(model, save_key, epochs):
    # save model
    torch.save(model.state_dict(), "model/model_save/model_" + save_key + "_ep" + str(epochs) + ".pt")
    # save model as script
    model_scripted = torch.jit.script(model)
    model_scripted.save("model/model_scripted_save/model_scripted_" + save_key + "_ep" + str(epochs) + ".pt")
    
def load_model(model_path_name):
    model = UNet3D()
    model.load_state_dict(torch.load(model_path_name))
    model.eval()
    return model