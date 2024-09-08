import torch

# %%
class UNet3D(torch.nn.Module):
    def __init__(self,
                 width1=4,
                 width2=8,
                 width3=16,
                 width4=32,
                 width5=64,):
        super(UNet3D, self).__init__()

        self.activation = torch.nn.LeakyReLU()
        self.retrain = False
        self.depth = 5
        self.width1 = width1
        self.width2 = width2
        self.width3 = width3
        self.width4 = width4
        self.width5 = width5
        self.kernel_size_conv1 = 3
        self.kernel_size_conv2 = 3
        self.stride_conv1 = 1
        self.stride_conv2 = 1
        self.padding_conv1 = "same"
        self.padding_conv2 = "same"
        self.padding_mode_conv1 = "circular"
        self.padding_mode_conv2 = "circular"
        self.kernel_size_pool = 2
        self.stride_pool = 2

        # input: 64x64x64, output: 32x32x32
        self.e11 = torch.nn.Conv3d(1, self.width1, kernel_size=self.kernel_size_conv1, stride=self.stride_conv1, padding=self.padding_conv1, padding_mode=self.padding_mode_conv1)
        self.e12 = torch.nn.Conv3d(self.width1, self.width1, kernel_size=self.kernel_size_conv2, stride=self.stride_conv2, padding=self.padding_conv2, padding_mode=self.padding_mode_conv2)
        self.pool1 = torch.nn.MaxPool3d(kernel_size=self.kernel_size_pool, stride=self.stride_pool)

        # input: 32x32x32, output: 16x16x16
        self.e21 = torch.nn.Conv3d(self.width1, self.width2, kernel_size=self.kernel_size_conv1, stride=self.stride_conv1, padding=self.padding_conv1, padding_mode=self.padding_mode_conv1)
        self.e22 = torch.nn.Conv3d(self.width2, self.width2, kernel_size=self.kernel_size_conv2, stride=self.stride_conv2, padding=self.padding_conv2, padding_mode=self.padding_mode_conv2)
        self.pool2 = torch.nn.MaxPool3d(kernel_size=self.kernel_size_pool, stride=self.stride_pool)

        # input: 16x16x16, output: 8x8x8
        self.e31 = torch.nn.Conv3d(self.width2, self.width3, kernel_size=self.kernel_size_conv1, stride=self.stride_conv1, padding=self.padding_conv1, padding_mode=self.padding_mode_conv1)
        self.e32 = torch.nn.Conv3d(self.width3, self.width3, kernel_size=self.kernel_size_conv2, stride=self.stride_conv2, padding=self.padding_conv2, padding_mode=self.padding_mode_conv2)
        self.pool3 = torch.nn.MaxPool3d(kernel_size=self.kernel_size_pool, stride=self.stride_pool)

        # input: 8x8x8, output: 4x4x4
        self.e41 = torch.nn.Conv3d(self.width3, self.width4, kernel_size=self.kernel_size_conv1, stride=self.stride_conv1, padding=self.padding_conv1, padding_mode=self.padding_mode_conv1)
        self.e42 = torch.nn.Conv3d(self.width4, self.width4, kernel_size=self.kernel_size_conv2, stride=self.stride_conv2, padding=self.padding_conv2, padding_mode=self.padding_mode_conv2)
        self.pool4 = torch.nn.MaxPool3d(kernel_size=self.kernel_size_pool, stride=self.stride_pool)

        # input: 4x4x4, output: 1x1x1
        self.e51 = torch.nn.Conv3d(self.width4, self.width5, kernel_size=self.kernel_size_conv1, stride=self.stride_conv1, padding=self.padding_conv1, padding_mode=self.padding_mode_conv1)
        self.e52 = torch.nn.Conv3d(self.width5, self.width5, kernel_size=self.kernel_size_conv2, stride=self.stride_conv2, padding=self.padding_conv2, padding_mode=self.padding_mode_conv2)
        
        # Decoder
        # input: 4x4x4, output: 8x8x8
        self.upconv1 = torch.nn.ConvTranspose3d(self.width5, self.width4, kernel_size=self.kernel_size_pool, stride=self.stride_pool)
        self.d11 = torch.nn.Conv3d(self.width5, self.width4, kernel_size=self.kernel_size_conv1, stride=self.stride_conv1, padding=self.padding_conv1, padding_mode=self.padding_mode_conv1)
        self.d12 = torch.nn.Conv3d(self.width4, self.width4, kernel_size=self.kernel_size_conv2, stride=self.stride_conv2, padding=self.padding_conv2, padding_mode=self.padding_mode_conv2)

        # input: 8x8x8, output: 16x16x16
        self.upconv2 = torch.nn.ConvTranspose3d(self.width4, self.width3, kernel_size=self.kernel_size_pool, stride=self.stride_pool)
        self.d21 = torch.nn.Conv3d(self.width4, self.width3, kernel_size=self.kernel_size_conv1, stride=self.stride_conv1, padding=self.padding_conv1, padding_mode=self.padding_mode_conv1)
        self.d22 = torch.nn.Conv3d(self.width3, self.width3, kernel_size=self.kernel_size_conv2, stride=self.stride_conv2, padding=self.padding_conv2, padding_mode=self.padding_mode_conv2)
        
        # input: 16x16x16, output: 32x32x32
        self.upconv3 = torch.nn.ConvTranspose3d(self.width3, self.width2, kernel_size=self.kernel_size_pool, stride=self.stride_pool)
        self.d31 = torch.nn.Conv3d(self.width3, self.width2, kernel_size=self.kernel_size_conv1, stride=self.stride_conv1, padding=self.padding_conv1, padding_mode=self.padding_mode_conv1)
        self.d32 = torch.nn.Conv3d(self.width2, self.width2, kernel_size=self.kernel_size_conv2, stride=self.stride_conv2, padding=self.padding_conv2, padding_mode=self.padding_mode_conv2)

        # input: 32x32x32, output: 64x64x64
        self.upconv4 = torch.nn.ConvTranspose3d(self.width2, self.width1, kernel_size=self.kernel_size_pool, stride=self.stride_pool)
        self.d41 = torch.nn.Conv3d(self.width2, self.width1, kernel_size=self.kernel_size_conv1, stride=self.stride_conv1, padding=self.padding_conv1, padding_mode=self.padding_mode_conv1)
        self.d42 = torch.nn.Conv3d(self.width1, self.width1, kernel_size=self.kernel_size_conv2, stride=self.stride_conv2, padding=self.padding_conv2, padding_mode=self.padding_mode_conv2)

        # Output layer 64x64x64
        self.outconv = torch.nn.Conv3d(self.width1, 1, kernel_size=1, stride=1, padding=0, padding_mode="zeros")

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
