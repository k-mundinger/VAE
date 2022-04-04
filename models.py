from torch import nn
import torch

class FCEncoder(nn.Module):

    def __init__(self, latent_dim: int):

        super().__init__()

        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.latent_dim)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor):

        out = torch.flatten(x, start_dim = 1)
        out = self.silu(self.fc1(out))
        out = self.silu(self.fc2(out))

        return self.silu(self.fc3(out))

class FCDecoder(nn.Module):

    def __init__(self, latent_dim: int):

        super().__init__()

        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(self.latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 784)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor):
        
        b, _ = x.shape
        
        out = self.silu(self.fc1(x))
        out = self.silu(self.fc2(out))
        out = nn.Tanh()(self.fc3(out))

        return out.reshape(b, 1, 28, 28)
        
class Autoencoder(nn.Module):

    def __init__(self, latent_dim):

        super().__init__()

        self.encoder = FCEncoder(latent_dim)
        self.decoder = FCDecoder(latent_dim)

    def forward(self, x:torch.Tensor):

        out = self.encoder(x)
        
        return self.decoder(out)



        

def build_encoder_block(in_ch, out_ch):

    conv = nn.Conv2d(in_ch,
                     out_ch,
                     kernel_size=3,
                     padding="same",
                     bias=False)

    conv2 = nn.Conv2d(out_ch,
                     out_ch,
                     kernel_size=3,
                     padding="same",
                     bias=False)

    return nn.Sequential(conv, 
                         nn.SiLU(inplace = True),
                         conv2,
                         nn.SiLU(inplace = True))


def build_decoder_block(in_ch, out_ch):

    deconv = nn.ConvTranspose2d(in_ch,
                                out_ch,
                                2,
                                2)

    conv = nn.Conv2d(out_ch,
                     out_ch,
                     kernel_size=3,
                     padding="same",
                     bias=False)

    depth_conv = nn.Conv2d(out_ch,
                           out_ch,
                           kernel_size=1,
                           padding="same",
                           bias=False)

    conv2 = nn.Conv2d(out_ch,
                     out_ch,
                     kernel_size=3,
                     padding="same",
                     bias=True)

    return nn.Sequential(deconv, 
                         nn.SiLU(inplace = True),
                         conv,
                         nn.SiLU(inplace = True),
                         depth_conv,
                         nn.SiLU(inplace = True),
                         conv2)