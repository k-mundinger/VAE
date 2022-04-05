from torch import nn
import torch

class FCEncoder(nn.Module):

    def __init__(self, latent_dim: int, shape: int, ch: int):

        super().__init__()

        self.shape = shape
        self.ch = ch

        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(self.ch * self.shape**2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.latent_dim)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor):

        out = torch.flatten(x, start_dim = 1)
        out = self.silu(self.fc1(out))
        out = self.silu(self.fc2(out))

        return self.silu(self.fc3(out))

class FCDecoder(nn.Module):

    def __init__(self, latent_dim: int, shape: int, ch: int):

        super().__init__()

        self.shape = shape
        self.ch = ch

        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(self.latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, self.ch * self.shape**2)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor):
        
        b, _ = x.shape
        
        out = self.silu(self.fc1(x))
        out = self.silu(self.fc2(out))
        out = nn.Tanh()(self.fc3(out))

        return out.reshape(b, self.ch, self.shape, self.shape)
        
class Autoencoder(nn.Module):

    def __init__(self, latent_dim, shape, ch):

        super().__init__()

        self.encoder = FCEncoder(latent_dim, shape, ch)
        self.decoder = FCDecoder(latent_dim, shape, ch)

    def forward(self, x:torch.Tensor):

        out = self.encoder(x)
        
        return self.decoder(out)



class VariationalEncoder(nn.Module):

    def __init__(self, latent_dim: int, shape: int, ch: int):

        super().__init__()

        self.shape = shape
        self.ch = ch

        self.fc1 = nn.Linear(self.ch * self.shape**2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)
        self.silu = nn.SiLU()

    def forward(self, x):

        out = torch.flatten(x, start_dim=1)
        out = self.silu(self.fc1(out))
        out = self.silu(self.fc2(out))

        mu =  self.fc_mean(out)

        sigma = torch.exp(self.fc_var(out))

        return mu, sigma
        

class VAE(nn.Module):

    def __init__(self, latent_dim: int, shape: int, ch: int):

        super().__init__()

        self.encoder = VariationalEncoder(latent_dim, shape, ch)
        self.decoder = FCDecoder(latent_dim, shape, ch)
        self.kl_loss = 0

    def forward(self, x:torch.Tensor):

        mu, sigma = self.encoder(x)

        self.kl_loss = (-0.5*(1+torch.log(sigma) - mu**2- sigma).sum(dim = 1)).mean(dim =0)

        z = torch.randn(size = (mu.size(0),mu.size(1)))
        z= z.type_as(mu) 
        reparametrized = mu + sigma*z
        
        return self.decoder(reparametrized)


class VariationalConvEncoder(nn.Module):

    def __init__(self, latent_dim: int = 20, shape: int = 28, ch: int = 1):

        super().__init__()

        self.shape = shape
        self.ch = ch

        self.fc1 = nn.Linear(self.ch * self.shape**2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)
        self.silu = nn.SiLU()

    def forward(self, x):

        out = torch.flatten(x, start_dim=1)
        out = self.silu(self.fc1(out))
        out = self.silu(self.fc2(out))

        mu =  self.fc_mean(out)

        sigma = torch.exp(self.fc_var(out))

        return mu, sigma