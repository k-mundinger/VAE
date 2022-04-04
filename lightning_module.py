from dataset import MNISTDataset
from models import Autoencoder
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt



class MNISTAutoencoder(pl.LightningModule):

    def __init__(self, latent_dim = 100, run_name = "test_run"):

        super().__init__()

        self.dataset = MNISTDataset()
        #self.model_type = model_type
        # if self.model_type == "conv":
        #     self.generator = ConvGenerator(depth = depth, latent_dim = latent_dim, img_size=28)
        # elif self.model_type == "fc":
        #     self.generator = FCGenerator(depth = 4, latent_dim = latent_dim, img_size=28)
        # else:
        #     raise NotImplementedError
        self.model = Autoencoder(latent_dim)

        self.run_name = run_name
        self.l2loss = torch.nn.MSELoss()
        self.save_hyperparameters()


    def forward(self, z):
        return self.model(z)



    def training_step(self, batch, batch_idx):

        imgs, _ = batch

        reconstructed = self.model(imgs)

        loss = self.l2loss(imgs, reconstructed)

        self.log("L2-loss", loss)

        return loss



    def configure_optimizers(self):

        lr = 0.001
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)


        return opt


    def on_epoch_end(self):
        

        if self.current_epoch % 3 == 0:

            save_path = Path("/home/konrad/fun/VAE/data/imgs/").joinpath(self.run_name)

            save_path.mkdir(exist_ok = True, parents = True)

            save_path = save_path.joinpath(str(self.current_epoch) + ".png")

            fig, axs = plt.subplots(8, 8, figsize = (20, 20))

            loader = torch.utils.data.DataLoader(self.dataset, batch_size = 64)

            z, _ = next(iter(loader))

            gen_imgs = self.model(z.cuda()).detach().cpu().numpy().transpose(0, 2, 3, 1)

            fig.suptitle(f"Epoch: {self.current_epoch}")

            for i, ax in enumerate(axs.flat):

                renormalized = gen_imgs[i]*0.5 + 0.5*np.ones_like(gen_imgs[i])
                ax.imshow(renormalized, cmap = plt.cm.magma)
                ax.axis("off")
                #ax.set_title(f"")


            plt.savefig(save_path)
            plt.close()


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size = 256)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size = 256)