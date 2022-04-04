import pytorch_lightning as pl
from lightning_module import MNISTAutoencoder, MNISTVAE


MODEL_TYPE = "conv"



dim = 20
LR = .001

RUN_NAME = "fewer_params_alpha_0.008_VAE_lat_dim_" + str(dim) + "_lr_" + str(LR)


#model = MNISTAutoencoder(latent_dim=dim, run_name = RUN_NAME)
model = MNISTVAE(latent_dim=dim, lr = LR, run_name = RUN_NAME)

logger = pl.loggers.TensorBoardLogger("/home/konrad/fun/VAE/logs", name=RUN_NAME)

lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(gpus = [0],
                    max_epochs = 100,
                    logger = logger,
                    callbacks = [lr_monitor])

trainer.fit(model)