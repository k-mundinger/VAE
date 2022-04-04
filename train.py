import pytorch_lightning as pl
from lightning_module import MNISTAutoencoder


MODEL_TYPE = "conv"



dim = 10

RUN_NAME = "fc_lat_dim_" + str(dim)


model = MNISTAutoencoder(latent_dim=dim, run_name = RUN_NAME)

logger = pl.loggers.TensorBoardLogger("/home/konrad/fun/VAE/logs", name=RUN_NAME)

lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

trainer = pl.Trainer(gpus = [0],
                    max_epochs = 30,
                    logger = logger,
                    callbacks = [lr_monitor])

trainer.fit(model)