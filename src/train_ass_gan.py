import os
import torch
import lightning as L
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from gan_model import CycleGAN  # Importa tu modelo CycleGAN en Lightning
from gan_datamodule import ImageDataModule  # Importa tu DataModule
from lightning.pytorch import seed_everything

# seed everything
seed_everything(seed=2024, workers=True)


torch.set_float32_matmul_precision('medium')

# hyperparameters
BATCH_SIZE = 8
WORKERS = 4
EPOCHS = 300
LEARNING_RATE_G = 3e-4
LEARNING_RATE_D = 3e-3

#Path config
base_dir = "/data/ob_process/csv"
data_dir_low = "/data/ob_process/low_quality_images"
data_dir_high = "/data/ob_process/high_quality_images"

#logger de WandB
wandb_logger = WandbLogger(project="CycleGAN", name="training_run")

# Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="CycleGAN_15res_steplr-{epoch:02d}-{val_ssim:.4f}",
    monitor="val_ssim",
    save_top_k=3,
    mode="max",  # Queremos el SSIM más alto
    save_weights_only=True
)


lr_monitor = LearningRateMonitor(logging_interval="epoch")

# Instantiate DataModule
data_module = ImageDataModule(
    batch_size=BATCH_SIZE,
    workers=WORKERS,
    train_csv=os.path.join(base_dir, "train.csv"),
    val_csv=os.path.join(base_dir, "validation.csv"),
    test_csv=os.path.join(base_dir, "test.csv"),
    data_dir_low=data_dir_low,
    data_dir_high=data_dir_high
)

# Instantiate CycleGAN model
model = CycleGAN(
    lr_g=LEARNING_RATE_G,
    lr_d=LEARNING_RATE_D,
    lambda_adv=1,
    lambda_cycle=12,
    lambda_id=1,
    lambda_per=1
)

# trainer Lightning
trainer = L.Trainer(
    max_epochs=EPOCHS,
    accelerator="gpu",
    strategy='ddp_find_unused_parameters_true',
    devices=2,  # Change this to the number of GPUs you want to use
    logger=wandb_logger,
    callbacks=[checkpoint_callback, lr_monitor],
    precision='bf16'  # Mixed precision 
)

# TRain model
trainer.fit(model, datamodule=data_module)

test_results = trainer.test(model, datamodule=data_module)


# close wandb
wandb.finish()
