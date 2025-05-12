import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
import torch
from src.datamodules.unet_datamodule import SegmentationDataModule
from src.models.unet_model import UNetLightningModule


def cli_main():
    cli = LightningCLI(
        model_class=UNetLightningModule, datamodule_class=SegmentationDataModule
    )


if __name__ == "__main__":
    """
    Entrypoint for training using Lightning CLI.

    Configuration should be provided through a YAML file, e.g.:
    lightning run train --config=config.yaml
    """
    L.seed_everything(42)  # Replace 42 with your desired seed

    # Set matrix multiplication precision to high
    torch.set_float32_matmul_precision("high")

    # Initialize the Wandb logger
    wandb_logger = WandbLogger(project="bluehive", log_model=True)
    cli_main()
