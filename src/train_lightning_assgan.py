import argparse
import yaml
from addict import Dict
import wandb
import os

import torch
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import matplotlib.pyplot as plt
import cv2
import lightning as L

# from model_lightning_seg import MyModel
from src.models.assgan_model import ASSGAN as USModel
from src.datamodules.data_datamodule_seg_un import WSIDataModule
import random
import numpy as np


def main():
    trainparser = argparse.ArgumentParser(
        description="[StratifIAD] Parameters for training", allow_abbrev=False
    )
    trainparser.add_argument(
        "-c",
        "--config-file",
        type=str,
        default="/data/GitHub/first_year_exam/configs/assgan_config_train.yaml",
    )

    args = trainparser.parse_args()

    conf = Dict(yaml.safe_load(open(args.config_file, "r")))

    #

    torch.set_float32_matmul_precision("medium")
    data_dir = conf.dataset.data_dir
    train_file = conf.dataset.train
    dev_file = conf.dataset.dev
    test_file = conf.dataset.test
    cache_data = conf.dataset.cache_data
    rescale_factor = conf.dataset.rescale_factor
    tb_exp_name = f"{conf.dataset.experiment}_model"
    wandb.init(
        project=conf.dataset.project, entity="ia-lim", config=conf, name=tb_exp_name
    )

    # Setting a random seed for reproducibility
    if conf.train_par.random_seed == "default":
        random_seed = 2024
    else:
        random_seed = conf.train_par.random_seed

    seed_everything(seed=random_seed, workers=True)

    # Create a DataModule
    data_module = WSIDataModule(
        batch_size=conf.train_par.batch_size,
        workers=conf.train_par.workers,
        train_file=train_file,
        dev_file=dev_file,
        test_file=test_file,
        data_dir=data_dir,
        cache_data=cache_data,
        random_seed=random_seed,
    )

    results_path = os.path.join(conf.train_par.results_path, conf.dataset.experiment)
    os.makedirs(results_path, exist_ok=True)
    conf.train_par.results_model_filename = os.path.join(results_path, f"{tb_exp_name}")

    seed_everything(seed=2024, workers=True)

    # Configuración de logging y callbacks
    # wandb_logger = WandbLogger(project="first_year_exam", entity="ia-lim", config=conf)

    wandb_logger = WandbLogger(
        project=conf.dataset.project, entity="ia-lim", config=conf, name=tb_exp_name
    )
    # actualiza el config existente
    # wandb_logger.experiment.config.update(conf, allow_val_change=True)
    # early_stop_callback = EarlyStopping(
    #     monitor="valid_dataset_iou", patience=10, mode="max"
    # )

    checkpoint = ModelCheckpoint(
        monitor="valid_dataset_iou",
        mode="max",
        save_top_k=1,
        save_last=True,
        dirpath=os.path.join(conf.train_par.results_path, conf.dataset.experiment),
        filename="assgan-{epoch:03d}-{valid_dataset_iou:.4f}",
    )

    # lightning_model = MyModel(model_opts=conf.model_opts, train_par=conf.train_par)
    model = USModel(model_opts=conf.model_opts, train_par=conf.train_par)
    # Configurar el entrenador con los valores optimizados
    trainer = Trainer(
        max_epochs=conf.train_par.epochs,
        accelerator="auto",
        devices=conf.train_par.devices,
        strategy=conf.train_par.strategy,
        logger=wandb_logger,
        profiler=conf.train_par.profiler,
        callbacks=[checkpoint],
        precision="bf16-mixed",
    )
    # deterministic=True,
    # )

    # Entrenar modelo
    trainer.fit(model, datamodule=data_module)

    trainer.test(model=model, datamodule=data_module)

    # Evaluar en validación del mejor modelo
    if checkpoint.best_model_path:
        best_model = USModel.load_from_checkpoint(
            checkpoint.best_model_path,
            model_opts=conf.model_opts,
            train_par=conf.train_par,
        )
        metrics = trainer.validate(best_model, datamodule=data_module)
        val_iou = metrics[0]["valid_dataset_iou"]

        # Loggear el mejor resultado en WandB
        wandb.log({"valid dataset iou (best model)": val_iou})

        # Evaluar en test solo si el mejor modelo tiene val_iou > 0.4
        if val_iou > 0.4:
            trainer.test(best_model, datamodule=data_module)
            best_model.log_test_images(
                data_module,
                num_images=50,
                val_iou=val_iou,
                threshold=0.4,
                only_roi_frames=True,
            )

    return val_iou


if __name__ == "__main__":
    main()
