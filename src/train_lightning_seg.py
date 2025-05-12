import argparse
import yaml
from addict import Dict
import wandb
import os

import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import matplotlib.pyplot as plt
import cv2
import lightning as L

# from model_lightning_seg import MyModel
from model_smp import USModel
from data_datamodule_seg import WSIDataModule
import random
import numpy as np

if __name__ == "__main__":
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    trainparser = argparse.ArgumentParser(
        description="[StratifIAD] Parameters for training", allow_abbrev=False
    )
    trainparser.add_argument(
        "-c",
        "--config-file",
        type=str,
        default="/data/GitHub/UNET2D/default_config_train_seg copy.yaml",
    )

    args = trainparser.parse_args()

    conf = Dict(yaml.safe_load(open(args.config_file, "r")))

    #torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")
    data_dir = conf.dataset.data_dir
    train_file = conf.dataset.train
    dev_file = conf.dataset.dev
    test_file = conf.dataset.test
    cache_data = conf.dataset.cache_data
    rescale_factor = conf.dataset.rescale_factor

    # name = dev_file.replace('dev','test').split('.')[0].split('/')[-1]
    # tb_exp_name = f'{conf.dataset.experiment}_{name}_patchSize_{rescale_factor}'
    tb_exp_name = f"{conf.dataset.experiment}_model"

    # Setting a random seed for reproducibility
    if conf.train_par.random_seed == "default":
        random_seed = 2024
    else:
        random_seed = conf.train_par.random_seed

    seed_everything(seed=random_seed, workers=True)
    # print(dev_file)
    # print(data_dir)
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
    # Tamaño del batch de imágenes: torch.Size([1, 1, 128, 128, 128])
    # Tamaño del batch de etiquetas: torch.Size([1])

  
    results_path = os.path.join(conf.train_par.results_path, conf.dataset.experiment)
    os.makedirs(results_path, exist_ok=True)
    # print(results_path)
    conf.train_par.results_model_filename = os.path.join(results_path, f"{tb_exp_name}")
    # print(conf.train_par.results_model_filename)
    # wandb logger
    wandb_logger = WandbLogger(project="2dheart_segmentation", entity="ia-lim", config=conf)
    early_stop_callback = EarlyStopping(
        monitor="valid_dataset_iou",
        min_delta=0.00,
        patience=conf.train_par.patience,
        verbose=True,
        mode="max",
    )

    model_checkpoint = ModelCheckpoint(
        filename=conf.train_par.results_model_filename,
        monitor="valid_dataset_iou",
        mode="max",
    )
    # lightning_model = MyModel(model_opts=conf.model_opts, train_par=conf.train_par)
    lightning_model = USModel(model_opts=conf.model_opts)

    trainer = L.Trainer(
        # max_epochs=conf.train_par.epochs, accelerator="auto", devices="auto", precision='bf16-mixed',logger=wandb_logger,callbacks=[early_stop_callback,model_checkpoint],
        max_epochs=conf.train_par.epochs,
        accelerator="auto",
        devices=conf.train_par.devices,
        strategy=conf.train_par.strategy,
        logger=wandb_logger,
        profiler=conf.train_par.profiler,
        callbacks=[early_stop_callback, model_checkpoint],
        deterministic=True,
        # default_root_dir=results_path,  # +"/"#,log_every_n_steps=46
    )

    trainer.fit(model=lightning_model, datamodule=data_module)
    #   TEST FINAL
    trainer.test(model=lightning_model, datamodule=data_module)

    # GUARDAR OVERLAYS DE TEST
    lightning_model.save_test_overlays(data_module, results_path,"test_overlays1e4cosine")