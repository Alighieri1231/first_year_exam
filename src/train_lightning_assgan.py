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
from src.datamodules.data_datamodule_seg import WSIDataModule
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
        default="/data/GitHub/first_year_exam/configs/assgan_config_train.yaml",
    )

    args = trainparser.parse_args()
    wandb.init()
    conf = Dict(yaml.safe_load(open(args.config_file, "r")))

    # torch.backends.cudnn.benchmark = True
  

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
    seed_everything(seed=2024, workers=True)

    # Configuración de logging y callbacks
    wandb_logger = WandbLogger(project="first_year_exam", entity="ia-lim", config=conf)
    # early_stop_callback = EarlyStopping(
    #     monitor="valid_dataset_iou", patience=10, mode="max"
    # )
    # model_checkpoint = ModelCheckpoint(
    #     monitor="valid_dataset_iou", mode="max", save_top_k=1, save_last=True
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

    # print(f"Best model path: {model_checkpoint.best_model_path}")
    #   TEST FINAL
    trainer.test(model=model, datamodule=data_module)
    test_metrics = trainer.test(model=model, datamodule=data_module)


    # GUARDAR OVERLAYS DE TEST
    # model.save_test_overlays(data_module, results_path, "test")
    # 3) cargar el mejor modelo y validar de nuevo
    best_path = checkpoint.best_model_path
    if best_path:
        print("Best ckpt:", best_path)
        best_model = USModel.load_from_checkpoint(
            best_path,
            model_opts=conf.model_opts,
            train_par=conf.train_par,
        )
        # validar
        val_metrics = trainer.validate(
            best_model, datamodule=data_module, verbose=False
        )
        val_iou = val_metrics[0]["valid_dataset_iou"]
        # loggeo en wandb
        #wandb.log({"valid_dataset_iou (best)": val_iou})

        # 4) si supera el umbral, loggear ejemplos de test
        if val_iou > 0.4:
            best_model.log_ctest_images(
                data_module,
                threshold=0.2,
                val_iou=val_iou,
                only_roi_frames=True,
                num_images=30,
            )

    else:
        print("No se encontró ningún checkpoint guardado.")

    # cerrar WandB
    wandb.finish()
