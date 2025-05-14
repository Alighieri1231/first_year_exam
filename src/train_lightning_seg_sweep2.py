import wandb
import argparse
import yaml
from addict import Dict
import os
import torch
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import lightning as L

from src.models.model_smp import USModel
from src.datamodules.data_datamodule_seg import WSIDataModule


# Definir la función de entrenamiento que WandB Sweeps ejecutará
def main():
    # Inicializar el experimento de WandB con los valores del Sweep
    wandb.init()
    print("Wandb initialized")
    torch.set_float32_matmul_precision("medium")

    # Cargar configuración YAML
    with open(
        "/data/GitHub/first_year_exam/configs/default_config_train.yaml",
        "r",
    ) as f:
        conf = Dict(yaml.safe_load(f))
    # Obtener hiperparámetros del Sweep
    sweep_config = wandb.config

    conf.train_par.lr = sweep_config.learning_rate
    conf.train_par.weight_decay = sweep_config.weight_decay
    conf.train_par.optimizer = sweep_config.optimizer
    # conf.train_par.scheduler = sweep_config.scheduler
    conf.train_par.scheduler = "step_lr"
    conf.model_opts.args.encoder_name = sweep_config.encoder_name
    # arch
    conf.model_opts.args.arch = sweep_config.arch

    # Configurar DataModule con los nuevos hiperparámetros
    data_module = WSIDataModule(
        batch_size=conf.train_par.batch_size,
        workers=conf.train_par.workers,
        train_file=conf.dataset.train,
        dev_file=conf.dataset.dev,
        test_file=conf.dataset.test,
        data_dir=conf.dataset.data_dir,
        cache_data=conf.dataset.cache_data,
        random_seed=2024,
    )

    # Configuración reproducible
    seed_everything(seed=2024, workers=True)

    # Configuración de logging y callbacks
    wandb_logger = WandbLogger(project="first_year_sweep", entity="ia-lim", config=conf)
    early_stop_callback = EarlyStopping(
        monitor="valid_dataset_iou", patience=10, mode="max"
    )
    model_checkpoint = ModelCheckpoint(
        monitor="valid_dataset_iou", mode="max", save_top_k=1, save_last=True
    )

    # Crear modelo con hiperparámetros ajustados
    model = USModel(model_opts=conf.model_opts, train_par=conf.train_par)

    # Configurar el entrenador con los valores optimizados
    trainer = Trainer(
        max_epochs=conf.train_par.epochs,
        accelerator="auto",
        devices=conf.train_par.devices,
        strategy=conf.train_par.strategy,
        logger=wandb_logger,
        profiler=conf.train_par.profiler,
        callbacks=[early_stop_callback, model_checkpoint],
        precision="bf16-mixed",)
    #     deterministic=True,
    # )

    # Entrenar modelo
    trainer.fit(model, datamodule=data_module)

    print(f"Best model path: {model_checkpoint.best_model_path}")

    # Evaluar en validación del mejor modelo
    if model_checkpoint.best_model_path:
        best_model = USModel.load_from_checkpoint(
            model_checkpoint.best_model_path,
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
                data_module, num_images=10, val_iou=val_iou, threshold=0.4, only_roi_frames=True
            )

    return val_iou


if __name__ == "__main__":
    main()
