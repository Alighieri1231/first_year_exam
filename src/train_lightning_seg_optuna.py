import optuna
import yaml
from addict import Dict
import os
import torch
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import lightning as L
import optuna.visualization as vis

from model_smp import USModel
from data_datamodule_seg import WSIDataModule

# Función de optimización con Optuna
def objective(trial):
    # Hiperparámetros a optimizar
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    #encoder_depth = trial.suggest_int("encoder_depth", 4, 5)

    # Cargar configuración YAML
    with open("/data/UNET2D/default_config_train_seg_optuna.yaml", "r") as f:
        conf = Dict(yaml.safe_load(f))

    # Reemplazar los valores optimizados en la configuración
    conf.train_par.lr = learning_rate
    conf.train_par.weight_decay = weight_decay
    #conf.model_opts.args.encoder_depth = encoder_depth

    # Configuración reproducible
    seed_everything(seed=2024, workers=True)

    # Definir DataModule con los nuevos hiperparámetros
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

    # Configuración de logging y callbacks
    wandb_logger = WandbLogger(project="2dheart_optuna", entity="ia-lim", config=conf)
    early_stop_callback = EarlyStopping(
        monitor="valid_dataset_iou",
        patience=5,
        mode="max"
    )
    model_checkpoint = ModelCheckpoint(
        monitor="valid_dataset_iou",
        mode="max",
    )

    # Crear modelo con hiperparámetros ajustados
    model = USModel(model_opts=conf.model_opts,train_par=conf.train_par)

    # Configurar el entrenador con los valores optimizados
    trainer = Trainer(
        max_epochs=conf.train_par.epochs,
        accelerator="auto",
        devices=conf.train_par.devices,
        strategy=conf.train_par.strategy,
        logger=wandb_logger,
        profiler=conf.train_par.profiler,
        callbacks=[early_stop_callback, model_checkpoint],
        deterministic=True,
    )

    # Entrenar modelo
    trainer.fit(model, datamodule=data_module)

    # Evaluar en validación
    metrics = trainer.validate(model, datamodule=data_module)
    val_iou = metrics[0]["valid_dataset_iou"]

    return val_iou  # Optuna intentará maximizar esta métrica

# Configurar el estudio de Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=2)  # Número de combinaciones a probar

# Imprimir mejores hiperparámetros
print("Best hyperparameters:", study.best_params)


# Gráfico de la evolución del entrenamiento
vis.plot_optimization_history(study).show()

# Importancia de cada hiperparámetro en la optimización
vis.plot_param_importances(study).show()
