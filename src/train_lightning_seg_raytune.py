import wandb
import yaml
import os
import torch
import lightning as L
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from addict import Dict
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from model_smp import USModel
from data_datamodule_seg import WSIDataModule

# Función de entrenamiento compatible con Ray Tune
def train_tune(config):
    #set matmult precision high
    torch.set.float32_matmul_precision("high")
    
    # Inicializar W&B dentro del experimento
    #wandb.init(project="2dheart_ray", entity="ia-lim", config=config)

    # Cargar configuración YAML
    with open("/data/UNET2D/default_config_train_seg copy.yaml", "r") as f:
        conf = Dict(yaml.safe_load(f))

    # Asignar hiperparámetros desde Ray Tune
    conf.train_par.lr = config["learning_rate"]
    conf.train_par.weight_decay = config["weight_decay"]

    # Configuración reproducible
    seed_everything(2024, workers=True)

    # Inicializar el módulo de datos
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

    # Inicializar el modelo
    model = USModel(model_opts=conf.model_opts,train_par=conf.train_par)

    # Configurar logger y callbacks
    wandb_logger = WandbLogger(project="2dheart_ray", entity="ia-lim", config=conf)
    early_stop = EarlyStopping(monitor="valid_dataset_iou", patience=10, mode="max")
    checkpoint = ModelCheckpoint(monitor="valid_dataset_iou", mode="max")

    # Configurar Trainer
    trainer = Trainer(
        max_epochs=conf.train_par.epochs,
        accelerator="auto",
        devices=conf.train_par.devices,
        strategy=conf.train_par.strategy,
        logger=wandb_logger,
        profiler=conf.train_par.profiler,
        callbacks=[early_stop, checkpoint],
        deterministic=True,
        precision="bf16-mixed",
        enable_progress_bar=False

    )

    # Entrenar modelo
    trainer.fit(model, datamodule=data_module)

    # Evaluar en validación
    metrics = trainer.validate(model, datamodule=data_module)
    val_iou = metrics[0].get("valid_dataset_iou", 0)

    # Loggear el resultado en W&B y Ray Tune
    #wandb.log({"valid_dataset_iou": val_iou})
    tune.report(valid_dataset_iou=val_iou)

    # Finalizar W&B correctamente
    #wandb.finish()


# Definir espacio de búsqueda de hiperparámetros
search_space = {
    "learning_rate": tune.loguniform(1e-5, 1e-2),
    "weight_decay": tune.loguniform(1e-6, 1e-2),
}

# Configurar ASHAScheduler para optimización eficiente
scheduler = ASHAScheduler(
    metric="valid_dataset_iou",
    mode="max",
    max_t=100,
    grace_period=10,
    reduction_factor=2
)

if __name__ == "__main__":
    # Inicializar Ray
    ray.init()

    # Ejecutar búsqueda de hiperparámetros con Ray Tune
    analysis = tune.run(
        train_tune,
        loggers=[WandbLogger],
        config=search_space,
        num_samples=4,  # Número de configuraciones a probar
        scheduler=scheduler,
        resources_per_trial={"cpu": 8, "gpu": 1},  # Ajusta según disponibilidad
        max_concurrent_trials=2,  # Máximo 2 pruebas simultáneas


    )

    # Obtener la mejor configuración de hiperparámetros
    best_config = analysis.get_best_config(metric="valid_dataset_iou", mode="max")
    print(f"✅ Mejor configuración encontrada: {best_config}")

    # Cerrar Ray
    ray.shutdown()
