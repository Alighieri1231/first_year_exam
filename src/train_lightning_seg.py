import argparse
import yaml
from addict import Dict
import wandb
import os

import torch
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.models.model_smp import USModel
from src.datamodules.data_datamodule_seg import WSIDataModule
from src.datamodules.data_datamodule_seg_un import WSIDataModule as WSIDataModuleUn
import random


def main():
    # ——— Parser de argumentos ———
    parser = argparse.ArgumentParser(
        description="[StratifIAD] Parámetros para training", allow_abbrev=False
    )
    parser.add_argument(
        "-c",
        "--config-file",
        type=str,
        default="/data/GitHub/first_year_exam/configs/default_config_train.yaml",
        help="Ruta al YAML de configuración",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="run0",
        help="Identificador de la corrida (se usa en el nombre de W&B)",
    )
    parser.add_argument(
        "--data-dir-override", type=str, help="Reemplaza a conf.dataset.data_dir"
    )
    parser.add_argument(
        "--train-file-override",
        type=str,
        help="Nombre del CSV de train (relativo a data_dir)",
    )
    parser.add_argument(
        "--dev-file-override",
        type=str,
        help="Nombre del CSV de validación (relativo a data_dir)",
    )
    parser.add_argument(
        "--test-file-override",
        type=str,
        help="Nombre del CSV de test (relativo a data_dir)",
    )
    parser.add_argument(
        "--arch-override",
        type=str,
        choices=["pspnet", "manet", "unetplusplus", "unet", "deeplabv3plus"],
        help="Override de la arquitectura (model_opts.arch)",
    )
    args = parser.parse_args()

    # ——— Carga del YAML de configuración ———
    conf = Dict(yaml.safe_load(open(args.config_file, "r")))

    # ——— Overrides por CLI ———
    if args.data_dir_override:
        conf.dataset.data_dir = args.data_dir_override
    if args.train_file_override:
        conf.dataset.train = args.train_file_override
    if args.dev_file_override:
        conf.dataset.dev = args.dev_file_override
    if args.test_file_override:
        conf.dataset.test = args.test_file_override
    if args.arch_override:
        # sobreescribimos arch y lo pasamos a mayúsculas como en tu YAML original si lo necesitas
        conf.model_opts.arch = args.arch_override

    # ——— Variables actualizadas desde conf ———
    torch.set_float32_matmul_precision("medium")
    data_dir = conf.dataset.data_dir
    train_file = conf.dataset.train
    dev_file = conf.dataset.dev
    test_file = conf.dataset.test
    cache_data = conf.dataset.cache_data
    rescale_factor = conf.dataset.rescale_factor

    # ——— Nombre de la corrida en W&B ———
    tb_exp_name = f"{conf.dataset.experiment}_{args.run_id}"
    wandb.init(
        project=conf.dataset.project, entity="ia-lim", config=conf, name=tb_exp_name
    )

    # ——— Fijar semilla para reproducibilidad ———
    if conf.train_par.random_seed == "default":
        random_seed = 2024
    else:
        random_seed = conf.train_par.random_seed
    seed_everything(seed=random_seed, workers=True)

    # ——— Creación del DataModule ———
    if conf.dataset.unlabeled_dataset:
        data_module = WSIDataModuleUn(
            batch_size=conf.train_par.batch_size,
            workers=conf.train_par.workers,
            train_file=train_file,
            dev_file=dev_file,
            test_file=test_file,
            data_dir=data_dir,
            cache_data=cache_data,
            random_seed=random_seed,
        )
    else:
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
    wandb_logger = WandbLogger(
        project=conf.dataset.project, entity="ia-lim", config=conf, name=tb_exp_name
    )
    if conf.train_par.early_stopping_flag:
        early_stop_callback = EarlyStopping(
            monitor="valid_dataset_iou", patience=conf.train_par.patience, mode="max"
        )
    model_checkpoint = ModelCheckpoint(
        monitor="valid_dataset_iou", mode="max", save_top_k=1, save_last=True
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
        callbacks=[early_stop_callback, model_checkpoint],
        precision="bf16-mixed",
    )
    # deterministic=True,
    # )

    # Entrenar modelo
    trainer.fit(model, datamodule=data_module)

    print(f"Best model path: {model_checkpoint.best_model_path}")
    #   TEST FINAL
    trainer.test(model=model, datamodule=data_module)

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
        if val_iou > conf.train_par.eval_threshold:
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
