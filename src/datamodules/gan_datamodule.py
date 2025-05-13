import os
import pandas as pd
import random
import numpy as np
import torch
import lightning as L
from torch.utils.data import DataLoader
from src.datamodules.datasets.gan_dataset import ImageDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

def seed_worker(worker_id):
    """Asegura reproducibilidad en la inicialización del worker"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
class ImageDataModule(L.LightningDataModule):
    def __init__(
        self, 
        batch_size=2, 
        workers=4, 
        train_csv=None, 
        val_csv=None, 
        test_csv=None, 
        data_dir_low=None, 
        data_dir_high=None,
        random_seed=2024
    ):
        """
        Args:
            batch_size (int): Tamaño del batch.
            workers (int): Número de workers para DataLoader.
            train_csv (str): Ruta al CSV del conjunto de entrenamiento.
            val_csv (str): Ruta al CSV del conjunto de validación.
            test_csv (str): Ruta al CSV del conjunto de prueba.
            data_dir (str): Ruta base donde están las imágenes.
            random_seed (int): Semilla para reproducibilidad.
        """
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.data_dir_low = data_dir_low
        self.data_dir_high = data_dir_high
        self.random_seed = random_seed

    def prepare_data(self):
        """Descargar o procesar datos si es necesario (aquí no se necesita)"""
        pass

    def setup(self, stage=None):
        """Inicializa los datasets según la fase"""
        normalize_transform = A.Normalize(mean=[0.5], std=[0.5])

        if stage == "fit" or stage is None:
            train_transform = A.Compose([
                A.Resize(256, 256),  # Redimensionar a 256x256
                # A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1),
                # A.HorizontalFlip(p=0.5),
                # A.RandomBrightnessContrast(p=0.2, brightness_limit=0.1, contrast_limit=0.1),
                # A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                # A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.3),
                normalize_transform,  # Normalización de imágenes
                ToTensorV2()
            ], additional_targets={"image1": "image"})  # Para transformar ambas imágenes

            val_transform = A.Compose([
                A.Resize(256, 256),
                normalize_transform,
                ToTensorV2()
            ], additional_targets={"image1": "image"})

            self.train_dataset = ImageDataset(self.train_csv, data_dir_low=self.data_dir_low,data_dir_high=self.data_dir_high, transform=train_transform)
            self.val_dataset = ImageDataset(self.val_csv,data_dir_low=self.data_dir_low,data_dir_high=self.data_dir_high, transform=val_transform)

        if stage == "test" or stage is None:
            test_transform = A.Compose([
                A.Resize(256, 256),
                normalize_transform,
                ToTensorV2()
            ], additional_targets={"image1": "image"})

            self.test_dataset = ImageDataset(self.test_csv, data_dir_low=self.data_dir_low,data_dir_high=self.data_dir_high, transform=test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.random_seed),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.random_seed),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.random_seed),
        )
