from src.datamodules.datasets.dataset_seg import WSIDataset
import lightning as L
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import torch
import numpy as np

import torch


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class WSIDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size=2,
        workers=1,
        train_file=None,
        dev_file=None,
        test_file=None,
        data_dir=None,
        cache_data=False,
        random_seed=2024,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.data_dir = data_dir
        self.cache_data = cache_data
        self.random_seed = random_seed

    def prepare_data(self):
        # Download / locate your data
        pass

    def setup(self, stage: str):
        if stage == "fit":
            train_transform = A.Compose(
                [
                    # A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1),
                    # A.HorizontalFlip(p=0.5),
                    # A.RandomBrightnessContrast(
                    #     p=0.2,
                    #     brightness_limit=0.1,
                    #     contrast_limit=0.1,
                    # ),
                    # A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                    # A.Affine(
                    #     scale=(0.9, 1.1),
                    #     translate_percent=(-0.05, 0.05),
                    #     rotate=(
                    #         -10,
                    #         10,
                    #     ),  # Puedes poner (-10, 10) por ejemplo si quieres rotaciones suaves
                    #     p=0.3,
                    # ),
                    ToTensorV2(),
                ]
            )
            # print transforms on console

            val_transform = A.Compose([ToTensorV2()])

            self.train_dataset = WSIDataset(
                meta_data=self.train_file,
                root_dir=self.data_dir,
                transform=train_transform,
            )

            self.dev_dataset = WSIDataset(
                meta_data=self.dev_file,
                root_dir=self.data_dir,
                transform=val_transform,
            )

        if stage == "test":
            test_transform = A.Compose([ToTensorV2()])

            self.test_dataset = WSIDataset(
                meta_data=self.test_file,
                root_dir=self.data_dir,
                transform=test_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            # worker_init_fn=seed_worker,
            # generator=torch.Generator().manual_seed(self.random_seed),
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            # worker_init_fn=seed_worker,
            # generator=torch.Generator().manual_seed(self.random_seed),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            # worker_init_fn=seed_worker,
            # generator=torch.Generator().manual_seed(self.random_seed),
        )
