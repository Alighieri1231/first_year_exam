import sys

import torch
from lightning.pytorch import LightningModule
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch.cli import LightningCLI


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        return self(batch).sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=2)


sys.argv = ["bug_report_model.py", "--config", "configs/unet.yaml", "fit"]
cli = LightningCLI(BoringModel)
