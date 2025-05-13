import os
import pandas as pd
import numpy as np
import lightning as L
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, csv_file, data_dir_low=None,data_dir_high=None, transform=None):
        """
        Args:
            csv_file (str): Ruta al archivo CSV con los nombres de las imágenes.
            root_dir (str): Directorio raíz donde se encuentran las imágenes.
            transform (callable, optional): Transformaciones de Albumentations a aplicar.
        """
        self.data = pd.read_csv(csv_file)
        self.data_dir_low = data_dir_low
        self.data_dir_high = data_dir_high
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        low_quality_path = os.path.join(self.data_dir_low, self.data.iloc[idx, 0])  # Primera columna del CSV
        high_quality_path = os.path.join(self.data_dir_high, self.data.iloc[idx, 1])  # Segunda columna del CSV

        # Cargar imágenes en escala de grises
        low_quality_image = np.array(Image.open(low_quality_path).convert("L"))
        high_quality_image = np.array(Image.open(high_quality_path).convert("L"))

        # Aplicar transformaciones de Albumentations
        if self.transform:
            augmented = self.transform(image=low_quality_image, image1=high_quality_image)
            low_quality_image = augmented["image"]
            high_quality_image = augmented["image1"]

        return {"x": low_quality_image, "y": high_quality_image}