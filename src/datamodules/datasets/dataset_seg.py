import os
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset


class WSIDataset(Dataset):
    def __init__(self, meta_data, root_dir, transform=None):
        # 1) Cargo el CSV con los nombres base
        df = pd.read_csv(meta_data)
        filenames = df["file_name"].tolist()

        # 2) Construyo las rutas a imágenes y máscaras
        self.frame_paths = [os.path.join(root_dir, "gt", fname) for fname in filenames]
        self.label_paths = [
            # inserta "_mask" antes de la extensión .png
            os.path.join(root_dir, "label", os.path.splitext(fname)[0] + "_mask.png")
            for fname in filenames
        ]

        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame = cv2.imread(self.frame_paths[idx], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)

        # if file name has malignant then category is 1 else 0
        category = 1 if "malignant" in self.frame_paths[idx] else 0

        frame = frame.astype(np.float32) / 255.0
        label = (label == 255).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=frame, mask=label)
            frame, label = augmented["image"], augmented["mask"]

        return {"image": frame, "mask": label, "category": category}
