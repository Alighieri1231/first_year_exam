import os
import numpy as np
import pandas as pd
import torch
import glob
from torch.utils.data import Dataset
import skvideo.io
import cv2
import torchio as tio


class WSIDataset(Dataset):
    def __init__(self, meta_data, root_dir, transform=None):
        df = pd.read_csv(meta_data)

        self.frame_paths = []
        self.label_paths = []

        for i in range(len(df)):
            sweep_name = df.file_name[i].split("_gt")[0]
            sweep_frames = sorted(glob.glob(f"{root_dir}/gt/{sweep_name}_*.png"))
            sweep_labels = sorted(glob.glob(f"{root_dir}/label/{sweep_name}_*.png"))

            self.frame_paths.extend(sweep_frames)
            self.label_paths.extend(sweep_labels)

        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        label_path = self.label_paths[idx]

        frame = cv2.imread(frame_path,cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        frame = frame.astype(np.float32) / 255.0
        label = (label == 255).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=frame, mask=label)
            frame = augmented["image"]
            label = augmented["mask"]

        # return in format batch['image'], batch['mask']
        return {"image": frame, "mask": label}
