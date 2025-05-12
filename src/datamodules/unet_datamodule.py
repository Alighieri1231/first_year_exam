import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchvision.transforms import functional as F
import torchio as tio


def numpy_reader(path):
    data = np.expand_dims(np.load(path), axis=0)  # Ensure channel dimension
    return data, np.eye(4)


class SegmentationDataset(Dataset):
    def __init__(self, file_paths, data_path, transform=None):
        """
        Args:
            file_paths (list of str): List of .npz file names.
            data_path (str): Path to the directory containing the .npz files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_paths = file_paths
        self.data_path = Path(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        base_name = Path(self.file_paths[idx]).stem  # Strip .npz extension
        video_path = self.data_path / f"{base_name}_gt.npy"
        label_path = self.data_path / f"{base_name}_label.npy"

        subject = tio.Subject(
            video_gt=tio.ScalarImage(video_path, reader=numpy_reader),
            label=tio.LabelMap(label_path, reader=numpy_reader),
        )

        # apply compose transform made by resize and rescaleintensity
        resize_transform = tio.Compose(
            [
                # tio.transforms.Resize((128, 128, 128)),
                # remap all the labels different from 0 to 1
                tio.transforms.RemapLabels(
                    {2: 1, 3: 1, 4: 1, 5: 1, 6: 1}, include=["label"]
                ),
                tio.transforms.RescaleIntensity((0, 1), include=["video_gt"]),
            ]
        )

        # # define the resize transform
        # resize_transform = tio.transforms.Resize(
        #     (128, 128, 128)
        # )  # New shape: [frames, width, height]

        # # Apply the resize transform to the subject
        subject = resize_transform(subject)

        # # # Extract resized video GT and label from the subject
        # video = resized_subject.video_gt.tensor  # Remove channel dimension
        # label = resized_subject.label.tensor.squeeze(
        #     0
        # ).long()  # Ensure integers for label

        # # # Normalize video values to [0, 1]
        # video = video.float() / 255.0

        # # # Apply transform if provided
        # if self.transform:
        #     video, label = self.transform((video, label))

        return subject


class SegmentationDataModule(L.LightningDataModule):
    def __init__(
        self, data_path, train_csv, val_csv, test_csv, batch_size=4, num_workers=4
    ):
        """
        Args:
            data_path (str): Path to the directory containing .npz files.
            train_csv (str): Path to the CSV file listing training .npz file names.
            val_csv (str): Path to the CSV file listing validation .npz file names.
            test_csv (str): Path to the CSV file listing testing .npz file names.
            batch_size (int): Batch size for dataloaders.
            num_workers (int): Number of workers for dataloaders.
        """
        super().__init__()
        self.data_path = data_path
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Load datasets based on CSV files.
        Args:
            stage (str, optional): Stage of the setup (fit, validate, test, predict).
        """
        train_files = pd.read_csv(self.train_csv)["file_name"].tolist()
        val_files = pd.read_csv(self.val_csv)["file_name"].tolist()
        test_files = pd.read_csv(self.test_csv)["file_name"].tolist()

        self.train_dataset = SegmentationDataset(train_files, self.data_path)
        self.val_dataset = SegmentationDataset(val_files, self.data_path)
        self.test_dataset = SegmentationDataset(test_files, self.data_path)

    # using subjectdataloaders from torchio
    def train_dataloader(self):
        return tio.data.SubjectsLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return tio.data.SubjectsLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return tio.data.SubjectsLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    # def train_dataloader(self):
    #     return DataLoader(
    #         self.train_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=True,
    #     )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #     )

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.test_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #     )
