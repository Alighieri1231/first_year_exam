import lightning as L
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torchmetrics import (
    MetricCollection,
    Accuracy,
    Precision,
    Recall,
    Dice,
    JaccardIndex,
)
from src.models.model import (
    UNet3D,
)  # Replace with the actual class name from your model file
from src.utils.ls import (
    get_loss_function,
)  # A utility to dynamically load loss functions
from torch.optim.lr_scheduler import StepLR  # Example of a scheduler
import gc


class UNetLightningModule(L.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        # self.save_hyperparameters(config)  # Save the config for reproducibility

        # Model configuration
        model_params = config.get("model", {})
        self.model = UNet3D(**model_params)

        # Loss function configuration
        self.loss_fn = get_loss_function(config.get("loss", "CrossEntropyLoss"))
        # self.loss_fn = criterion
        # Optimizer configuration
        self.optimizer_name = config.get("optimizer", "Adam")
        self.learning_rate = config.get("lr", 1e-3)  # Acceso directo a lr
        self.optimizer_params = {"lr": self.learning_rate}

        # Scheduler configuration
        self.step_size = config.get("step_size", 10)
        self.gamma = config.get("gamma", 0.1)
        self.scheduler_params = {"step_size": self.step_size, "gamma": self.gamma}
        # Metrics
        # num_classes = config.get("num_classes", 2)
        # metrics for segmentation
        # metrics = MetricCollection(
        #     {
        #         "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
        #         "precision": Precision(task="multiclass", num_classes=num_classes),
        #         "recall": Recall(task="multiclass", num_classes=num_classes),
        #         "dice": Dice(num_classes=num_classes),
        #         "iou": JaccardIndex(num_classes=num_classes, task="multiclass"),
        #     }
        # )

        # self.train_metrics = metrics.clone(prefix="train/")
        # self.val_metrics = metrics.clone(prefix="val/")
        # self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, step: str):
        # batch is subject from torchio made by video_gt and label
        inputs = batch["video_gt"]["data"]
        targets = batch["label"]["data"].squeeze(0).long()
        # inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        # preds = torch.argmax(outputs, dim=1)

        # metrics = getattr(self, f"{step}_metrics")
        # metrics.update(preds, targets)

        self.log(f"{step}/loss", loss)
        # self.log_dict(metrics, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        # Instantiate optimizer
        if self.optimizer_name == "Adam":
            optimizer = Adam(self.parameters(), **self.optimizer_params)
        elif self.optimizer_name == "SGD":
            optimizer = SGD(self.parameters(), **self.optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        # Optionally add a scheduler
        scheduler = StepLR(optimizer, **self.scheduler_params)
        return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        """Hook que se ejecuta al final de cada época de entrenamiento."""
        torch.cuda.empty_cache()
        gc.collect()
        self.log("info/memory_cleaned", 1)  # Log para verificar que se ejecutó
