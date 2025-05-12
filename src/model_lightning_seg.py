import lightning as L
import torch
from torch import nn
import utils
import torchmetrics
from models_arq.att_unet_seg import attention_unet  # attention_unet
from loss import (
    BCEDiceLoss,
    DiceLoss,
    FocalLoss,
    BCELogitsLoss,
    BCELoss,
    FocalLoss_weights,
    SigmoidFocalLoss,
    TopologicalPoolingLoss,
    CrossEntropyLoss,
    FocalLoss_multiclass,
    FocalLossb,
)
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
    BinaryAccuracy,
    BinaryF1Score,
    BinaryStatScores,
    Dice,
)
from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassStatScores,
)
from torchmetrics.detection import IntersectionOverUnion
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU


class MyModel(L.LightningModule):
    def __init__(self, model_opts, train_par):
        super().__init__()
        # Define your model architecture here. For example:
        model_def = globals()[model_opts.name]
        # self.model = model_def(input_channels=1)  # Assuming your model's __init__ doesn't take parameters
        self.model = model_def(
            input_channels=model_opts.args.inchannels,
            output_channels=model_opts.args.outchannels,
        )  # Assuming your model's __init__ doesn't take parameters

        self.train_par = train_par
        self.eval_threshold = train_par.eval_threshold
        self.num_classes = train_par.num_classes
        # Initialize metrics for train, dev, test
        self.init_metrics("train")
        self.init_metrics("val")
        self.init_metrics("test")
        # Define any metrics you want to use dice, tp, fp,tn, fn,precision,recall,accuracy,f1 for train,dev,test

    def init_metrics(self, stage):
        """
        Initializes metrics for a given stage.

        Args:
        - stage (str): The stage for which to initialize the metrics. Should be 'train', 'dev', or 'test'.
        """

        if self.num_classes == 2:
            metrics = {
                #'stat_scores': BinaryStatScores(threshold=self.eval_threshold),
                "IoU (lightning AI)": BinaryJaccardIndex(threshold=self.eval_threshold),
                "precision": BinaryPrecision(threshold=self.eval_threshold),
                "accuracy": BinaryAccuracy(threshold=self.eval_threshold),
                "f1": BinaryF1Score(threshold=self.eval_threshold),
            }
        else:
            metrics = {
                #'stat_scores':MulticlassStatScores(num_classes=self.num_classes),
                "IoU (lightning AI)": MulticlassJaccardIndex(
                    num_classes=self.num_classes, average="macro"
                ),
                "precision": MulticlassPrecision(
                    num_classes=self.num_classes, average="macro"
                ),
                "accuracy": MulticlassAccuracy(
                    num_classes=self.num_classes, average="macro"
                ),
                "f1": MulticlassF1Score(num_classes=self.num_classes, average="macro"),
                "f1_per_class": MulticlassF1Score(
                    num_classes=self.num_classes, average="none"
                ),
            }

        for metric_name, metric in metrics.items():
            setattr(self, f"{stage}_{metric_name}", metric)

    def log_metrics(self, stage, pred, label):
        """
        Logs metrics for a given stage.

        Args:
        - stage (str): Stage of the model ('train', 'dev', 'test').
        - pred (torch.Tensor): Predictions from the model.
        - mask (torch.Tensor): True labels.
        """

        metrics = [
            "IoU (lightning AI)",
            "precision",
            "accuracy",
            "f1",
        ]
        for metric in metrics:
            metric_fn = getattr(self, f"{stage}_{metric}")
            metric_fn(pred, label)
            self.log(
                f"{stage}_{metric}",
                metric_fn,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        if self.num_classes != 2:
            # Log F1-score for each class
            f1_per_class_fn = getattr(self, f"{stage}_f1_per_class")
            f1_scores = f1_per_class_fn(pred, label)
            for class_idx, f1_score in enumerate(f1_scores):
                self.log(
                    f"{stage}_f1_class_{class_idx}",
                    f1_score,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        if self.num_classes == 2:
            dice_metric, _ = utils.dice_coeff_batch(pred.float(), label.float())
            self.log(
                f"{stage}_Dice (formula)",
                dice_metric,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def forward(self, x):
        # Implement the forward pass
        return self.model(x)

    def shared_step(self, batch, batch_idx, stage: str):
        """
        A shared step function for training, validation, and testing steps,
        adaptable for binary and multiclass cases.

        Args:
            batch: The input batch of data.
            batch_idx: The index of the batch.
            stage (str): One of "train", "val", or "test".
            num_classes (int): Number of classes (default: 2 for binary classification).
        """
        img, label = batch

        # Forward pass
        pred_mask = self(img)  # Use the forward method to predict

        if self.num_classes == 2:
            # Binary case
            loss = self.get_loss(pred_mask.float(), label.float())
            self.log(f"{stage}_loss", loss, sync_dist=True)

            # Predicted probabilities and binary thresholding
            pred_prob = torch.sigmoid(pred_mask)
            pred = (pred_prob >= self.eval_threshold).to(torch.int64)
        else:
            # Multiclass case

            loss = self.get_loss(
                pred_mask, label
            )  # pred_mask: logits, label: int64 classes
            self.log(f"{stage}_loss", loss, sync_dist=True)

            # Predicted class indices
            pred = torch.argmax(pred_mask, dim=1).to(torch.int64)

        # Log metrics (common for both cases)
        self.log_metrics(stage, pred, label)

        if stage == "train":
            # Additional training-specific logic if needed
            return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, batch_idx, stage="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # Implement the prediction step
        img, label = batch
        # img, label = img.to(self.device), label.to(self.device)

        # self.pos_weights = utils.pos_weight_batch(label)
        pred_mask = self(img)
        pred_prob = torch.sigmoid(pred_mask)  # Obtener probabilidad
        pred = (pred_prob >= self.eval_threshold).to(torch.float)

        return pred

    def predict_step_aux(self, batch, batch_idx, dataloader_idx=None):
        # Implement the prediction step
        img, label = batch

        # self.pos_weights = utils.pos_weight_batch(label)
        pred_mask = self(img)
        pred_prob = torch.sigmoid(pred_mask)  # Obtener probabilidad
        pred = (pred_prob >= self.eval_threshold).to(torch.float)
        label = (label >= self.eval_threshold).to(torch.float)

        return pred, pred_prob, pred_mask, label

    def configure_optimizers(self):
        # Configure optimizers and LR schedulers
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.train_par.lr)
        # Adamw
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_par.lr,
            weight_decay=self.train_par.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    # Implement the get_loss method here, adjust it based on your original Trainer class
    def get_loss(self, y_hat, y):
        if self.train_par.loss_opts.name != "default":
            self.loss_f = globals()[self.train_par.loss_opts.name]
        else:
            print(y_hat.shape, y.shape, y_hat.dtype, y.dtype)
            self.loss_f = nn.CrossEntropyLoss(y_hat, y)

        if self.train_par.loss_opts.name == "BCELogitsLoss":
            if self.train_par.loss_opts.args.weight == "default":
                return self.loss_f(y_hat, y)
            else:
                return self.loss_f(y_hat, y, weight=self.pos_weights)

        if self.train_par.loss_opts.name == "FocalLossb":
            return self.loss_f(
                y_hat,
                y,
                alpha=self.train_par.loss_opts.args.alpha,
                gamma=self.train_par.loss_opts.args.gamma,
                logits=False,
                reduce=True,
            )

        if self.train_par.loss_opts.name == "BCEDiceLoss":
            if self.train_par.loss_opts.args.weight == "default":
                return self.loss_f(y_hat, y, device=self.device)
            else:
                return self.loss_f(
                    y_hat,
                    y,
                    weight=self.train_par.loss_opts.args.weight,
                    device=self.device,
                )

        if self.train_par.loss_opts.name == "DiceLoss":
            return self.loss_f(y_hat, y)

        if self.train_par.loss_opts.name == "BCELoss":
            return self.loss_f(y_hat, y)

        if self.train_par.loss_opts.name == "FocalLoss_weights":
            return self.loss_f(y_hat, y)

        if self.train_par.loss_opts.name == "SigmoidFocalLoss":
            return self.loss_f(y_hat, y)

        if self.train_par.loss_opts.name == "TopologicalPoolingLoss":
            return self.loss_f(y_hat, y)

        if self.train_par.loss_opts.name == "TopologicalPoolingLoss_function":
            return self.loss_f(y_hat, y)
        # cross entropy
        if self.train_par.loss_opts.name == "CrossEntropyLoss":
            return self.loss_f(y_hat, y)
        # FocalLoss_multiclass
        if self.train_par.loss_opts.name == "FocalLoss_multiclass":
            return self.loss_f(
                y_hat,
                y,
                alpha=self.train_par.loss_opts.args.alpha,
                gamma=self.train_par.loss_opts.args.gamma,
            )
        # FocalLoss como clase
        if self.train_par.loss_opts.name == "FocalLoss":
            if not hasattr(self, "_focal_loss"):
                self._focal_loss = FocalLoss(
                    gamma=self.train_par.loss_opts.args.gamma,
                    alpha=self.train_par.loss_opts.args.alpha,
                )
            return self._focal_loss(y_hat, y)
