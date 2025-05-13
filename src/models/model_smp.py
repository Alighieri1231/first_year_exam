import lightning as L
import torch
from torch import nn
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import os
import cv2
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb
import numpy as np


class USModel(L.LightningModule):
    def __init__(self, model_opts, train_par, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            model_opts.args.arch,
            encoder_name=model_opts.args.encoder_name,
            in_channels=model_opts.args.inchannels,
            classes=model_opts.args.outchannels,
            encoder_weights=model_opts.args.encoder_weights,
            **kwargs,
        )
        self.lr = train_par.lr
        self.weight_decay = train_par.weight_decay
        self.optimizer_name = train_par.optimizer
        self.scheduler_name = train_par.scheduler
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(model_opts.args.encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        # self.register_buffer("std", torch.tensor(params["std"][0]).view(1, 1, 1, 1))
        # self.register_buffer("mean", torch.tensor(params["mean"][0]).view(1, 1, 1, 1))

        # for image segmentation dice loss could be the best first choice
        # self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True) a 0.7 b=0.3
        self.loss_fn = smp.losses.TverskyLoss(
            smp.losses.BINARY_MODE, from_logits=True, alpha=0.5, beta=0.5
        )

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4
        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]
        # mask is (B,H,W) expand to (B, 1, H, W)
        mask = mask.unsqueeze(1)
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def save_test_overlays(self, data_module, results_path, name):
        # Crear carpeta para guardar overlays
        overlay_path = os.path.join(results_path, name)
        os.makedirs(overlay_path, exist_ok=True)

        test_loader = data_module.test_dataloader()

        self.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                image = batch["image"].to(self.device)
                mask = batch["mask"].to(self.device)

                logits_mask = self.forward(image)
                prob_mask = logits_mask.sigmoid()
                pred_mask = (prob_mask > 0.5).float()

                for j in range(image.shape[0]):
                    img = (
                        image[j].cpu().numpy().transpose(1, 2, 0)
                    )  # (C, H, W) -> (H, W, C)
                    img = (
                        img * self.std.cpu().numpy().squeeze()
                        + self.mean.cpu().numpy().squeeze()
                    ) * 255
                    img = np.clip(img, 0, 255).astype(np.uint8)

                    gt_mask = mask[j].cpu().numpy().squeeze()
                    pred_mask_np = pred_mask[j].cpu().numpy().squeeze()

                    if gt_mask.sum() == 0:
                        continue  # Omitir si ground truth es solo background

                    # Crear overlay con Ground Truth (verde)
                    gt_overlay = img.copy()
                    gt_overlay[gt_mask > 0] = np.clip(
                        gt_overlay[gt_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5,
                        0,
                        255,
                    )

                    # Crear overlay con Predicción (rojo)
                    pred_overlay = img.copy()
                    pred_overlay[pred_mask_np > 0] = np.clip(
                        pred_overlay[pred_mask_np > 0] * 0.5
                        + np.array([255, 0, 0]) * 0.5,
                        0,
                        255,
                    )

                    # Concatenar las imágenes (3 columnas)
                    combined_image = np.concatenate(
                        (img, gt_overlay, pred_overlay), axis=1
                    )

                    # Guardar imagen
                    save_path = os.path.join(overlay_path, f"test_overlay_{i}_{j}.png")
                    cv2.imwrite(
                        save_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
                    )

        print(f"Overlays guardados en: {overlay_path}")

    def log_test_images(self, data_module, threshold=0.4, val_iou=None,only_roi_frames=False):
        if val_iou is None or val_iou <= threshold:
            print(f"No se loggearán imágenes porque val_iou ({val_iou}) ≤ {threshold}")
            return  # No hacer nada si val_iou es menor o igual a 0.4

        self.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        test_loader = data_module.test_dataloader()
        wandb_images = []

        with torch.no_grad():
            for batch in test_loader:
                images, masks = batch["image"].to(device), batch["mask"].to(device)
                logits_mask = self.forward(images)
                preds = torch.sigmoid(logits_mask) > 0.5

                for i in range(images.shape[0]):
                    mask_np = masks[i].cpu().numpy()

                    if only_roi_frames:
                        if np.max(mask_np) == 0:
                            continue

                    # Extraer la imagen y convertirla al formato adecuado
                    img_np = (
                        images[i].cpu().numpy().transpose(1, 2, 0)
                    )  # Convertir (C, H, W) -> (H, W, C)

                    # Verificar si es imagen en escala de grises (1 canal)
                    if img_np.shape[-1] == 1:
                        img_np = img_np.squeeze(-1)  # Convertir (H, W, 1) a (H, W)

                    # Ajustar rango de valores si están entre -1 y 1
                    if img_np.min() < 0:
                        img_np = (img_np + 1) / 2.0  # Rango de [-1,1] a [0,1]

                    # Convertir al rango de 0 a 255
                    img_np = (img_np * 255).astype(np.uint8)

                    # Obtener la predicción
                    pred_np = preds[i].cpu().numpy().squeeze()

                    # Crear figura con 3 imágenes: original, máscara real y predicción
                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                    axs[0].imshow(img_np, cmap="gray")
                    axs[0].set_title("Imagen Original")
                    axs[1].imshow(mask_np, cmap="gray")
                    axs[1].set_title("Máscara Real")
                    axs[2].imshow(pred_np, cmap="gray")
                    axs[2].set_title("Predicción")

                    plt.tight_layout()

                    # Guardar en wandb
                    wandb_images.append(
                        wandb.Image(fig, caption=f"Ejemplo {len(wandb_images) + 1}")
                    )
                    plt.close(fig)  # Cerrar la figura para evitar problemas de memoria

        if wandb_images:
            wandb.log({"Ejemplos de Segmentación": wandb_images})
        else:
            print("No se encontraron imágenes con máscara para loggear.")

    def configure_optimizers(self):
        # Seleccionar el optimizador basado en el hiperparámetro
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "adamw":
            optimizer = optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not recognized")

        # Seleccionar el scheduler basado en el hiperparámetro
        if self.scheduler_name == "cosine_annealing":
            T_MAX = (
                self.trainer.estimated_stepping_batches
            )  # Considera epochs * steps_per_epoch
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_MAX, eta_min=1e-5
            )
        elif self.scheduler_name == "step_lr":
            scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        else:
            raise ValueError(f"Scheduler {self.scheduler_name} not recognized")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
