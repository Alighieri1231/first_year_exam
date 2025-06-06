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
from src.utils.loss import FocalTverskyLoss
import torch.nn.functional as F
from src.utils.loss import BCEWithLogitsLoss2d


class USModel(L.LightningModule):
    def __init__(self, model_opts, train_par, **kwargs):
        super().__init__()

        aux_params = dict(
            pooling="avg",  # GlobalAveragePooling
            dropout=0.5,  # opcional
            activation=None,  # logits de cls
            classes=2,  # # categorías
        )
        self.model = smp.create_model(
            model_opts.args.arch,
            encoder_name=model_opts.args.encoder_name,
            in_channels=model_opts.args.inchannels,
            classes=model_opts.args.outchannels,
            encoder_weights=model_opts.args.encoder_weights,
            aux_params=aux_params,
            **kwargs,
        )
        self.lr = train_par.lr
        self.weight_decay = train_par.weight_decay
        self.optimizer_name = train_par.optimizer
        self.loss = train_par.loss_opts.name
        self.classification_loss = model_opts.args.classification_loss
        self.gamma_mom = train_par.loss_opts.args.gamma_mom
        self.gamma_round = train_par.loss_opts.args.gamma_round
        self.gamma_class = train_par.loss_opts.args.gamma_class
        self.gamma_close = train_par.loss_opts.args.gamma_close
        self.gamma_topo = train_par.loss_opts.args.gamma_topo

        self.binary_split = model_opts.args.binary_split
        # if self.optimizer is a dict with key 'name':'adamw', convert it to a string only adamw
        if isinstance(self.optimizer_name, dict):
            self.optimizer_name = self.optimizer_name["name"]
        self.scheduler_name = train_par.scheduler
        if isinstance(self.scheduler_name, dict):
            self.scheduler_name = self.scheduler_name["name"]
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(model_opts.args.encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        # self.register_buffer("std", torch.tensor(params["std"][0]).view(1, 1, 1, 1))
        # self.register_buffer("mean", torch.tensor(params["mean"][0]).view(1, 1, 1, 1))

        # for image segmentation dice loss could be the best first choice
        # self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True) a 0.7 b=0.3
        if self.loss == "dice":
            self.loss_fn = smp.losses.DiceLoss(
                smp.losses.BINARY_MODE, from_logits=True, smooth=1e-6
            )
        elif self.loss == "focal":
            self.loss_fn = smp.losses.FocalLoss(
                smp.losses.BINARY_MODE, from_logits=True, alpha=0.25, gamma=2.0
            )
        elif self.loss == "tversky":
            self.loss_fn = smp.losses.TverskyLoss(
                smp.losses.BINARY_MODE,
                from_logits=True,
                alpha=train_par.loss_opts.args.alpha,
                beta=train_par.loss_opts.args.beta,
            )
        elif self.loss == "focal_tversky":
            self.loss_fn = FocalTverskyLoss(
                smp.losses.BINARY_MODE,
                from_logits=True,
                alpha=train_par.loss_opts.args.alpha,
                beta=train_par.loss_opts.args.beta,
                gamma=train_par.loss_opts.args.gamma,
            )
        elif self.loss == "bce":
            self.loss_fn = BCEWithLogitsLoss2d(ignore_label=255)

        self.clas_loss_fn = nn.CrossEntropyLoss()

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask, pred_cat = self.model(image)
        return mask, pred_cat

    def shared_step(self, batch, stage):
        image = batch["image"]
        mask = batch["mask"].unsqueeze(1)
        gt_class = batch["category"].to(self.device)  # 0=benigno, 1=maligno

        logits_mask, logits_clas = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        if self.classification_loss:
            loss += self._add_classification_losses(logits_clas, gt_class, logits_mask)

        # probabilidades y predicción binaria
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # Si no pedimos split, devolvemos solo stats globales
        if not self.binary_split:
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask.long(), mask.long(), mode="binary"
            )
            out = {
                "loss": loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
            if self.classification_loss:
                acc_clas = (logits_clas.argmax(1) == gt_class).float().mean()
                self.log(f"{stage}_acc_clas", acc_clas, prog_bar=True, sync_dist=True)
            return out

        # ——————————————————————————————————————————————
        # Si self.binary_split=True: calculamos stats por clase
        # ——————————————————————————————————————————————
        # stats globales
        else:
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask.long(), mask.long(), mode="binary"
            )

            # máscara de índices
            benign_idx = gt_class == 0
            malignant_idx = gt_class == 1
            zeros = torch.tensor(0, device=self.device)

            # stats benignos
            if benign_idx.any():
                tp_b, fp_b, fn_b, tn_b = smp.metrics.get_stats(
                    pred_mask[benign_idx].long(),
                    mask[benign_idx].long(),
                    mode="binary",
                )
            else:
                tp_b = fp_b = fn_b = tn_b = zeros

            # stats malignos
            if malignant_idx.any():
                tp_m, fp_m, fn_m, tn_m = smp.metrics.get_stats(
                    pred_mask[malignant_idx].long(),
                    mask[malignant_idx].long(),
                    mode="binary",
                )
            else:
                tp_m = fp_m = fn_m = tn_m = zeros

            out = {
                "loss": loss,
                # globales
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                # benignos
                "tp_b": tp_b,
                "fp_b": fp_b,
                "fn_b": fn_b,
                "tn_b": tn_b,
                # malignos
                "tp_m": tp_m,
                "fp_m": fp_m,
                "fn_m": fn_m,
                "tn_m": tn_m,
            }
            if self.classification_loss:
                acc_clas = (logits_clas.argmax(1) == gt_class).float().mean()
                self.log(f"{stage}_acc_clas", acc_clas, prog_bar=True, sync_dist=True)
            return out

    def shared_epoch_end(self, outputs, stage):
        # Si no pedimos split, solo usamos las stats globales
        if not self.binary_split:
            tp = torch.cat([x["tp"] for x in outputs])
            fp = torch.cat([x["fp"] for x in outputs])
            fn = torch.cat([x["fn"] for x in outputs])
            tn = torch.cat([x["tn"] for x in outputs])

            per_image_iou = smp.metrics.iou_score(
                tp, fp, fn, tn, reduction="micro-imagewise"
            )
            dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            per_image_f1 = smp.metrics.f1_score(
                tp, fp, fn, tn, reduction="micro-imagewise"
            )
            dataset_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            per_image_acc = smp.metrics.accuracy(
                tp, fp, fn, tn, reduction="micro-imagewise"
            )
            dataset_acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

            metrics = {
                f"{stage}_per_image_iou": per_image_iou,
                f"{stage}_dataset_iou": dataset_iou,
                f"{stage}_per_image_f1": per_image_f1,
                f"{stage}_dataset_f1": dataset_f1,
                f"{stage}_per_image_acc": per_image_acc,
                f"{stage}_dataset_acc": dataset_acc,
            }
            self.log_dict(metrics, prog_bar=True, sync_dist=True)
            return

        # ——————————————————————————————————————————————
        # Si self.binary_split=True: agregamos global + por clase
        # ——————————————————————————————————————————————
        # empaquetar stats globales
        else:
            tp = torch.cat([x["tp"] for x in outputs])
            fp = torch.cat([x["fp"] for x in outputs])
            fn = torch.cat([x["fn"] for x in outputs])
            tn = torch.cat([x["tn"] for x in outputs])
            # benignos
            tp_b = torch.cat([x["tp_b"] for x in outputs])
            fp_b = torch.cat([x["fp_b"] for x in outputs])
            fn_b = torch.cat([x["fn_b"] for x in outputs])
            tn_b = torch.cat([x["tn_b"] for x in outputs])
            # malignos
            tp_m = torch.cat([x["tp_m"] for x in outputs])
            fp_m = torch.cat([x["fp_m"] for x in outputs])
            fn_m = torch.cat([x["fn_m"] for x in outputs])
            tn_m = torch.cat([x["tn_m"] for x in outputs])

            # funciones helper para no repetir
            def _make_metrics(tp, fp, fn, tn, suffix=""):
                return {
                    f"{stage}_per_image_iou{suffix}": smp.metrics.iou_score(
                        tp, fp, fn, tn, reduction="micro-imagewise"
                    ),
                    f"{stage}_dataset_iou{suffix}": smp.metrics.iou_score(
                        tp, fp, fn, tn, reduction="micro"
                    ),
                    f"{stage}_per_image_f1{suffix}": smp.metrics.f1_score(
                        tp, fp, fn, tn, reduction="micro-imagewise"
                    ),
                    f"{stage}_dataset_f1{suffix}": smp.metrics.f1_score(
                        tp, fp, fn, tn, reduction="micro"
                    ),
                    f"{stage}_per_image_acc{suffix}": smp.metrics.accuracy(
                        tp, fp, fn, tn, reduction="micro-imagewise"
                    ),
                    f"{stage}_dataset_acc{suffix}": smp.metrics.accuracy(
                        tp, fp, fn, tn, reduction="micro"
                    ),
                }

            metrics = {}
            # globales
            metrics.update(_make_metrics(tp, fp, fn, tn))
            # benignos
            metrics.update(_make_metrics(tp_b, fp_b, fn_b, tn_b, suffix="_benign"))
            # malignos
            metrics.update(_make_metrics(tp_m, fp_m, fn_m, tn_m, suffix="_malignant"))

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
        # log the images
        self.log_test_images(
            data_module=self.trainer.datamodule,
            num_images=50,
            threshold=0.1,
            only_roi_frames=True,
        )
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

                logits_mask, _ = self.forward(image)
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

    def log_test_images(
        self,
        data_module,
        threshold=0.4,
        only_roi_frames=False,
        num_images=10,
    ):
        self.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        test_loader = data_module.test_dataloader()
        # Limitar el número de imágenes a loggear
        test_loader = torch.utils.data.Subset(
            test_loader.dataset,
            np.random.choice(
                range(len(test_loader.dataset)),
                size=min(num_images, len(test_loader.dataset)),
                replace=False,
            ),
        )
        test_loader = torch.utils.data.DataLoader(
            test_loader,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        wandb_images = []

        with torch.no_grad():
            for batch in test_loader:
                images, masks = batch["image"].to(device), batch["mask"].to(device)
                logits_mask, _ = self.forward(images)
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
                    axs[0].set_title("Image")
                    axs[1].imshow(mask_np, cmap="gray")
                    axs[1].set_title("GT")
                    axs[2].imshow(pred_np, cmap="gray")
                    axs[2].set_title("Pred")

                    plt.tight_layout()

                    # Guardar en wandb
                    wandb_images.append(
                        wandb.Image(fig, caption=f"Ejemplo {len(wandb_images) + 1}")
                    )
                    plt.close(fig)  # Cerrar la figura para evitar problemas de memoria

        # if wandb_images:
        #    self.log({"Ejemplos de Segmentación": wandb_images})
        if wandb_images:
            # key: nombre de la tabla en W&B
            # images: lista de tensores/arrays/PIL o wandb.Image
            self.logger.log_image(
                key="Ejemplos de Segmentación",
                images=wandb_images,
            )

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

    def _moment_loss(self, mask_prob, gt_class):
        """
        Penaliza la excentricidad e = (μ20 - μ02)/(μ20 + μ02).
        Queremos e≈0 en benignos, y menos restricción en malignos.
        mask_prob: (B,1,H,W) probabilidades
        gt_class: (B,) 0=benigno,1=maligno
        """
        B, _, H, W = mask_prob.shape
        # coordenadas
        ii = torch.arange(H, device=mask_prob.device).view(1, H, 1).float()
        jj = torch.arange(W, device=mask_prob.device).view(1, 1, W).float()
        losses = []
        for b in range(B):
            m = mask_prob[b, 0]
            A = m.sum() + 1e-6
            i_bar = (ii * m).sum() / A
            j_bar = (jj * m).sum() / A
            mu20 = (((ii - i_bar) ** 2) * m).sum()
            mu02 = (((jj - j_bar) ** 2) * m).sum()
            e = (mu20 - mu02) / (mu20 + mu02 + 1e-6)
            if gt_class[b] == 0:  # benigno → penalizar e≈0
                losses.append(e.pow(2))
            else:  # maligno → menos restricción (o alentar |e|>c)
                losses.append(torch.zeros_like(e))
        return torch.stack(losses).mean()

    def _roundness_loss(self, mask_prob, gt_class):
        """
        Penaliza la compacidad C = 4πA/P^2.
        Target C=1 en benigno, C=c en maligno (e.g. 0.7).
        """
        B, _, H, W = mask_prob.shape
        losses = []
        c_maligno = 0.3  # objetivo de redondez para malignos
        for b in range(B):
            m = mask_prob[b, 0]
            A = m.sum() + 1e-6
            # perímetro aproximado
            dx = torch.abs(m[:, 1:] - m[:, :-1]).sum()
            dy = torch.abs(m[1:, :] - m[:-1, :]).sum()
            P = dx + dy + 1e-6
            C = 4 * np.pi * A / (P**2)
            target = 1.0 if gt_class[b] == 0 else c_maligno
            losses.append((C - target).pow(2))
        return torch.stack(losses).mean()

    def _closing_loss(self, mask_prob, kernel_size=7):
        """
        mask_prob: (B,1,H,W) con probabilidades en [0,1]
        kernel_size: tamaño de vecindad para cerrar huecos
        """
        pad = kernel_size // 2
        # 1) dilatación suave: max‐pooling
        dil = F.max_pool2d(mask_prob, kernel_size, stride=1, padding=pad)
        # 2) erosión suave: negativa de max‐pool sobre negativo
        ero = -F.max_pool2d(-dil, kernel_size, stride=1, padding=pad)
        closed = ero
        # 3) solo penalizamos cuando closed > original
        diff = closed - mask_prob
        return torch.clamp(diff, min=0).pow(2).mean()
