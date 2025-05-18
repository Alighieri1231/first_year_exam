import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score, accuracy
from src.models.modules.gan_modules import Discriminator, FCDiscriminator
import numpy as np
import os
import cv2
import wandb
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import PolynomialLR

# ──────── LOSS DEFINITION ────────
# Option A: your original 2d‐masks + wrappers
from src.utils.loss import CrossEntropy2d, BCEWithLogitsLoss2d

# Option B: pure PyTorch (commented out)
# from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


class ASSGAN(L.LightningModule):
    def __init__(self, model_opts, train_par):
        super().__init__()
        # manual optimization so we can step G and D separately
        self.automatic_optimization = False

        aux_params = dict(
            pooling="avg",  # GlobalAveragePooling
            dropout=0.5,  # opcional
            activation=None,  # logits de cls
            classes=2,  # # categorías
        )

        # — Generators (each outputs 1‐channel logits) —
        self.generator1 = smp.create_model(
            model_opts.args.arch1,
            encoder_name=model_opts.args.encoder_name1,
            in_channels=3,
            classes=1,
            encoder_weights="imagenet",
            aux_params=aux_params,
        )
        self.generator2 = smp.create_model(
            model_opts.args.arch2,
            encoder_name=model_opts.args.encoder_name2,
            in_channels=3,
            classes=1,
            encoder_weights="imagenet",
            aux_params=aux_params,
        )

        # Preprocessors for each generator
        params1 = smp.encoders.get_preprocessing_params(model_opts.args.encoder_name1)
        params2 = smp.encoders.get_preprocessing_params(model_opts.args.encoder_name2)

        # register buffer for each generator like
        # self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        # self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.register_buffer("std1", torch.tensor(params1["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean1", torch.tensor(params1["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std2", torch.tensor(params2["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean2", torch.tensor(params2["mean"]).view(1, 3, 1, 1))
        # — Discriminator (takes 1‐channel probability maps) —
        self.discriminator = FCDiscriminator(num_classes=1)
        self.classification_loss = model_opts.args.classification_loss

        # ─── Segmentation loss ───
        # Option A: custom 2D BCE wrapper (handles ignore internally)
        self.seg_loss = BCEWithLogitsLoss2d(ignore_label=255)
        # Option B: pure PyTorch (commented)
        # self.seg_loss = BCEWithLogitsLoss(reduction="mean")

        # ─── Adversarial loss ───
        # Option A: custom 2D BCE wrapper
        self.adv_loss = BCEWithLogitsLoss2d(ignore_label=255)
        # Option B: pure PyTorch (commented)
        # self.adv_loss = BCEWithLogitsLoss(reduction="mean")

        # ─── Classification loss ───
        # Option A: custom 2D CrossEntropy wrapper (handles ignore internally)
        self.clas_loss_fn = nn.CrossEntropyLoss()

        # ─── Hyperparameters ───
        self.lr_g = train_par.lr_g
        self.lr_d = train_par.lr_d
        self.weight_decay = train_par.weight_decay
        self.adam_betas = (0.9, 0.9)
        self.poly_power = train_par.polynomial_power
        self.lambda_seg = train_par.lambda_seg
        self.lambda_adv = train_par.lambda_adv
        self.lambda_adv_u = train_par.lambda_adv_u  # weight for unlabeled adv loss
        self.lambda_semi = train_par.lambda_semi  # weight for pseudo‐loss
        self.lambda_clas = train_par.lambda_clas
        self.lambda_clas_u = train_par.lambda_clas_u  # weight for unlabeled cls loss
        self.supervised_epochs = train_par.supervised_epochs  # e.g. 200
        self.gamma_thresh = train_par.gamma_thresh  # e.g. 0.2
        self.adam_g = train_par.adam_g

        # buffer for pulling unlabeled data
        self.unlab_iter = None

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        raise NotImplementedError("Use training_step to control G/D separately")

    def processg1(self, x):
        # Preprocess for generator 1
        x = (x - self.mean1) / self.std1
        return x

    def processg2(self, x):
        # Preprocess for generator 2
        x = (x - self.mean2) / self.std2
        return x

    def training_step(self, batch, batch_idx):
        epoch = self.current_epoch
        imgs_l = batch["image"]  # (B,3,H,W)
        masks_l = batch["mask"].unsqueeze(1)  # (B,1,H,W), values {0,1}
        gt_class = batch["category"].to(self.device)  # (B,)

        # print(f"imgs_l.shape: {imgs_l.shape}, masks_l.shape: {masks_l.shape}")

        opt_g, opt_d = self.optimizers()

        # ─── GENERATOR STEP ───
        self.toggle_optimizer(opt_g)

        # Supervised training

        # 1) forward
        pred1, logits_clas1 = self.generator1(
            self.processg1(imgs_l)
        )  # (B,1,H,W) logits
        pred2, logits_clas2 = self.generator2(self.processg2(imgs_l))

        # 2) segmentation loss on labeled
        loss_seg1 = self.seg_loss(pred1, masks_l)
        loss_seg2 = self.seg_loss(pred2, masks_l)

        loss_clas1 = loss_clas2 = 0.0

        if self.classification_loss:
            loss_clas1 = self.clas_loss_fn(logits_clas1, gt_class)
            loss_clas2 = self.clas_loss_fn(logits_clas2, gt_class)

        # 3) adversarial “fake” loss  → we want D(sigmoid(pred)) ≈ 1
        prob1 = torch.sigmoid(pred1)
        prob2 = torch.sigmoid(pred2)

        score1 = self.discriminator(prob1)  # (B,1,H,W)
        score2 = self.discriminator(prob2)
        adv1_l = self.adv_loss(score1, torch.ones_like(score1))
        adv2_l = self.adv_loss(score2, torch.ones_like(score2))

        # optional: unlabeled adversarial
        adv1_u = adv2_u = 0.0
        semi1 = semi2 = 0.0
        loss_clas1_u = loss_clas2_u = 0.0
        # — after supervised_epochs, pull unlabeled for unsupervised & pseudo —
        if epoch >= self.supervised_epochs and hasattr(
            self.trainer.datamodule, "unlabeled_dataloader"
        ):
            # change lambda_adv to lambda_adv_u
            self.lambda_adv = self.lambda_adv_u
            # get a batch of unlabeled images
            if self.unlab_iter is None:
                self.unlab_iter = iter(self.trainer.datamodule.unlabeled_dataloader())
            try:
                unlab = next(self.unlab_iter)
            except StopIteration:
                self.unlab_iter = iter(self.trainer.datamodule.unlabeled_dataloader())
                unlab = next(self.unlab_iter)

            imgs_u = unlab["image"].to(self.device)  # (B,3,H,W)
            # forward G on unlabeled
            pred1_u, logits_clas1_u = self.generator1(self.processg1(imgs_u))
            pred2_u, logits_clas2_u = self.generator2(self.processg2(imgs_u))
            prob1_u = torch.sigmoid(pred1_u)
            prob2_u = torch.sigmoid(pred2_u)

            score1_u = self.discriminator(prob1_u)
            score2_u = self.discriminator(prob2_u)

            # 4) pseudo‐label mutual supervision (Alg 2)
            # discriminator’s confidence
            conf1 = torch.sigmoid(score1_u).detach()  # (B,1,H,W)
            conf2 = torch.sigmoid(score2_u).detach()  # (B,1,H,W)

            # masks of “trusted” pixels
            mask1 = (conf1 > self.gamma_thresh).float()
            mask2 = (conf2 > self.gamma_thresh).float()

            # — ADVERSARIAL UNLABELED CRUZADO —
            # G2 trata como “reales” solo los parches que D aprobó de G1
            target2_adv = torch.ones_like(score2_u)
            target2_adv[mask1 == 0] = 255  # 255 = ignore en BCEWithLogitsLoss2d
            adv2_u = self.lambda_adv_u * self.adv_loss(score2_u, target2_adv)

            # G1 trata como “reales” solo los parches que D aprobó de G2
            target1_adv = torch.ones_like(score1_u)
            target1_adv[mask2 == 0] = 255
            adv1_u = self.lambda_adv_u * self.adv_loss(score1_u, target1_adv)

            # — PSEUDO‐LABEL MUTUAL SUPERVISION (Alg. 2) CRUZADO —
            pseudo1 = (prob1_u > 0.5).float()  # pseudo‐GT de G1 (0/1) full-res
            pseudo2 = (prob2_u > 0.5).float()  # pseudo‐GT de G2

            # Target para G2: la pseudo‐máscara de G1, ignorando donde D NO la aprobó
            target2_semi = pseudo1.clone()
            # Primero expandimos mask1 a full-res:
            scale = pred1_u.shape[-1] // mask1.shape[-1]  # ej. 320//10=32
            mask1_full = mask1.repeat_interleave(scale, 2).repeat_interleave(scale, 3)
            target2_semi[mask1_full == 0] = 255
            semi2 = self.lambda_semi * self.seg_loss(pred2_u, target2_semi)

            # Target para G1: la pseudo‐máscara de G2, ignorando donde D NO la aprobó
            target1_semi = pseudo2.clone()
            mask2_full = mask2.repeat_interleave(scale, 2).repeat_interleave(scale, 3)
            target1_semi[mask2_full == 0] = 255
            semi1 = self.lambda_semi * self.seg_loss(pred1_u, target1_semi)

            # —— 1) extraer pseudo-etiquetas y confidencias ——
            if self.classification_loss:
                probs1_cls = torch.softmax(logits_clas1_u.detach(), dim=1)  # (B,2)
                conf1_cls, pseudo1_cls = probs1_cls.max(dim=1)  # (B,) ambas
                probs2_cls = torch.softmax(logits_clas2_u.detach(), dim=1)
                conf2_cls, pseudo2_cls = probs2_cls.max(dim=1)

                # —— 2) máscaras de confianza alta ——
                mask1_cls = conf1_cls > self.gamma_thresh  # BoolTensor (B,)
                mask2_cls = conf2_cls > self.gamma_thresh

                # —— 3) perder clasificación cruzada (solo donde mask==True) ——
                if mask1_cls.any():
                    loss_clas2_u = self.clas_loss_fn(
                        logits_clas2_u[mask1_cls],  # pred del G2
                        pseudo1_cls[mask1_cls],  # pseudo-target de G1
                    )
                else:
                    loss_clas2_u = 0.0

                if mask2_cls.any():
                    loss_clas1_u = self.clas_loss_fn(
                        logits_clas1_u[mask2_cls],  # pred del G1
                        pseudo2_cls[mask2_cls],  # pseudo-target de G2
                    )
                else:
                    loss_clas1_u = 0.0
        # combine G losses
        g_loss = (
            self.lambda_seg * (loss_seg1 + loss_seg2)
            + self.lambda_adv * (adv1_l + adv2_l)
            + (adv1_u + adv2_u)
            + (semi1 + semi2)
            + self.lambda_clas * (loss_clas1 + loss_clas2)
            + self.lambda_clas_u * (loss_clas1_u + loss_clas2_u)
        )

        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)
        self.log(
            "train/loss_g",
            g_loss,
            on_step=True,  # loguea cada paso
            on_epoch=False,  # y también agrupa en el cierre de cada epoch
            prog_bar=True,  # lo ves en la barra de progreso
            sync_dist=True,  # necesario en DDP para que sólo rank0 escriba
        )

        # ─── DISCRIMINATOR STEP ───
        self.toggle_optimizer(opt_d)

        # real masks → target = 1
        real_prob = masks_l  # already 0/1
        score_real = self.discriminator(real_prob)  # (B,1,H,W)
        loss_d_real = self.adv_loss(score_real, torch.ones_like(score_real))

        # fake masks from G1/G2 → target = 0

        score_f1 = self.discriminator(prob1.detach())
        score_f2 = self.discriminator(prob2.detach())

        loss_d_f1 = self.adv_loss(score_f1, torch.zeros_like(score_f1))
        loss_d_f2 = self.adv_loss(score_f2, torch.zeros_like(score_f2))

        # fake on unlabeled if used
        if epoch >= self.supervised_epochs and adv1_u != 0.0:
            score_d_u1 = self.discriminator(prob1_u.detach())
            score_d_u2 = self.discriminator(prob2_u.detach())
            loss_d_u1 = self.adv_loss(score_d_u1, torch.zeros_like(score_d_u1))
            loss_d_u2 = self.adv_loss(score_d_u2, torch.zeros_like(score_d_u2))
            total_d = loss_d_real + loss_d_f1 + loss_d_f2 + loss_d_u1 + loss_d_u2
            d_loss = total_d / 5.0
        else:
            total_d = loss_d_real + loss_d_f1 + loss_d_f2
            d_loss = total_d / 3.0

        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)
        self.log(
            "train/loss_d",
            d_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )

    def _shared_eval_step(self, batch, stage: str):
        imgs = batch["image"]
        masks = batch["mask"].unsqueeze(1).long()
        stats = {}

        # forward G1/G2
        p1, c1 = self.generator1(self.processg1(imgs))
        p2, c2 = self.generator2(self.processg2(imgs))

        # máscara promediada
        prob_avg = 0.5 * (torch.sigmoid(p1) + torch.sigmoid(p2))
        pred_mask = (prob_avg > 0.5).long()
        stats.update(
            dict(
                zip(
                    ("tp", "fp", "fn", "tn"), get_stats(pred_mask, masks, mode="binary")
                )
            )
        )

        # clasificación (si está activada)
        if self.classification_loss:
            # promedia probabilidades de cls
            probs1 = torch.softmax(c1, dim=1)
            probs2 = torch.softmax(c2, dim=1)
            probs_avg = 0.5 * (probs1 + probs2)
            pred_cls = probs_avg.argmax(dim=1)
            gt_cls = batch["category"].to(self.device)
            correct = (pred_cls == gt_cls).sum()
            total = gt_cls.size(0)
            stats["class_correct"] = correct
            stats["class_total"] = total

        return stats

    def shared_epoch_end(self, outputs, stage: str):
        # segmentación
        tp = torch.cat([o["tp"] for o in outputs])
        fp = torch.cat([o["fp"] for o in outputs])
        fn = torch.cat([o["fn"] for o in outputs])
        tn = torch.cat([o["tn"] for o in outputs])

        self.log_dict(
            {
                f"{stage}_per_image_iou": iou_score(
                    tp, fp, fn, tn, reduction="micro-imagewise"
                ),
                f"{stage}_dataset_iou": iou_score(tp, fp, fn, tn, reduction="micro"),
                f"{stage}_per_image_f1": f1_score(
                    tp, fp, fn, tn, reduction="micro-imagewise"
                ),
                f"{stage}_dataset_f1": f1_score(tp, fp, fn, tn, reduction="micro"),
                f"{stage}_per_image_acc": accuracy(
                    tp, fp, fn, tn, reduction="micro-imagewise"
                ),
                f"{stage}_dataset_acc": accuracy(tp, fp, fn, tn, reduction="micro"),
            },
            prog_bar=True,
            sync_dist=True,
        )

        # clasificación
        if self.classification_loss:
            total_correct = sum(o["class_correct"] for o in outputs)
            total_samples = sum(o["class_total"] for o in outputs)
            class_acc = total_correct.float() / total_samples
            self.log(f"{stage}_class_acc", class_acc, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        out = self._shared_eval_step(batch, "valid")
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        out = self._shared_eval_step(batch, "test")
        self.test_step_outputs.append(out)
        return out

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.log_test_images(
            data_module=self.trainer.datamodule,
            num_images=50,
            threshold=0.1,
            only_roi_frames=True,
        )
        self.test_step_outputs.clear()

    def save_test_overlays(self, data_module, results_path, name="test"):
        """
        Guarda overlays de test *por separado* para G1 y G2.
        results_path/   ├─ test_G1/...
                         └─ test_G2/...
        """
        loader = data_module.test_dataloader()
        self.eval()

        for gen_idx, gen in enumerate([self.generator1, self.generator2], start=1):
            out_dir = os.path.join(results_path, f"{name}_G{gen_idx}")
            os.makedirs(out_dir, exist_ok=True)

            with torch.no_grad():
                for i, batch in enumerate(loader):
                    img = batch["image"].to(self.device)
                    gt = batch["mask"].to(self.device)

                    logits = gen(
                        self.processg1(img) if gen_idx == 1 else self.processg2(img)
                    )
                    prob = logits.sigmoid()
                    pred = (prob > 0.5).float()

                    for j in range(img.size(0)):
                        # reconstruir imagen  [igual que en USModel.save_test_overlays]
                        im_np = (
                            img[j].cpu().numpy().transpose(1, 2, 0)
                            * self.std1.cpu().numpy().squeeze()
                            + self.mean1.cpu().numpy().squeeze()
                        ) * 255
                        im_np = np.clip(im_np, 0, 255).astype(np.uint8)

                        gt_np = gt[j].cpu().numpy().squeeze()
                        pred_np = pred[j].cpu().numpy().squeeze()

                        # ignora volúmenes sin ROI si quieres
                        if gt_np.sum() == 0:
                            continue

                        # overlay GT y pred
                        gt_ov = im_np.copy()
                        gt_ov[gt_np > 0] = (
                            gt_ov[gt_np > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
                        ).astype(np.uint8)

                        pr_ov = im_np.copy()
                        pr_ov[pred_np > 0] = (
                            pr_ov[pred_np > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
                        ).astype(np.uint8)

                        combo = np.concatenate([im_np, gt_ov, pr_ov], axis=1)
                        cv2.imwrite(
                            os.path.join(out_dir, f"img_{i}_{j}.png"),
                            cv2.cvtColor(combo, cv2.COLOR_RGB2BGR),
                        )

        print(f"Saved test overlays for G1/G2 under {results_path}")

    def log_test_images(
        self,
        data_module,
        threshold=0.4,
        only_roi_frames=False,
        num_images=10,
    ):
        self.eval()
        device = self.device
        # tomamos una pequeña muestra del test
        full_loader = data_module.test_dataloader()
        indices = np.random.choice(
            len(full_loader.dataset),
            size=min(num_images, len(full_loader.dataset)),
            replace=False,
        )
        subset = torch.utils.data.Subset(full_loader.dataset, indices)
        loader = torch.utils.data.DataLoader(
            subset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
        )

        wandb_images = []
        with torch.no_grad():
            for batch in loader:
                img = batch["image"].to(device)  # (1,3,H,W)
                gt = batch["mask"].to(device)  # (1,H,W)
                # preprocesos y forward G1/G2
                logits1,_ = self.generator1(self.processg1(img))
                logits2,_ = self.generator2(self.processg2(img))
                prob1 = torch.sigmoid(logits1) > 0.5
                prob2 = torch.sigmoid(logits2) > 0.5

                # sacamos numpy para plotting
                img_np = img[0].cpu().numpy().transpose(1, 2, 0)
                # invertimos standarización
                # Verificar si es imagen en escala de grises (1 canal)
                if img_np.shape[-1] == 1:
                    img_np = img_np.squeeze(-1)  # Convertir (H, W, 1) a (H, W)

                # Ajustar rango de valores si están entre -1 y 1
                if img_np.min() < 0:
                    img_np = (img_np + 1) / 2.0  # Rango de [-1,1] a [0,1]

                # Convertir al rango de 0 a 255
                img_np = (img_np * 255).astype(np.uint8)

                gt_np = gt[0].cpu().numpy().squeeze()
                p1_np = prob1[0].cpu().numpy().squeeze().astype(np.uint8)
                p2_np = prob2[0].cpu().numpy().squeeze().astype(np.uint8)

                # saltarnos imágenes sin ROI si queremos
                if only_roi_frames and gt_np.max() == 0:
                    continue

                # montar figura 1×4
                fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                axs[0].imshow(img_np, cmap="gray")
                axs[0].set_title("Imagen")
                axs[1].imshow(gt_np, cmap="gray")
                axs[1].set_title("GT")
                axs[2].imshow(p1_np, cmap="gray")
                axs[2].set_title("G1")
                axs[3].imshow(p2_np, cmap="gray")
                axs[3].set_title("G2")
                for ax in axs:
                    ax.axis("off")
                plt.tight_layout()

                wandb_images.append(
                    wandb.Image(fig, caption=f"Ejemplo {len(wandb_images) + 1}")
                )
                plt.close(fig)

        if wandb_images:
            # key: nombre de la tabla en W&B
            # images: lista de tensores/arrays/PIL o wandb.Image
            self.logger.log_image(
                key="test_examples_G1_G2",
                images=wandb_images,
            )
        else:
            print("No se encontraron imágenes con máscara para loggear.")

    def configure_optimizers(self):
        if self.adam_g:
            # Adam for generator
            opt_g = optim.Adam(
                itertools.chain(
                    self.generator1.parameters(), self.generator2.parameters()
                ),
                lr=self.lr_g,
                betas=self.adam_betas,
                weight_decay=self.weight_decay,
            )
        else:
            # Sgd for generator
            opt_g = optim.SGD(
                itertools.chain(
                    self.generator1.parameters(), self.generator2.parameters()
                ),
                lr=self.lr_g,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_d,
            betas=self.adam_betas,
            weight_decay=self.weight_decay,
        )
        # 2) Schedulers
        total_epochs = self.trainer.max_epochs
        power = self.poly_power

        scheduler_g = {
            "scheduler": PolynomialLR(opt_g, total_iters=total_epochs, power=power),
            "interval": "epoch",  # ajustar lr al final de cada época
            "frequency": 1,
            "name": "poly_lr_g",
        }
        scheduler_d = {
            "scheduler": PolynomialLR(opt_d, total_iters=total_epochs, power=power),
            "interval": "epoch",
            "frequency": 1,
            "name": "poly_lr_d",
        }

        # 3) Devolver listas de optimizadores y schedulers
        return [opt_g, opt_d], [scheduler_g, scheduler_d]
