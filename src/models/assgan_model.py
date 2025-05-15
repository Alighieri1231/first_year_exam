import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import segmentation_models_pytorch as smp

from gan_modules import Discriminator, FCDiscriminator

# ──────── LOSS DEFINITION ────────
# Option A: your original 2d‐masks + wrappers
from utils.loss import CrossEntropy2d, BCEWithLogitsLoss2d

# Option B: pure PyTorch (commented out)
# from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


class ASSGAN(L.LightningModule):
    def __init__(self, model_opts, train_par):
        super().__init__()
        # manual optimization so we can step G and D separately
        self.automatic_optimization = False

        # — Generators (each outputs 1‐channel logits) —
        self.generator1 = smp.create_model(
            model_opts.args.arch1,
            encoder_name=model_opts.args.encoder_name1,
            in_channels=3,
            classes=1,
            encoder_weights="imagenet",
        )
        self.generator2 = smp.create_model(
            model_opts.args.arch2,
            encoder_name=model_opts.args.encoder_name2,
            in_channels=3,
            classes=1,
            encoder_weights="imagenet",
        )

        # — Discriminator (takes 1‐channel probability maps) —
        self.discriminator = FCDiscriminator(num_classes=1)

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

        # ─── Hyperparameters ───
        self.lr_g = train_par.lr_g
        self.lr_d = train_par.lr_d
        self.weight_decay = train_par.weight_decay
        self.adam_betas = (0.9, 0.9)
        self.lambda_seg = train_par.lambda_seg
        self.lambda_adv = train_par.lambda_adv
        self.lambda_adv_u = train_par.lambda_adv_u  # weight for unlabeled adv loss
        self.lambda_semi = train_par.lambda_semi  # weight for pseudo‐loss
        self.supervised_epochs = train_par.supervised_epochs  # e.g. 200
        self.gamma_thresh = train_par.gamma_thresh  # e.g. 0.2

        # buffer for pulling unlabeled data
        self.unlab_iter = None

    def forward(self, x):
        raise NotImplementedError("Use training_step to control G/D separately")

    def training_step(self, batch, batch_idx):
        epoch = self.current_epoch
        imgs_l = batch["image"]  # (B,3,H,W)
        masks_l = batch["mask"].unsqueeze(1)  # (B,1,H,W), values {0,1}

        opt_g, opt_d = self.optimizers()

        # ─── GENERATOR STEP ───
        self.toggle_optimizer(opt_g)

        # Supervised training

        # 1) forward
        pred1 = self.generator1(imgs_l)  # (B,1,H,W) logits
        pred2 = self.generator2(imgs_l)

        # 2) segmentation loss on labeled
        loss_seg1 = self.seg_loss(pred1, masks_l)
        loss_seg2 = self.seg_loss(pred2, masks_l)

        # 3) adversarial “fake” loss  → we want D(sigmoid(pred)) ≈ 1
        prob1 = torch.sigmoid(pred1)
        prob2 = torch.sigmoid(pred2)
        adv1_l = self.adv_loss(self.discriminator(prob1), torch.ones_like(prob1))
        adv2_l = self.adv_loss(self.discriminator(prob2), torch.ones_like(prob2))

        # optional: unlabeled adversarial
        adv1_u = adv2_u = 0.0
        semi1 = semi2 = 0.0
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

            imgs_u = unlab["image"]  # (B,3,H,W)
            # forward G on unlabeled
            pred1_u = self.generator1(imgs_u)
            pred2_u = self.generator2(imgs_u)
            prob1_u = torch.sigmoid(pred1_u)
            prob2_u = torch.sigmoid(pred2_u)

            # 3) adversarial on unlabeled
            adv1_u = self.adv_loss(
                self.discriminator(prob1_u), torch.ones_like(prob1_u)
            )
            adv2_u = self.adv_loss(
                self.discriminator(prob2_u), torch.ones_like(prob2_u)
            )

            # 4) pseudo‐label mutual supervision (Alg 2)
            # discriminator’s confidence
            conf1 = torch.sigmoid(self.discriminator(prob1_u))  # (B,1,H,W)
            conf2 = torch.sigmoid(self.discriminator(prob2_u))

            # masks of “trusted” pixels
            mask1 = (conf1 > self.gamma_thresh).float()
            mask2 = (conf2 > self.gamma_thresh).float()

            # hard pseudo‐labels
            pseudo1 = (prob1_u > 0.5).float()
            pseudo2 = (prob2_u > 0.5).float()

            # supervise G₂ where mask1==1 using G₁’s pseudo
            if mask1.sum() > 0:
                target2 = pseudo1.clone()
                target2[mask1 == 0] = 255
                semi2 = self.lambda_semi * self.seg_loss(pred2_u, target2)

            # supervise G₁ where mask2==1 using G₂’s pseudo
            if mask2.sum() > 0:
                target1 = pseudo2.clone()
                target1[mask2 == 0] = 255
                semi1 = self.lambda_semi * self.seg_loss(pred1_u, target1)

        # combine G losses
        g_loss = (
            self.lambda_seg * (loss_seg1 + loss_seg2)
            + self.lambda_adv * (adv1_l + adv2_l + adv1_u + adv2_u)
            + (semi1 + semi2)
        )

        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)
        self.log("train/loss_g", g_loss, prog_bar=True)

        # ─── DISCRIMINATOR STEP ───
        self.toggle_optimizer(opt_d)

        # real masks → target = 1
        real_prob = masks_l  # already 0/1
        loss_d_real = self.adv_loss(
            self.discriminator(real_prob), torch.ones_like(real_prob)
        )

        # fake masks from G1/G2 → target = 0
        loss_d_f1 = self.adv_loss(
            self.discriminator(prob1.detach()), torch.zeros_like(prob1)
        )
        loss_d_f2 = self.adv_loss(
            self.discriminator(prob2.detach()), torch.zeros_like(prob2)
        )

        # fake on unlabeled if used
        if epoch >= self.supervised_epochs and adv1_u != 0.0:
            loss_d_u1 = self.adv_loss(
                self.discriminator(prob1_u.detach()), torch.zeros_like(prob1_u)
            )
            loss_d_u2 = self.adv_loss(
                self.discriminator(prob2_u.detach()), torch.zeros_like(prob2_u)
            )
            total_d = loss_d_real + loss_d_f1 + loss_d_f2 + loss_d_u1 + loss_d_u2
            d_loss = total_d / 5.0
        else:
            total_d = loss_d_real + loss_d_f1 + loss_d_f2
            d_loss = total_d / 3.0

        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)
        self.log("train/loss_d", d_loss, prog_bar=True)

    def configure_optimizers(self):
        opt_g = optim.Adam(
            itertools.chain(self.generator1.parameters(), self.generator2.parameters()),
            lr=self.lr_g,
            betas=self.adam_betas,
            weight_decay=self.weight_decay,
        )
        opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_d,
            betas=self.adam_betas,
            weight_decay=self.weight_decay,
        )
        return [opt_g, opt_d]
