import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
from gan_modules import Discriminator


class DualSegGAN(L.LightningModule):
    def __init__(
        self,
        model_opts,  # debe contener args.arch1, args.encoder_name1  para G1 (DeepLabV3+)
        #               args.arch2, args.encoder_name2  para G2 (PSPNet)
        #               args.inchannels, args.outchannels, args.encoder_weights
        train_par,  # debe contener lr_g, lr_d, weight_decay, lambda_seg, lambda_adv
    ):
        super().__init__()
        # ⚠️ optimization manual para controlar pasos G/D
        self.automatic_optimization = False

        # ——————————————————————————————
        # 🔧 Generadores
        # — G1: DeepLabV3+
        self.generator1 = smp.create_model(
            model_opts.args.arch1,
            encoder_name=model_opts.args.encoder_name1,
            in_channels=model_opts.args.inchannels,
            classes=model_opts.args.outchannels,
            encoder_weights=model_opts.args.encoder_weights,
        )
        # — G2: PSPNet
        self.generator2 = smp.create_model(
            model_opts.args.arch2,
            encoder_name=model_opts.args.encoder_name2,
            in_channels=model_opts.args.inchannels,
            classes=model_opts.args.outchannels,
            encoder_weights=model_opts.args.encoder_weights,
        )

        # ——————————————————————————————
        # 🔧 Discriminador (igual al de CycleGAN)
        self.discriminator = Discriminator(img_channels=model_opts.args.outchannels)

        # ——————————————————————————————
        # 🏷 Pérdidas y Hiperparámetros
        # pérdida de segmentación: BCE para binario, CrossEntropy si classes>1
        if model_opts.args.outchannels == 1:
            self.seg_loss = nn.BCEWithLogitsLoss(reduction="mean")
        else:
            self.seg_loss = nn.CrossEntropyLoss()

        # adversarial: queremos que D(fake)≈1 para engañar al discriminador
        self.adv_loss = nn.BCEWithLogitsLoss()

        # learning‐rates y betas
        self.lr_g = train_par.lr_g
        self.lr_d = train_par.lr_d
        self.weight_decay = train_par.weight_decay
        self.adam_betas = (0.9, 0.9)

        # coeficientes de cada término
        self.lambda_seg = train_par.lambda_seg
        self.lambda_adv = train_par.lambda_adv

        # ——————————————————————————————
        # 📋 Normalización según cada encoder
        p1 = smp.encoders.get_preprocessing_params(model_opts.args.encoder_name1)
        self.register_buffer("mean1", torch.tensor(p1["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std1", torch.tensor(p1["std"]).view(1, 3, 1, 1))
        p2 = smp.encoders.get_preprocessing_params(model_opts.args.encoder_name2)
        self.register_buffer("mean2", torch.tensor(p2["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std2", torch.tensor(p2["std"]).view(1, 3, 1, 1))

    def forward(self, x):
        # no lo usamos directamente, entrenamos por training_step
        raise NotImplementedError("usa training_step para controlar G y D por separado")

    def training_step(self, batch, batch_idx):
        imgs = batch["image"]  # (B, C, H, W)
        masks = batch["mask"].unsqueeze(1)  # (B, 1, H, W)
        conf = batch.get(
            "confidence", 1.0
        )  # coeficiente de confianza (escalar o tensor (B,))

        opt_g, opt_d = self.optimizers()

        # =======================
        # 🟢 Paso Generadores
        # =======================
        self.toggle_optimizer(opt_g)

        # normalizar cada uno
        x1 = (imgs - self.mean1) / self.std1
        x2 = (imgs - self.mean2) / self.std2

        # inferencia
        pred1 = self.generator1(x1)  # (B,1,H,W) logits
        pred2 = self.generator2(x2)

        # — Pérdidas de segmentación
        loss1 = self.seg_loss(pred1, masks)
        loss2 = self.seg_loss(pred2, masks)
        # aplicamos coeficiente de confianza
        loss1 = loss1 * conf
        loss2 = loss2 * conf

        # — Pérdida adversarial para G: D(fake) debería ser 1
        score_fake1 = self.discriminator(pred1)
        score_fake2 = self.discriminator(pred2)
        adv1 = self.adv_loss(score_fake1, torch.ones_like(score_fake1))
        adv2 = self.adv_loss(score_fake2, torch.ones_like(score_fake2))

        total_g_loss = self.lambda_seg * (loss1 + loss2) + self.lambda_adv * (
            adv1 + adv2
        )

        # backward y paso
        self.manual_backward(total_g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        self.log("train/loss_g", total_g_loss, prog_bar=True, sync_dist=True)

        # =======================
        # 🔴 Paso Discriminador
        # =======================
        self.toggle_optimizer(opt_d)

        # D(real)=1, D(fake)=0
        score_real = self.discriminator(masks)
        score_fake1D = self.discriminator(pred1.detach())
        score_fake2D = self.discriminator(pred2.detach())

        loss_d_real = self.adv_loss(score_real, torch.ones_like(score_real))
        loss_d_f1 = self.adv_loss(score_fake1D, torch.zeros_like(score_fake1D))
        loss_d_f2 = self.adv_loss(score_fake2D, torch.zeros_like(score_fake2D))

        total_d_loss = (loss_d_real + loss_d_f1 + loss_d_f2) / 3

        self.manual_backward(total_d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        self.log("train/loss_d", total_d_loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # — Optimizador de Generadores (ambos juntos)
        opt_g = optim.Adam(
            itertools.chain(self.generator1.parameters(), self.generator2.parameters()),
            lr=self.lr_g,
            betas=self.adam_betas,
            weight_decay=self.weight_decay,
        )
        # — Optimizador Discriminador
        opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_d,
            betas=self.adam_betas,
            weight_decay=self.weight_decay,
        )

        # schedulers estilo CycleGAN
        sched_g = lr_scheduler.StepLR(opt_g, step_size=100, gamma=0.5)
        sched_d = lr_scheduler.StepLR(opt_d, step_size=100, gamma=0.5)

        return [
            {
                "optimizer": opt_g,
                "lr_scheduler": {"scheduler": sched_g, "interval": "epoch"},
            },
            {
                "optimizer": opt_d,
                "lr_scheduler": {"scheduler": sched_d, "interval": "epoch"},
            },
        ]
