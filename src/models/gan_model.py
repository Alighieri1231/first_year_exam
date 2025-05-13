import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.utils import spectral_norm
import lightning as L
from lpips import LPIPS

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr
import itertools

from gan_modules import GeneratorResNet, Discriminator
import numpy as np
from metrics import normalized_cross_correlation,mutual_information,contrast_structure_similarity,bhattacharyya_coefficient, mean_absolute_error

def denormalize(img):
    """Converts [-1,1] to [0,1]""" 
    return (img + 1) / 2

class CycleGAN(L.LightningModule):
    def __init__(self, img_channels=1, n_residual_blocks=15, lr_g=3e-4, lr_d=3e-3, lambda_cycle=12, lambda_id=1, lambda_adv=1, lambda_per=1):
        super().__init__()

        # Device detection for future models
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ⚠️ Manual optimization
        #Necessary because we have two optimizers
        # and we want to control the order of the optimization steps
        self.automatic_optimization = False  

        # Losses
        self.adversarial_loss = nn.MSELoss()
        self.cycle_loss = nn.L1Loss()
        self.identity_loss = nn.L1Loss()
        self.perceptual_loss = LPIPS(net='vgg')

        # Hyperparameters
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.adam_betas = (0.9, 0.9)
        self.n_residual_blocks = n_residual_blocks
        self.lambda_cycle = lambda_cycle
        self.lambda_id = lambda_id
        self.lambda_adv = lambda_adv
        self.lambda_per = lambda_per

        # Models
        self.generator_xy = GeneratorResNet(img_channels, n_residual_blocks)
        self.generator_yx = GeneratorResNet(img_channels, n_residual_blocks)
        self.discriminator_x = Discriminator(img_channels)
        self.discriminator_y = Discriminator(img_channels)

    def forward(self, img_x):
        return self.generator_xy(img_x)

    def training_step(self, batch, batch_idx):
        real_x, real_y = batch["x"], batch["y"]
        fake_y = self.generator_xy(real_x)
        fake_x = self.generator_yx(real_y)

        # ⚡ Obtaining optimizers
        opt_g, opt_d = self.optimizers()

        # =======================
        # 🟢 Generator training
        # =======================
        self.toggle_optimizer(opt_g)

        rec_x = self.generator_yx(fake_y)
        rec_y = self.generator_xy(fake_x)

        identity_loss = self.identity_loss(self.generator_yx(real_x), real_x) + \
                        self.identity_loss(self.generator_xy(real_y), real_y)
        
        adv_loss = self.adversarial_loss(self.discriminator_y(fake_y), torch.ones_like(self.discriminator_y(fake_y))) + \
                   self.adversarial_loss(self.discriminator_x(fake_x), torch.ones_like(self.discriminator_x(fake_x)))

        cycle_loss = self.cycle_loss(rec_x, real_x) + self.cycle_loss(rec_y, real_y)
        
        #perceptual_loss = self.perceptual_loss(fake_y.repeat(1, 3, 1, 1), real_y.repeat(1, 3, 1, 1)).mean()
        perceptual_loss = self.perceptual_loss(fake_y.repeat(1, 3, 1, 1), real_y.repeat(1, 3, 1, 1)).mean() + \
                  self.perceptual_loss(fake_x.repeat(1, 3, 1, 1), real_x.repeat(1, 3, 1, 1)).mean()


        total_g_loss = self.lambda_id * identity_loss + \
                       self.lambda_cycle * cycle_loss + \
                       self.lambda_adv * adv_loss + \
                       self.lambda_per * perceptual_loss

        self.manual_backward(total_g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        self.log("loss_generator", total_g_loss, prog_bar=True,sync_dist=True)

        # =======================
        # 🔴 Discriminator training
        # =======================
        self.toggle_optimizer(opt_d)

        D_H_real = self.discriminator_x(real_x)
        D_L_real = self.discriminator_y(real_y)
        D_H_fake = self.discriminator_x(fake_x.detach())
        D_L_fake = self.discriminator_y(fake_y.detach())

        # Label smoothing
        mean_D_H = torch.mean(D_H_real).item()
        mean_D_L = torch.mean(D_L_real).item()
        label_real = 1.0 if mean_D_H < 0.9 and mean_D_L < 0.9 else 0.9
        label_fake = 0.0

        real_loss = self.adversarial_loss(D_H_real, torch.full_like(D_H_real, label_real)) + \
                    self.adversarial_loss(D_L_real, torch.full_like(D_L_real, label_real))

        fake_loss = self.adversarial_loss(D_H_fake, torch.full_like(D_H_fake, label_fake)) + \
                    self.adversarial_loss(D_L_fake, torch.full_like(D_L_fake, label_fake))

        total_d_loss = (real_loss + fake_loss) / 2

        self.manual_backward(total_d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        self.log("loss_discriminator", total_d_loss, prog_bar=True,sync_dist=True)

    # def configure_optimizers(self):
    #     opt_g = optim.Adam(itertools.chain(self.generator_xy.parameters(), self.generator_yx.parameters()), 
    #                        lr=self.lr_g, betas=self.adam_betas)
    #     opt_d = optim.Adam(itertools.chain(self.discriminator_x.parameters(), self.discriminator_y.parameters()), 
    #                        lr=self.lr_d, betas=self.adam_betas)
        
    #     #add step scheduler
    #     scheduler_g = optim.lr_scheduler.StepLR(opt_g, step_size=100, gamma=0.5)
    #     scheduler_d = optim.lr_scheduler.StepLR(opt_d, step_size=100, gamma=0.5)

    #     return [opt_g, opt_d]
    
    def configure_optimizers(self):
        # 📌 **Optimizadores para Generadores y Discriminadores**
        opt_g = optim.Adam(itertools.chain(self.generator_xy.parameters(), self.generator_yx.parameters()), 
                        lr=self.lr_g, betas=self.adam_betas)
        
        opt_d = optim.Adam(itertools.chain(self.discriminator_x.parameters(), self.discriminator_y.parameters()), 
                        lr=self.lr_d, betas=self.adam_betas)

        # 📌 **Schedulers con StepLR para reducir el LR cada 100 épocas**
        scheduler_g = lr_scheduler.StepLR(opt_g, step_size=100, gamma=0.5)
        scheduler_d = lr_scheduler.StepLR(opt_d, step_size=100, gamma=0.5)

        return [
            {"optimizer": opt_g, "lr_scheduler": {"scheduler": scheduler_g, "interval": "epoch", "frequency": 1}},
            {"optimizer": opt_d, "lr_scheduler": {"scheduler": scheduler_d, "interval": "epoch", "frequency": 1}},
        ]

    def validation_step(self, batch, batch_idx):
        return self._compute_metrics(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._compute_metrics(batch, "test")

    # Update: Metrics believed to be corrected.
    # TO-DO: Tune CycleGAN based on metrics.
    def _compute_metrics(self, batch, prefix):
        real_x, real_y = batch["x"], batch["y"]
        fake_y = self.generator_xy(real_x)
        fake_x = self.generator_yx(real_y)

        ssim_scores, psnr_scores, lncc_scores, css_scores, mi_scores, bc_scores, mae_scores = [], [], [], [], [], [], []

        for i in range(real_y.shape[0]):
            real_y_np = denormalize(real_y[i]).float().cpu().detach().numpy().squeeze()
            fake_y_np = denormalize(fake_y[i]).float().cpu().detach().numpy().squeeze()

            ssim_scores.append(ssim(fake_y_np, real_y_np, data_range=1))
            psnr_scores.append(psnr(fake_y_np, real_y_np, data_range=1))
            lncc_scores.append(normalized_cross_correlation(fake_y_np, real_y_np))
            css_scores.append(contrast_structure_similarity(fake_y_np, real_y_np))
            mi_scores.append(mutual_information(fake_y_np, real_y_np))
            bc_scores.append(bhattacharyya_coefficient(fake_y_np, real_y_np))
            mae_scores.append(mean_absolute_error(fake_y_np, real_y_np))

        avg_ssim = sum(ssim_scores) / len(ssim_scores)
        avg_psnr = sum(psnr_scores) / len(psnr_scores)
        avg_lncc = sum(lncc_scores) / len(lncc_scores)
        avg_css = sum(css_scores) / len(css_scores)
        avg_mi = sum(mi_scores) / len(mi_scores)
        avg_bc = sum(bc_scores) / len(bc_scores)
        avg_mae = sum(mae_scores) / len(mae_scores)

        self.log(f"{prefix}_ssim", avg_ssim, sync_dist=True)
        self.log(f"{prefix}_psnr", avg_psnr, sync_dist=True)
        self.log(f"{prefix}_lncc", avg_lncc, sync_dist=True)
        self.log(f"{prefix}_css", avg_css, sync_dist=True)
        self.log(f"{prefix}_mi", avg_mi, sync_dist=True)
        self.log(f"{prefix}_bc", avg_bc, sync_dist=True)
        self.log(f"{prefix}_mae", avg_mae, sync_dist=True)

    def statistics_step(self, batch, batch_idx):
        return self._compute_individual_metrics(batch, "test")

    def _compute_individual_metrics(self, batch):
        """
        Computes *per-image* metrics for the batch, returning 7 separate lists.

        Returns:
            (ssim_scores, psnr_scores, lncc_scores, css_scores, mi_scores, bc_scores, mae_scores)
        """
        real_x = batch["x"].to(self.device_)
        real_y = batch["y"].to(self.device_)
        fake_y = self.generator_xy(real_x)
        fake_x = self.generator_yx(real_y)

        ssim_scores, psnr_scores = [], []
        lncc_scores, css_scores = [], []
        mi_scores, bc_scores, mae_scores = [], [], []

        for i in range(real_y.shape[0]):
            real_y_np = denormalize(real_y[i]).float().cpu().detach().numpy().squeeze()
            fake_y_np = denormalize(fake_y[i]).float().cpu().detach().numpy().squeeze()

            ssim_scores.append(ssim(fake_y_np, real_y_np, data_range=1))
            psnr_scores.append(psnr(fake_y_np, real_y_np, data_range=1))

            lncc_scores.append(normalized_cross_correlation(fake_y_np, real_y_np))
            css_scores.append(contrast_structure_similarity(fake_y_np, real_y_np))
            mi_scores.append(mutual_information(fake_y_np, real_y_np))
            bc_scores.append(bhattacharyya_coefficient(fake_y_np, real_y_np))
            mae_scores.append(mean_absolute_error(fake_y_np, real_y_np))

        return (ssim_scores, psnr_scores, lncc_scores, css_scores, mi_scores, bc_scores, mae_scores)    
