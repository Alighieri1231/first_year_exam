import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from lpips import LPIPS

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Assuming you have these metrics available in metrics.py
from metrics import (
    normalized_cross_correlation,
    mutual_information,
    contrast_structure_similarity,
    bhattacharyya_coefficient,
    mean_absolute_error
)

def denormalize(img: torch.Tensor) -> torch.Tensor:
    """
    Converts a tensor from [-1,1] range to [0,1].
    """
    return (img + 1) / 2

class ResidualBlock(nn.Module):
    """
    Basic residual block used inside the generator’s ResNet architecture.
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    """
    CycleGAN Generator (ResNet-based), adapted for single-channel images (default).
    """
    def __init__(self, input_channels: int, n_residual_blocks: int = 15):
        super().__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        res_blocks = [ResidualBlock(256) for _ in range(n_residual_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)

        # Decoder with skip connections
        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, input_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        res = self.res_blocks(enc3)

        # "Skip" additions
        dec1 = self.decoder1(res) + enc2
        dec2 = self.decoder2(dec1) + enc1
        return self.output_layer(dec2)


class Discriminator(nn.Module):
    """
    CycleGAN PatchGAN Discriminator (with Spectral Norm).
    """
    def __init__(self, input_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            # First layer (no spectral_norm here to match your code exactly):
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Spectral Norm blocks:
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Output:
            nn.Conv2d(512, 1, kernel_size=4, padding=1),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.model(img)


class CycleGAN(nn.Module):
    """
    Plain PyTorch CycleGAN model that holds two generators (X->Y and Y->X) and two discriminators.
    Also provides a method `_compute_individual_metrics` to gather metrics on a batch,
    so that external scripts can call `model._compute_individual_metrics(...)` just like in the Lightning version.
    """
    def __init__(
        self,
        img_channels=1,
        n_residual_blocks=15,
        lr_g=3e-4,
        lr_d=3e-3,
        lambda_cycle=12,
        lambda_id=1,
        lambda_adv=1,
        lambda_per=1
    ):
        super().__init__()

        # Device detection for future models
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ----------------
        # Hyperparameters
        # ----------------
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.lambda_cycle = lambda_cycle
        self.lambda_id = lambda_id
        self.lambda_adv = lambda_adv
        self.lambda_per = lambda_per

        # ---------------
        # Loss functions
        # ---------------
        self.adversarial_loss = nn.MSELoss()
        self.cycle_loss = nn.L1Loss()
        self.identity_loss = nn.L1Loss()
        self.perceptual_loss = LPIPS(net='vgg')  # from lpips library

        # ---------------
        #   Networks
        # ---------------
        self.generator_xy = GeneratorResNet(img_channels, n_residual_blocks)
        self.generator_yx = GeneratorResNet(img_channels, n_residual_blocks)
        self.discriminator_x = Discriminator(img_channels)
        self.discriminator_y = Discriminator(img_channels)

    def forward(self, img_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass is defined as X->Y generator (fake_y).
        This is mostly for convenience, e.g.:
            `output = model(x)`
        calls `generator_xy`.
        """
        return self.generator_xy(img_x)

    def generate_y(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate "high-quality" image from a "low-quality" image.
        """
        return self.generator_xy(x)

    def generate_x(self, y: torch.Tensor) -> torch.Tensor:
        """
        Generate "low-quality" image from a "high-quality" image.
        """
        return self.generator_yx(y)

    def _compute_individual_metrics(self, batch: dict):
        """
        Computes *per-image* metrics for the batch, returning 7 lists:
            (ssim_scores, psnr_scores, lncc_scores, css_scores, mi_scores, bc_scores, mae_scores)

        Exactly as in your Lightning version’s `_compute_individual_metrics()`.
        Used by external scripts to gather statistics on the entire test set.
        """
        real_x = batch["x"].to(self.device_)
        real_y = batch["y"].to(self.device_)

        # Generate
        fake_y = self.generate_y(real_x)
        fake_x = self.generate_x(real_y)

        ssim_scores, psnr_scores = [], []
        lncc_scores, css_scores = [], []
        mi_scores, bc_scores, mae_scores = [], [], []

        batch_size = real_x.size(0)

        for i in range(batch_size):
            # Denormalize from [-1,1] -> [0,1]
            real_y_np = denormalize(real_y[i]).float().cpu().numpy().squeeze()
            fake_y_np = denormalize(fake_y[i]).float().cpu().numpy().squeeze()

            # Compute metrics
            ssim_scores.append(ssim(fake_y_np, real_y_np, data_range=1))
            psnr_scores.append(psnr(fake_y_np, real_y_np, data_range=1))
            lncc_scores.append(normalized_cross_correlation(fake_y_np, real_y_np))
            css_scores.append(contrast_structure_similarity(fake_y_np, real_y_np))
            mi_scores.append(mutual_information(fake_y_np, real_y_np))
            bc_scores.append(bhattacharyya_coefficient(fake_y_np, real_y_np))
            mae_scores.append(mean_absolute_error(fake_y_np, real_y_np))

        return (
            ssim_scores,
            psnr_scores,
            lncc_scores,
            css_scores,
            mi_scores,
            bc_scores,
            mae_scores
        )
