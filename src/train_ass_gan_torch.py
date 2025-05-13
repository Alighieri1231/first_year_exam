import os
import torch
import wandb
import itertools
import numpy as np

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# Your custom DataModule, presumably in datamodule.py
from gan_datamodule import ImageDataModule

# Import the CycleGAN model from cyclegan_model.py
from gan_model_torch import CycleGAN

# For reproducibility (instead of seed_everything from Lightning)
def set_seed(seed=2024):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(
    model: CycleGAN,
    dataloader,
    opt_g,
    opt_d,
    device: torch.device
):
    """
    One epoch of training for both generator(s) and discriminator(s).
    Returns average losses for logging.
    """
    model.train()

    # Shortcuts to generator/discriminator
    G_xy = model.generator_xy
    G_yx = model.generator_yx
    D_x  = model.discriminator_x
    D_y  = model.discriminator_y

    adv_loss_fn       = model.adversarial_loss
    cycle_loss_fn     = model.cycle_loss
    identity_loss_fn  = model.identity_loss
    perceptual_lossFn = model.perceptual_loss

    lambda_adv   = model.lambda_adv
    lambda_cycle = model.lambda_cycle
    lambda_id    = model.lambda_id
    lambda_per   = model.lambda_per

    running_g_loss = 0.0
    running_d_loss = 0.0
    num_batches    = 0

    for batch in dataloader:
        num_batches += 1
        real_x = batch["x"].to(device)
        real_y = batch["y"].to(device)

        # ---------------------------
        #  Train Generators (X->Y, Y->X)
        # ---------------------------
        opt_g.zero_grad()

        # Forward pass for G
        fake_y = G_xy(real_x)
        fake_x = G_yx(real_y)
        rec_x  = G_yx(fake_y)
        rec_y  = G_xy(fake_x)

        # Identity loss
        id_loss = identity_loss_fn(G_yx(real_x), real_x) + \
                  identity_loss_fn(G_xy(real_y), real_y)

        # Adversarial loss
        pred_fake_y = D_y(fake_y)
        loss_adv_y  = adv_loss_fn(pred_fake_y, torch.ones_like(pred_fake_y))
        pred_fake_x = D_x(fake_x)
        loss_adv_x  = adv_loss_fn(pred_fake_x, torch.ones_like(pred_fake_x))
        adv_loss = loss_adv_y + loss_adv_x

        # Cycle loss
        cycle_loss = cycle_loss_fn(rec_x, real_x) + cycle_loss_fn(rec_y, real_y)

        # Perceptual loss
        # (convert 1‐channel to 3‐channel for LPIPS)
        per_loss = perceptual_lossFn(
            fake_y.repeat(1,3,1,1), real_y.repeat(1,3,1,1)
        ).mean() + perceptual_lossFn(
            fake_x.repeat(1,3,1,1), real_x.repeat(1,3,1,1)
        ).mean()

        total_g_loss = (
            lambda_id    * id_loss +
            lambda_cycle * cycle_loss +
            lambda_adv   * adv_loss +
            lambda_per   * per_loss
        )

        total_g_loss.backward()
        opt_g.step()

        # ---------------------------
        #  Train Discriminators (X, Y)
        # ---------------------------
        opt_d.zero_grad()

        # Real/fake predictions
        D_H_real = D_x(real_x)   # D_x sees real X
        D_L_real = D_y(real_y)   # D_y sees real Y
        D_H_fake = D_x(fake_x.detach())
        D_L_fake = D_y(fake_y.detach())

        # Label smoothing
        mean_D_H = torch.mean(D_H_real).item()
        mean_D_L = torch.mean(D_L_real).item()
        label_real_val = 1.0 if (mean_D_H < 0.9 and mean_D_L < 0.9) else 0.9
        label_real = torch.full_like(D_H_real, label_real_val)
        label_fake = torch.zeros_like(D_H_fake)

        # Discriminator loss for real
        real_loss = adv_loss_fn(D_H_real, label_real) + \
                    adv_loss_fn(D_L_real, label_real)

        # Discriminator loss for fake
        fake_loss = adv_loss_fn(D_H_fake, label_fake) + \
                    adv_loss_fn(D_L_fake, label_fake)

        total_d_loss = (real_loss + fake_loss) / 2
        total_d_loss.backward()
        opt_d.step()

        running_g_loss += total_g_loss.item()
        running_d_loss += total_d_loss.item()

    return (running_g_loss / num_batches, running_d_loss / num_batches)


@torch.no_grad()
def validate_one_epoch(model: CycleGAN, dataloader, device: torch.device):
    """
    Run a validation epoch to compute average ssim, psnr, etc.
    (Matches the logic from _compute_metrics in your Lightning code.)
    Returns a dict of average metrics.
    """
    model.eval()

    ssim_scores, psnr_scores, lncc_scores = [], [], []
    css_scores, mi_scores, bc_scores, mae_scores = [], [], [], []

    for batch in dataloader:
        real_x = batch["x"].to(device)
        real_y = batch["y"].to(device)

        # Generate
        fake_y = model.generator_xy(real_x)
        fake_x = model.generator_yx(real_y)

        # For each sample in the batch, compute metrics
        for i in range(real_x.size(0)):
            # Denormalize from [-1,1] -> [0,1]
            real_y_np = (fake_y[i].detach().cpu().numpy() + 1) / 2
            fake_y_np = (real_y[i].detach().cpu().numpy() + 1) / 2

            # NOTE: The above lines might be swapped based on your original logic,
            # typically: real_y_np = denormalize(real_y[i]); fake_y_np = denormalize(fake_y[i])
            # but let's stay consistent with your function usage.
            # If you prefer the exact code:
            # real_y_np = denormalize(real_y[i]).cpu().numpy().squeeze()
            # fake_y_np = denormalize(fake_y[i]).cpu().numpy().squeeze()

            # Calculate the metrics
            from skimage.metrics import structural_similarity as ssim
            from skimage.metrics import peak_signal_noise_ratio as psnr
            from metrics import (
                normalized_cross_correlation,
                mutual_information,
                contrast_structure_similarity,
                bhattacharyya_coefficient,
                mean_absolute_error
            )
            ssim_scores.append(ssim(fake_y_np.squeeze(), real_y_np.squeeze(), data_range=1))
            psnr_scores.append(psnr(fake_y_np.squeeze(), real_y_np.squeeze(), data_range=1))
            lncc_scores.append(normalized_cross_correlation(fake_y_np, real_y_np))
            css_scores.append(contrast_structure_similarity(fake_y_np, real_y_np))
            mi_scores.append(mutual_information(fake_y_np, real_y_np))
            bc_scores.append(bhattacharyya_coefficient(fake_y_np, real_y_np))
            mae_scores.append(mean_absolute_error(fake_y_np, real_y_np))

    def average(values):
        return float(np.mean(values)) if len(values) > 0 else 0.0

    return {
        "val_ssim": average(ssim_scores),
        "val_psnr": average(psnr_scores),
        "val_lncc": average(lncc_scores),
        "val_css":  average(css_scores),
        "val_mi":   average(mi_scores),
        "val_bc":   average(bc_scores),
        "val_mae":  average(mae_scores),
    }


def main():
    # ------------------------
    #  Global configurations
    # ------------------------
    set_seed(2024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Hyperparameters
    BATCH_SIZE = 8
    WORKERS    = 4
    EPOCHS     = 300
    LR_G       = 3e-4
    LR_D       = 3e-3

    # Paths (adapt to your environment)
    base_dir      = "/data/ob_process/csv"
    data_dir_low  = "/data/ob_process/low_quality_images"
    data_dir_high = "/data/ob_process/high_quality_images"

    # -------------
    #  WandB init
    # -------------
    wandb.init(project="CycleGAN", name="training_run_regular_torch")

    # -------------
    #  DataModule
    # -------------
    data_module = ImageDataModule(
        batch_size=BATCH_SIZE,
        workers=WORKERS,
        train_csv=os.path.join(base_dir, "train.csv"),
        val_csv=os.path.join(base_dir, "validation.csv"),
        test_csv=os.path.join(base_dir, "test.csv"),
        data_dir_low=data_dir_low,
        data_dir_high=data_dir_high
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader   = data_module.val_dataloader()
    test_loader  = data_module.test_dataloader()

    # -------------
    #  Model setup
    # -------------
    model = CycleGAN(
        img_channels=1,
        n_residual_blocks=15,  # or however many
        lr_g=LR_G,
        lr_d=LR_D,
        lambda_cycle=12,
        lambda_id=1,
        lambda_adv=1,
        lambda_per=1
    ).to(device)

    # -------------
    #  Optimizers
    # -------------
    # Combine G_xy and G_yx params for generator optimizer:
    opt_g = Adam(
        itertools.chain(model.generator_xy.parameters(), model.generator_yx.parameters()),
        lr=model.lr_g,
        betas=(0.9, 0.9)
    )
    # Combine both discriminators:
    opt_d = Adam(
        itertools.chain(model.discriminator_x.parameters(), model.discriminator_y.parameters()),
        lr=model.lr_d,
        betas=(0.9, 0.9)
    )

    # -------------
    #  LR schedulers
    # -------------
    scheduler_g = StepLR(opt_g, step_size=100, gamma=0.5)
    scheduler_d = StepLR(opt_d, step_size=100, gamma=0.5)

    # -------------
    #  Training Loop
    # -------------
    best_val_ssim = -1.0
    for epoch in range(1, EPOCHS + 1):
        g_loss, d_loss = train_one_epoch(model, train_loader, opt_g, opt_d, device)
        val_metrics = validate_one_epoch(model, val_loader, device)

        # Step schedulers
        scheduler_g.step()
        scheduler_d.step()

        # Log to Weights & Biases
        wandb.log({
            "epoch": epoch,
            "train_loss_generator": g_loss,
            "train_loss_discriminator": d_loss,
            **val_metrics,  # merges val_ssim, val_psnr, etc. into the log
        })

        print(f"Epoch {epoch} | G_loss: {g_loss:.4f} | D_loss: {d_loss:.4f} "
              f"| val_SSIM: {val_metrics['val_ssim']:.4f} | val_PSNR: {val_metrics['val_psnr']:.4f}")

        # Save top-k (or best) checkpoints based on val_ssim
        if val_metrics["val_ssim"] > best_val_ssim:
            best_val_ssim = val_metrics["val_ssim"]
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = os.path.join("checkpoints", f"CycleGAN_best_{epoch:03d}.pth")
            torch.save({"state_dict": model.state_dict()}, checkpoint_path)
            print(f"  [*] Saved new best checkpoint to {checkpoint_path}")

    # -------------
    #  Test the model
    # -------------
    print("=== Running final test evaluation ===")
    test_metrics = validate_one_epoch(model, test_loader, device)
    print("Test metrics:", test_metrics)

    wandb.finish()


if __name__ == "__main__":
    main()
