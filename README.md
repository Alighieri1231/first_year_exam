# First Year Exam – Semi-/Fully-Supervised Segmentation with Lightning ⚡️

[![python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-blue?logo=python&logoColor=white)](https://python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![lightning](https://img.shields.io/badge/Lightning-2.2+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai)
[![license](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end **PyTorch Lightning** project for breast-ultrasound lesion segmentation that supports

| Mode | Script | Config |
|------|--------|--------|
| **Fully-supervised** single-network training | `train_lightning_seg.py` | `configs/default_config_train.yaml` |
| **ASS-GAN** asymmetric semi-supervised training | `train_lightning_assgan.py` | `configs/assgan_config_train.yaml` |
| **Sweeps / HPO** (Ray Tune or W&B) | `train_lightning_seg_sweep2.py` | `configs/sweep_config.yaml` |

---

## Quick start 🚀

```bash
# 1. clone & enter
git clone https://github.com/<your-handle>/first_year_exam.git
cd first_year_exam

# 2. set up environment (recommended)
conda env create -n first_exam -f environment.yaml
conda activate first_exam
#  or  pip install -r requirements.txt

# 3. run a toy experiment (supervised)
python src/train_lightning_seg.py --config configs/default_config_train.yaml trainer.max_epochs=3

# 4. run semi-supervised ASS-GAN
python src/train_lightning_assgan.py --config configs/assgan_config_train.yaml
```

---

## Project layout

```
first_year_exam
├── configs/
│   ├── assgan_config_train.yaml   # semi-supervised defaults
│   ├── default_config_train.yaml  # fully-supervised defaults
│   └── sweep_config.yaml          # sweep template for HPO
├── src/
│   ├── callbacks/                 # 🧩 custom Lightning callbacks
│   ├── datamodules/               # 📦 LightningDataModule implementations
│   ├── models/                    # 🧠 segmentation nets & ASS-GAN
│   ├── utils/                     # 🔧 misc helpers (logging, metrics, viz)
│   ├── vendor/                    # 3rd-party code vendored for stability
│   ├── __init__.py
│   ├── train_lightning_assgan.py
│   ├── train_lightning_seg.py
│   └── train_lightning_seg_sweep2.py
├── data/                          # (ignored)  place raw / processed datasets here
├── environment.yaml               # conda spec
├── requirements.txt               # pip spec
└── README.md
```

---

## Key features ✨

* **Lightning CLI integration** – every script subclasses `LightningCLI`, so any argument
  can be overridden via the command line. Example:

  ```bash
  python src/train_lightning_seg.py         --config configs/default_config_train.yaml         --trainer.devices=4         --model.optimizer.lr=3e-4
  ```

* **Asymmetric Semi-Supervised GAN (ASS-GAN)** – two heterogeneous generators + one
  PatchGAN discriminator with confidence threshold γ for pseudo-label exchange.

* **Callbacks ready-to-go** – early-stopping, model-checkpoint, rich W&B / TensorBoard
  logging, LR-monitor, and experiment reproducibility seed.

* **Sweep script** – single command hyper-parameter searches with Ray Tune or W&B
  Sweeps:

  ```bash
  python src/train_lightning_seg_sweep2.py --config configs/sweep_config.yaml
  ```

---

## Datasets 📂

`datamodules/` already supports:

* **DBUI / SPDBUI / ADBUI / SDBUI** – breast-ultrasound datasets
* Automatic download (where licences allow), train/val/test split, patching, &
  on-the-fly augmentations.

Put your data under `data/` (or point `data.root_dir=<path>` in the YAML).

---

## Training & evaluation 🧐

| Command | Description |
|---------|-------------|
| `python src/train_lightning_seg.py --config …` | supervised single-network run |
| `python src/train_lightning_assgan.py --config …` | semi-supervised ASS-GAN |
| `python src/train_lightning_seg_sweep2.py --config …` | sweep multiple runs |
| `python -m src.utils.eval --ckpt <ckpt_path> --split test` | offline evaluation on saved checkpoints |

---

## Reproducing the paper numbers

```bash
# Semi-supervised (15 % labels) on SDBUI
python src/train_lightning_assgan.py        --config configs/assgan_config_train.yaml        data.label_fraction=0.15
```

Metrics will be saved to `lightning_logs/` and, if enabled, streamed to W&B.

---

## Citation

If you find this repo useful, please cite the accompanying manuscript:

```bibtex
@article{Ochoa2025ASSGAN,
  title   = {ASS-GAN: Asymmetric Semi-Supervised GAN for Breast Ultrasound Image Segmentation},
  author  = {Emilio J. Ochoa Alva and collaborators},
  journal = {Neurocomputing},
  year    = {2025}
}
```

---

## License

This project is distributed under the **MIT license** – see `LICENSE` for details.
