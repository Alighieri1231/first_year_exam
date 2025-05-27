# ASS-GAN & Baseline Backbones for Breast-Ultrasound Lesion Segmentation  
_First-Year Exam · Emilio José Ochoa Alva · University of Rochester · 2025-05-27_

![Python](https://img.shields.io/badge/Python_3.10‒3.12-blue?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch_2.4%2B-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning_2.4%2B-792ee5?style=flat-square&logo=pytorchlightning&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)

> **ASS-GAN** pairs *two complementary segmentation generators* (DeepLabV3+ and PSPNet) with a PatchGAN discriminator.  
> At each iteration the discriminator filters out noisy masks; only “real-looking” predictions are exchanged as pseudo-labels, giving robust semi-supervised learning even when **85 % of the masks are missing**.

---

## 🌟 Highlights

| Scenario | Method | Mean IoU (↑) | Gain vs. best backbone |
|----------|--------|--------------|------------------------|
| **100 % labels** | DeepLabV3+ | 0.654 | — |
| **15 % labels** | DeepLabV3+ | 0.421 | — |
| **15 % labels** | **ASS-GAN** | **0.630** | **+ 0.209** ( +50 %) |

*Numbers are on the FD-nl split, which removes patient-level duplicates to avoid leakage.*

---

## 🔧 Quick start

```bash
git clone https://github.com/Alighieri1231/first_year_exam.git
cd first_year_exam

# 🔹 Recommended – conda:
conda env create -n assgan -f environment.yaml
conda activate assgan

# 🔹 Alternative – pip:
pip install -r requirements.txt
