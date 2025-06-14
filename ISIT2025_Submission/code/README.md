# ISIT 2025 Competition – Wireless Wanderlust (D2I) – Code README

## Overview

This folder contains the code (or "src") for reproducing our model training, inference, and evaluation for the IEEE ISIT 2025 Data‑to‑Information (D2I) Competition ("Wireless Wanderlust"). The code is ready for Ubuntu 22.04 + CUDA 12.

## Reproducing the Environment

We provide a "requirements.txt" (or "environment.yml") (located in this folder) that lists all dependencies (for example, numpy, pandas, torch, torchvision, etc.) so that you can reproduce our environment. (If you prefer, you can also use "environment.yml" (for example, for conda) – please see the file for details.)

To reproduce our environment (using pip), please run:

  pip install –r requirements.txt

(If you use conda, please run "conda env create –f environment.yml" (or equivalent) instead.)

## Running the Code

### Training

To train our model (for example, for Task 1 (Vanilla Localization)), please run:

  python train.py ––task vanilla ––data_dir /path/to/data ––output_dir /path/to/output

(Adjust "––task" (or "––data_dir" and "––output_dir") as needed.)

### Inference

To run inference (for example, on a held‑out test set), please run:

  python inference.py ––task vanilla ––model /path/to/model ––data_dir /path/to/test ––output_dir /path/to/output

(Adjust "––task", "––model", "––data_dir", and "––output_dir" as needed.)

### Evaluation

To evaluate (for example, compute MAE, R90, and the combined loss (ℒ)), please run:

  python evaluate.py ––task vanilla ––pred /path/to/predictions ––gt /path/to/ground_truth ––output_dir /path/to/output

(Adjust "––task", "––pred", "––gt", and "––output_dir" as needed.)

## Notes

• The code (and "requirements.txt" (or "environment.yml")) is ready for Ubuntu 22.04 + CUDA 12.  
• (If you have any questions or issues, please contact us (or refer to our "presentation" folder (or "presentation/README.md" for further details).) 