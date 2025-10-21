# Telugu Wav2Vec2.0 Fine-tuning and Ablation

This repository documents the **fine-tuning of Wav2Vec2.0 for Telugu Automatic Speech Recognition (ASR)**, along with a detailed **ablation study** exploring how different hyperparameters affect model performance.  
Training and experimentation were conducted using **NVIDIA A40 GPUs on RunPod**, ensuring efficient large-scale model execution.

---

## Overview

Speech recognition for Indian languages, especially **Telugu**, remains a challenging task due to limited resources and phonetic diversity.  
This project fine-tunes **Facebook’s Wav2Vec2.0 base model** using the **Telugu subset of Mozilla Common Voice**, and conducts systematic experiments to analyze:

- **Learning rate variations**
- **Hidden layer dropout rates**
- **Number of training epochs**
- **Optimizer Variants**

The goal is to determine how each hyperparameter influences **WER (Word Error Rate)**, **CER (Character Error Rate)**, and general model robustness.

---

##  Experimental Setup

### Hardware and Compute
- **Platform:** [RunPod.io](https://www.runpod.io)
- **GPU Used:** NVIDIA A40 (48 GB VRAM)
- **Runtime:** Ubuntu 22.04 + CUDA 11.8
- **Python Version:** 3.10+
- **Deep Learning Framework:** PyTorch (Nightly Build with CUDA 11.8)
All required dependencies are listed in `requirements.txt`.

---

## Repository Structure

```text
wav2vec2.0_finetuning_and_ablation/
├── src/
│   ├── wav2vec2.0_finetuning_and_ablation.ipynb
│   ├── learning_rate_ablation.ipynb
│   ├── hidden_dropout_ablation.ipynb
│   ├── epoch_ablation.ipynb
│   └── optimizer_ablation.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup Instructions

1. Clone the Repository
2. Create and Activate Virtual Environment
3. Install Requirements
pip install -r requirements.txt

Model Training Workflow

Each notebook inside src/ explores one hyperparameter dimension:

->wav2vec2.0_finetuning_and_ablation.ipynb- Core notebook for fine-tuning Wav2Vec2.0 on dataset and evaluating base performance before ablation.
->learning_rate_ablation.ipynb- Tests convergence at different LR values.
->hidden_dropout_ablation.ipynb- Analyzes regularization effects of varying dropout.
->epoch_ablation.ipynb- Studies model improvement and overfitting over multiple epochs.
->optimizer_ablation.ipynb- Compares the performance of different optimizers — Adam and Adafactor — for training efficiency and generalization.

Each experiment logs:

Training loss and evaluation loss
WER and CER
WandB experiment dashboard

## Evaluation Metrics

Model performance is evaluated on:
WER (Word Error Rate) → measures transcription accuracy
CER (Character Error Rate) → measures fine-grained error

Metrics are calculated using evaluate and jiwer.


