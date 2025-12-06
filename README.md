# Dressing the Imagination: Text-to-Fashion Outfits with NeRA Adapters

> Official implementation of **“Dressing the Imagination: A Dataset for AI-Powered Translation of Text into Fashion Outfits and A Novel NeRA Adapter for Enhanced Feature Adaptation”**, accepted at **WACV 2026**.

<div align="center">

[![Project Page](https://img.shields.io/badge/%F0%9F%8C%8E-Project%20Website-purple)](https://candlelabai.github.io/WACV2026-FLORA-Dataset-NeRA-Adapter)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Dataset-orange)](https://huggingface.co/datasets/CandleLabAI/FLORA)
[![ArXiv](https://img.shields.io/badge/%F0%9F%93%96%20ArXiv-Paper-b31b1b)](https://arxiv.org/pdf/2411.13901)
[![WACV](https://img.shields.io/badge/%F0%9F%8C%B8%20WACV%202026-Paper-553C9A)](WACV_PAPER_URL_HERE)

</div>

---

## ✨ Highlights

- 📚 **New fashion dataset** for text-to-fashion outfit generation.
- 🧠 **NeRA adapter** for efficient feature adaptation.
- 🚀 End-to-end training and inference pipeline included.
- 🔧 **Model-agnostic adapter** – demonstrated with FLUX, but easily extendable to other architectures.

---

## 🧩 What is NeRA?

**NeRA (Nonlinear low-rank Expressive Representation Adapter)** is a novel parameter-efficient fine-tuning adapter inspired by Kolmogorov-Arnold Networks (KANs), replacing MLP-based transformations in methods like LoRA with learnable spline-based activations for superior modeling of complex, nonlinear semantic relationships.

> 🔁 Although this implementation demonstrates NeRA with the **Flux** model, NeRA is **model-agnostic** and can be integrated with *any* compatible architecture. The provided scripts serve as a reference and can be easily adapted to your preferred model.

---

## ⚙️ Installation & Setup

### 1. 🐍 Create a Virtual Environment

Ensure you have **Python 3.10+** installed.

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate
```

### 2. 📦 Install Dependencies

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

---

## 🏋️‍♀️ Training with NeRA

Update the config file [`training/flux_nera_config.yaml`](training/flux_nera_config.yaml) with your:
- Dataset paths
- Cache and output directories
- NeRA hyperparameters

Then run:

```bash
python train_nera_flux.py --config training/flux_nera_config.yaml
```

📦 Outputs:
- `adapter.pt` (learned adapter weights)
- `config.json` (model & adapter config)

---

## 🔍 Inference

To perform inference with trained NeRA point `infer_flux_Nera.py` to your adapter directory (weights + config), then run:

```bash
python infer_flux_Nera.py
```

---

📄 **YAML Configuration** includes:

- Pretrained model path
- Dataset root (images + CSV)
- NeRA hyperparameters (rank, alpha, target layers)
- Training params (batch size, LR, epochs)

---

## 📖 Citation

If you use this work in your research, please cite our paper:

```bibtex
@inproceedings{Deshmukh_2026_WACV,
  author    = {Deshmukh, Gayatri and De, Somsubhra and Sehgal, Chirag and Gupta, Jishu Sen and Mittal, Sparsh},   
  title     = {Dressing the Imagination: A Dataset for AI-Powered Translation of Text into Fashion Outfits and A Novel NeRA Adapter for Enhanced Feature Adaptation},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  month     = {March},
  year      = {2026},
}
```
