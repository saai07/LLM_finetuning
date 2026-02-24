# 🦙 LLM Fine-Tuning — Llama 3.2 3B ASCII Cats (LoRA)

Fine-tune **Meta Llama 3.2-3B** with **LoRA** to generate ASCII art cats using [Unsloth](https://github.com/unslothai/unsloth) and Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saai07/LLM_finetuning/blob/main/llama-3.1-8B-ascii-cats-lora_finetuning.ipynb)

---

## 📌 Overview

| Item | Detail |
|---|---|
| **Base Model** | [`meta-llama/llama-3.2-3B`](https://huggingface.co/meta-llama/Llama-3.2-3B) |
| **Fine-tuning Method** | LoRA (Low-Rank Adaptation) via [PEFT](https://github.com/huggingface/peft) |
| **Dataset** | [`pookie3000/ascii-cats`](https://huggingface.co/datasets/pookie3000/ascii-cats) (201 samples) |
| **Framework** | [Unsloth](https://github.com/unslothai/unsloth) + HuggingFace TRL `SFTTrainer` |
| **Hardware** | Google Colab — NVIDIA Tesla T4 (free tier) |
| **Trained Adapter** | [`sai237/llama-3.1-8B-ascii-cats-lora`](https://huggingface.co/sai237/llama-3.1-8B-ascii-cats-lora) |

---

## 🗂️ Repository Structure

```
LLM_finetuning/
├── README.md
└── llama-3.1-8B-ascii-cats-lora_finetuning.ipynb   # Full training + inference notebook
```

---

## 🚀 Quick Start

### 1. Open the notebook

Click the **"Open In Colab"** badge above, or run locally with Jupyter:

```bash
git clone https://github.com/saai07/LLM_finetuning.git
cd LLM_finetuning
jupyter notebook llama-3.1-8B-ascii-cats-lora_finetuning.ipynb
```

### 2. Install dependencies

The first cell installs everything automatically:

```bash
pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 trl triton
pip install --no-deps cut_cross_entropy unsloth_zoo
pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
pip install --no-deps unsloth
```

### 3. Set your Hugging Face token

The notebook uses `google.colab.userdata` to read a secret named **`HF_ACCESS_TOKEN`**.  
Create a [Hugging Face access token](https://huggingface.co/settings/tokens) with **read** access (and **write** if you want to push the adapter).

---

## 🔧 Training Configuration

| Hyperparameter | Value |
|---|---|
| LoRA rank (`r`) | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Max sequence length | 2048 |
| Batch size (per device) | 2 |
| Gradient accumulation steps | 4 |
| Effective batch size | 8 |
| Epochs | 5 |
| Learning rate | 2e-4 |
| LR scheduler | Linear |
| Optimizer | AdamW 8-bit |
| Weight decay | 0.01 |
| Precision | FP16 |
| Trainable params | 24,313,856 / 3,237,063,680 (**0.75 %**) |
| Total training steps | 130 |
| Training time | ~2 min 15 s (T4 GPU) |

---

## 📊 Training Results

Training loss drops from **~4.2** to **~0.74** over 130 steps (5 epochs), showing strong convergence on this small dataset.

---

## 🎨 Sample Outputs

After fine-tuning, the model generates ASCII cat art from an empty prompt:

```
  /\_/\  (
 ( ~.~ ) _)
   "/"  (
 ( | | )
(__d b__)
```

```
  /\ ___ /\
 (  >   <  )
  \  >#<  /
  /       \
 /         \
 |         |
  \       /
  /// /// --
```

---

## 💾 Saving & Sharing

The trained LoRA adapter is pushed to Hugging Face Hub:

```python
model.push_to_hub("sai237/llama-3.1-8B-ascii-cats-lora", tokenizer, token=HF_TOKEN)
```

Optionally export to GGUF (quantized) format:

```python
model.push_to_hub_gguf("sai237/llama-3.1-8B-ascii-cats-lora", tokenizer, quantization_method="q4_0", token=HF_TOKEN)
```

---

## 📚 Key Dependencies

- [Unsloth](https://github.com/unslothai/unsloth) — 2× faster fine-tuning
- [TRL](https://github.com/huggingface/trl) — `SFTTrainer` for supervised fine-tuning
- [PEFT](https://github.com/huggingface/peft) — LoRA adapter support
- [Transformers](https://github.com/huggingface/transformers) — Model & tokenizer
- [Datasets](https://github.com/huggingface/datasets) — Data loading

---

## 🙏 Acknowledgements

- **Meta AI** for the [Llama 3.2](https://ai.meta.com/llama/) model family
- **Unsloth AI** for making fine-tuning fast and memory-efficient
- **pookie3000** for the [ascii-cats](https://huggingface.co/datasets/pookie3000/ascii-cats) dataset

---

## 📄 License

This project is provided as-is for educational purposes.
