# VLM Logit-Based Uncertainty (Per-Token Confidence)

This repository demonstrates **logit-based uncertainty estimation** for a Vision-Language Model (VLM) by extracting **per-token probabilities (logprobs)** and **token entropy** from the model’s generation logits.  
Instead of trusting verbal confidence (e.g., “I am 90% sure”), we measure uncertainty directly from the model’s internal distribution during decoding.

## Motivation

VLMs (and LLMs) may produce confident-sounding answers even when they are wrong. A more reliable signal comes from the model’s **internal token probabilities**:

- **Low logprob** for key tokens suggests uncertainty.
- **High entropy** (spread probability across many tokens) suggests ambiguity and possible hallucination.
- **Top-k token alternatives** reveal “flicker” (e.g., cat vs dog) which is a strong hallucination/uncertainty indicator.

## Method Summary

Given an input image and prompt, we generate an answer while requesting the model’s per-step logits:

- `output_scores=True` returns logits for each generated token step.
- We compute:
  - **Token logprob**: `log p(token_t | context)`
  - **Token probability**: `p = exp(logprob)`
  - **Token entropy**: `H_t = - Σ p_i log p_i` (uncertainty of the distribution at step t)
  - **Top-k alternatives** (tokens + probabilities)

### Metrics

We report the following summary metrics per generated answer:

- **avg_logprob**: mean logprob across generated tokens (higher/closer to 0 = more confident)
- **min_logprob**: lowest logprob among generated tokens (the “weakest”/riskiest token)
- **avg_entropy**: mean entropy across generated tokens (higher = more uncertain)
- **max_entropy**: highest entropy across steps (often spikes at ambiguous object words)
- **internal_conf**: `exp(avg_logprob)` (geometric mean token probability; not calibrated correctness)

## Model & Environment

- Model: LLaVA 1.5 7B (`llava-hf/llava-1.5-7b-hf`)
- Inference: greedy decoding (`do_sample=False`) for stable token probability analysis
- Hardware: Google Colab (T4 GPU)
- Quantization: 4-bit (bitsandbytes) to fit 7B VLM on T4

## Environment Setup (Local Development)

### Prerequisites

- Python 3.11
- CUDA-compatible GPU
- Conda or Miniconda installed

### Installation Steps

1. **Create a new conda environment with Python 3.11:**

   ```bash
   conda create -n vlm python=3.11
   conda activate vlm
   ```

2. **Install PyTorch with CUDA support** (adjust CUDA version as needed):

   ```bash
   # For CUDA 11.8
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

   # For CUDA 12.1
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

   # For CPU only (not recommended for VLM inference)
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   ```

3. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook notebooks/01_logit_uncertainty_demo.ipynb
   ```

> **Note:** For Google Colab users, the notebook already includes installation commands. Simply enable GPU runtime and run the cells.

## How It Works (Brief)

During generation, at each decoding step the model outputs logits over the vocabulary:

1. logits → `log_softmax` → log-probabilities
2. chosen token logprob = confidence for the produced token
3. entropy of full distribution = uncertainty at that step
4. compare uncertainty across image conditions (clear vs blurred)

## Running the Notebook (Google Colab)

1. Open `notebooks/01_logit_uncertainty_demo.ipynb` in Colab:

   - In GitHub: open the notebook file → copy the URL
   - Go to Colab → **File → Open notebook → GitHub** → paste the URL (or search the repo)

2. Enable GPU:

   - **Runtime → Change runtime type → Hardware accelerator → GPU (T4)**

3. Run cells top-to-bottom:
   - Install dependencies
   - Load model (4-bit quantization)
   - Upload/select an image
   - Enter your prompt in the `user_text` variable
   - Run inference to print **Top-k token probabilities + entropy**

> Tip: If you get package conflicts, restart the runtime (**Runtime → Restart runtime**) and reinstall only:
> `transformers`, `accelerate`, `bitsandbytes` (avoid upgrading torch/pandas/pillow in Colab).
