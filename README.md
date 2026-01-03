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

---

## Experiment: Blur vs Uncertainty
We evaluate uncertainty on the **same prompt** using three images:
1. **Clear image**
2. **Slightly blurred image**
3. **Highly blurred image**

Hypothesis:
- Increasing blur reduces visual evidence → model uncertainty increases.


***Experiment 1(Clear Image)***

![pexels-pixabay-53435](https://github.com/user-attachments/assets/b53f4818-3c36-4690-a323-21b96504d162)

***Results***

<img width="397" height="139" alt="Screenshot 2026-01-03 132535" src="https://github.com/user-attachments/assets/b6c9d0bd-d75d-464d-84ce-987b12da5fce" />

***Experiment 2(Slightly blurred Image)***
![slighlty blur image](https://github.com/user-attachments/assets/6fd5249f-d6d4-4eb5-8b21-4c513fa0c824)


***Results***

<img width="446" height="145" alt="Screenshot 2026-01-03 133331" src="https://github.com/user-attachments/assets/a3918a4f-c0fc-4174-878e-62f2e097b4e0" />


***Experiment 3(Blurred Image)***
![blurred image](https://github.com/user-attachments/assets/d3788b22-1901-40ae-a57e-3d532a36a9ff)

***Results***

<img width="455" height="145" alt="Screenshot 2026-01-03 133653" src="https://github.com/user-attachments/assets/46288973-32ac-4eac-b412-ed4af88b7862" />


| Condition | Predicted token | Top1 prob | Top2 prob | Margin (Top1-Top2) | Entropy |
|---|---|---:|---:|---:|---:|
| Clear | Tree | 0.9694 | 0.0180 | 0.9514 | 0.198 |
| Slight blur | Tree | 0.9071 |  0.0481| 0.859 | 0.493 |
| Heavy blur | Tree | 0.8041 | 0.0821 | 0.722 |  0.822|

### Key Decisions from the Blur Experiment (Logit-Based Uncertainty)

- **Blur reliably increases uncertainty:** Token entropy grows monotonically (**0.198 → 0.493 → 0.822**), indicating the model’s next-token distribution becomes progressively less peaked as visual evidence degrades.

- **Stable label ≠ stable confidence:** The model outputs the same label (**Tree**) for all conditions, yet **Top-1 probability falls** (**0.969 → 0.907 → 0.804**). This shows logit-based signals can flag reduced reliability even when the predicted answer does not change.

- **Competition between alternatives increases:** The **runner-up probability rises** (**0.018 → 0.048 → 0.082**) while the **Top1–Top2 margin shrinks** (**0.951 → 0.859 → 0.722**), meaning the model is allocating more probability mass to competing tokens (higher ambiguity / higher error risk).

- **Actionable trust decision:** Use **entropy + margin** as a quality gate:
  - **Low-trust / warning** when entropy is high and margin is low (e.g., heavy blur: **entropy=0.822**, **margin=0.722**).
  - **High-trust** when entropy is low and margin is high (e.g., clear image: **entropy=0.198**, **margin=0.951**).


