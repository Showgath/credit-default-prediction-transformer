# Small Transformer-based Model for Credit Card Default Prediction

This repository contains a PyTorch implementation of a **Small Transformer-based Model (Small Language Model - SLM)** designed for predicting credit card default. Unlike standard “flat” tabular models, this architecture treats a customer’s financial history as a sequence of 11 tokens, allowing for deep relational learning between static demographics and six months of payment behaviour.

## 🚀 Project Overview

The goal of this project is to evaluate whether a Transformer-based architecture can compete with, or outperform, traditional tree-based models on structured financial data for credit card default prediction.

### Key Finding

Although the **Random Forest** benchmark and the **Temporal Transformer (SLM)** reach a similar macro F1-score (≈ 0.68), the Transformer model achieves a **much higher default recall** (≈ 0.65 vs. 0.36). In a banking context, identifying nearly twice as many potential defaulters provides substantially greater business value and risk mitigation.

---

## 🧠 Model Architecture

The SLM is designed to handle heterogeneous tabular data by converting each record into a short sequence that the self-attention mechanism can process natively.

### 1. Tokenisation & Sequence Structure

Each customer record is converted into an **11-token** sequence:

- **Tokens S1–S5 (Static):**  
  Dedicated tokens for `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, and `AGE`.

- **Tokens T1–T6 (Temporal):**  
  Monthly snapshots from April (T1, oldest) to September (T6, most recent). Each temporal token encodes:
  - Repayment status (`PAY_*`) – indices shifted by +2 for non-negativity  
  - Bill amount (`BILL_AMT*`)  
  - Payment amount (`PAY_AMT*`)

- **Sequence order:**  
  `[S1, S2, S3, S4, S5, T1, T2, T3, T4, T5, T6]`

This design lets the model attend jointly over static context and temporal payment history.

### 2. Embedding Design (d_model = 64)

Every token is projected into a 64-dimensional space:

- **Static projections:**  
  - Numerical features (e.g. `LIMIT_BAL`, `AGE`) use dedicated `Linear(1 → 64)` layers.  
  - Categorical features (e.g. `SEX`, `EDUCATION`, `MARRIAGE`) use separate `Embedding` layers.

- **Temporal fusion:**  
  - Numerical components (bill and payment amounts) are stacked and passed through a shared `Linear(2 → 64)` layer.  
  - Repayment status (`PAY_*`) is embedded via `Embedding(…, 64)`.  
  - The two 64-D vectors are concatenated (128-D) and compressed back to 64-D via a fusion `Linear(128 → 64)` layer.

This allows the model to learn intra-month interactions before attending across the sequence.

### 3. Learned Positional Encoding

To represent the passage of time, a **learned positional embedding** is added **only** to the temporal tokens `T1…T6`. The static tokens `S1…S5` receive no positional encoding, since their order is arbitrary and not temporal.

---

## 🔧 Transformer Stack

The core encoder consists of **two stacked Transformer blocks**:

- **Multi-Head Self-Attention:**  
  4 attention heads (each with 16-dimensional subspace) capture different relational patterns, such as:
  - Month-to-month repayment trends  
  - Interactions between `LIMIT_BAL` and recent delays  

- **Pre-Norm & Residuals:**  
  LayerNorm is applied before attention and FFN (Pre-Norm) with residual connections around each sub-layer to stabilise training.

- **Feed-Forward Network (FFN):**  
  Two-layer MLP with ReLU activation (hidden dimension typically 64 or 128, depending on the tuned config).

---

## 🎯 Classification & Training

### Classification Head

After the Transformer blocks, the 11 contextualised token embeddings are aggregated using **mean pooling** (rather than a single CLS token). The pooled 64-D vector is passed through a 2-layer MLP:

- 64 → 32 (ReLU) → 1 (logit)

The final sigmoid of this logit gives the probability of default.

### Training Configuration

- **Loss:** `BCEWithLogitsLoss` with `pos_weight ≈ 3.5` to counter class imbalance.  
- **Optimizer:** Adam with learning rate `1e-3`.  
- **Regularisation & Stability:**
  - Dropout `p = 0.1` on embeddings, attention, and FFN sub-layers  
  - Gradient norm clipping at 1.0  

---

## 📊 Performance Benchmark

Evaluated on a held-out test set (derived from 30,000 client records):

| Metric         | Random Forest (Benchmark) | Temporal Transformer (SLM) |
|----------------|---------------------------|----------------------------|
| Accuracy       | 0.82                      | 0.74                       |
| Macro F1-score | 0.68                      | 0.68                       |
| Default Recall | 0.36                      | 0.65                       |

The Transformer sacrifices some overall accuracy in exchange for **much higher recall on defaulters**, which is often the preferred trade-off in real-world credit risk management.

---

## 📁 Repository Contents (suggested)

- `data/` – Data loading / preprocessing scripts (no raw data in repo)  
- `models/` – Transformer and Random Forest implementations  
- `notebooks/` – EDA, training, and evaluation notebooks  
- `scripts/` – Training and hyperparameter tuning entry points  
- `report/` – Final coursework report (PDF)
