# Small Transformer-based Model for Credit Card Default Prediction

This repository contains a PyTorch implementation of a **Small Transformer-based Model (Small Language Model - SLM)** designed for predicting credit card default. Unlike standard "flat" tabular models, this architecture treats a customer's financial history as a sequence of 11 tokens, allowing for deep relational learning between static demographics and 6 months of payment behavior.

## 🚀 Project Overview
The goal of this project was to evaluate whether a Transformer-based architecture could outperform traditional tree-based models on structured financial data.

### Key Finding
While the **Random Forest** benchmark and the **Temporal Transformer** achieved a similar Macro F1-score (~0.68), the Transformer model demonstrated a massive advantage in **Default Recall (0.65 vs. 0.36)**. In a banking context, the ability to catch 65% of defaulters (compared to 36%) provides significantly higher business value and risk mitigation.



## Model Architecture
The SLM is designed to handle heterogeneous tabular data by transforming records into a sequential format that the self-attention mechanism can process natively.
### 1. Tokenization & Sequence Structure
Each customer record is converted into a **11-token sequence**:
- **Tokens S1–S5 (Static):** Dedicated tokens for 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', and 'AGE'.
- **Tokens T1–T6 (Temporal):** Monthly snapshots from April (T1) to September (T6). Each token encodes:
  - Repayment Status (PAY_*) — Indices shifted by +2 for non-negativity.
  - Bill Amount (BILL_AMT*)
  - Payment Amount (PAY_AMT*)
**Sequence Order:** $[S_1, S_2, S_3, S_4, S_5, T_1, T_2, T_3, T_4, T_5, T_6]$

### 2. Embedding Design ($d_{model} = 64$)
Every token is projected into a 64-dimensional representational space:
- **Static Projections:** Independent Linear layers for scalars and Embedding layers for categories.
- **Temporal Fusion:** Numerical components (Bill/Pay amounts) and categorical components (Status) are concatenated into a 128-D vector and compressed to 64-D via a fusion layer. This allows the model to learn interactions within a single month before attending to the sequence.

### 3. Learned Positional Encoding
To respect the passage of time, a learned positional embedding is added to the temporal tokens ($T_1 \dots T_6$) only. The static tokens ($S_1 \dots S_5$) receive no positional encoding, as their relative order does not carry temporal significance.



## The Transformer Stack
The core engine consists of **two stacked Transformer blocks**:
- **Multi-Head Self-Attention:** 4 attention heads ($d_k = 16$) allow the model to parallelize focus (e.g., one head tracking month-to-month trends, another relating LIMIT_BAL to recent delays).
- **Pre-Norm & Residuals:** Employs the Pre-Norm configuration (LayerNorm applied before attention/FFN) to improve gradient flow and training stability.
- **Feed-Forward Network (FFN):** A two-layer MLP with a hidden dimension of 128 and ReLU activation.


## Classification & Training
### Classification Head
Instead of using only the CLS token, this model utilizes Mean Pooling across all 11 contextualized tokens. This summary vector $z$ is passed through a final 2-layer MLP (64 $\to$ 32 $\to$ 1) to produce the default logit.

### Training Configuration
**Loss Function:** BCEWithLogitsLoss with a pos_weight of ~3.5 to combat class imbalance.
**Optimizer:** Adam ($LR = 1 \times 10^{-3}$).
**Stability:** Gradient Norm Clipping set at 1.0.
**Regularization:** Dropout (0.1) applied to embeddings, attention, and FFN sub-layers.



## Performance Benchmark
The following results were achieved on a test set of 30,000 records:

| Metric | Random Forest (Benchmark) | Temporal Transformer (SLM) |
| :--- | :---: | :---: |
| **Accuracy** | **0.82** | 0.74 |
| **Macro F1-Score** | 0.68 | 0.68 |
| **Default Recall** | 0.36 | **0.65** |

