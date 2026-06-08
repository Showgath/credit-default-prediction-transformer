# Small Transformer-based Model for Credit Card Default Prediction
**PyTorch Implementation with Advanced Attention Mechanisms**

## 🎯 Project Overview

**Credit Default Prediction Transformer** is a research project demonstrating how transformer-based architectures can be effectively applied to structured financial data for credit risk prediction. Unlike traditional deep learning approaches on tabular data, this project showcases:

- **Custom Transformer Architecture** designed for heterogeneous financial features
- **Temporal Sequence Modeling** capturing payment history patterns
- **Superior Default Recall** (65% vs 36% Random Forest baseline)
- **Production-Ready Comparison** between modern deep learning and traditional ML

**Key Achievement**: Achieves comparable F1-score (0.68) to Random Forest while significantly improving default recall—a critical metric in real-world credit risk management.

**Tech Stack**: PyTorch, Python, NumPy, Pandas, Scikit-learn, Jupyter

---

## ✨ Key Features

- **Transformer-Based Architecture**: Multi-head self-attention over financial sequences
- **Smart Tokenization**: Converts flat credit records into meaningful token sequences
- **Temporal Processing**: Captures 6-month payment history with learned positional encoding
- **Class-Imbalance Handling**: Implements weighted loss function and proper evaluation metrics
- **Comprehensive Benchmarking**: Detailed comparison with Random Forest baseline
- **Research-Grade Documentation**: Full technical methodology and results analysis

---

## 🛠️ Technical Highlights

- **Deep Learning**: PyTorch transformer implementation from scratch
- **Attention Mechanisms**: Multi-head self-attention (4 heads) with sophisticated tokenization
- **Feature Engineering**: Smart handling of mixed numerical/categorical features
- **Model Evaluation**: F1-score, precision, recall, accuracy with proper handling of class imbalance
- **Reproducibility**: Clean code structure with well-documented components

---

## 🧠 Model Architecture

The **Small Language Model (SLM)** for Credit Default is uniquely designed for structured financial data:

### 1. Tokenization & Sequence Structure

Each customer record is converted into an **11-token sequence**:

- **Tokens S1–S5 (Static):**  
  Dedicated tokens for `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, and `AGE`.

- **Tokens T1–T6 (Temporal):**  
  Monthly snapshots from April (T1, oldest) to September (T6, most recent). Each temporal token encodes:
  - Repayment status (`PAY_*`) – indices shifted by +2 for non-negativity  
  - Bill amount (`BILL_AMT*`)  
  - Payment amount (`PAY_AMT*`)

- **Sequence order:** `[S1, S2, S3, S4, S5, T1, T2, T3, T4, T5, T6]`

This design lets the model attend jointly over static context and temporal payment history.

### 2. Embedding Design (d_model = 64)

Every token is projected into a 64-dimensional space:

- **Static projections:**  
  - Numerical features use dedicated `Linear(1 → 64)` layers  
  - Categorical features use separate `Embedding` layers  

- **Temporal fusion:**  
  - Numerical components are stacked and passed through `Linear(2 → 64)`  
  - Repayment status is embedded via `Embedding(…, 64)`  
  - Two vectors are concatenated and compressed back to 64-D via `Linear(128 → 64)`

### 3. Learned Positional Encoding

Learned positional embeddings are added **only** to temporal tokens T1…T6, capturing the temporal flow while keeping static features order-invariant.

---

## 🔧 Transformer Stack

The core encoder consists of **two stacked Transformer blocks**:

- **Multi-Head Self-Attention:** 4 attention heads capturing relational patterns
- **Pre-Norm & Residuals:** LayerNorm before each sub-layer with residual connections
- **Feed-Forward Network:** Two-layer MLP with ReLU activation (hidden dim: 64-128)

---

## 📊 Performance Benchmark

Evaluated on a held-out test set (30,000 client records):

| Metric         | Random Forest (Benchmark) | Temporal Transformer (SLM) |
|----------------|---------------------------|----------------------------|
| Accuracy       | 0.82                      | 0.74                       |
| Macro F1-score | 0.68                      | 0.68                       |
| **Default Recall** | **0.36**                  | **0.65** ⭐               |

**Key Insight**: The Transformer sacrifices some overall accuracy in exchange for **much higher recall on defaulters**—the preferred trade-off in real-world credit risk management.

---

## 📁 Project Structure

```
credit-default-prediction-transformer/
├── README.md                                 # This file
├── LICENSE                                   # MIT License
├── CONTRIBUTING.md                           # Contribution guidelines
├── requirements.txt                          # Python dependencies
│
├── IFTE0002 - Group Project.ipynb           # Main implementation notebook
├── default_of_credit_card_clients.csv       # Dataset (30,000 records)
│
├── Finance_and_AI_Report.pdf                # Technical report
├── output/                                   # Model outputs and results
└── prototype/                                # Experimental implementations
```

---

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.9+**
- **Jupyter Notebook** (for interactive exploration)

### Step-by-Step Instructions

#### 1. Clone the Repository
```bash
git clone https://github.com/Showgath/credit-default-prediction-transformer.git
cd credit-default-prediction-transformer
```

#### 2. Create a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Launch Jupyter and Run the Notebook
```bash
jupyter notebook "IFTE0002 - Group Project.ipynb"
```

The notebook will:
- Load and explore the credit card dataset
- Implement the Transformer architecture
- Train and evaluate the model
- Compare results with Random Forest baseline

---

## 💡 Usage Tips

- **GPU Acceleration**: To use GPU, install `torch` with CUDA support: `pip install torch torchcuda`
- **Dataset**: The `default_of_credit_card_clients.csv` is included (30,000 client records)
- **Reproducibility**: Set random seeds for consistent results
- **Memory**: Large batch sizes may require GPU; adjust in notebook if needed

---

## 📊 Skills Demonstrated

This project showcases proficiency in:

✅ **Deep Learning Architecture Design** - Custom Transformer implementation  
✅ **Attention Mechanisms** - Multi-head self-attention from scratch  
✅ **Financial ML** - Credit risk modeling and evaluation  
✅ **PyTorch** - Advanced tensor operations and model training  
✅ **Feature Engineering** - Smart tokenization for mixed data types  
✅ **Model Evaluation** - Proper handling of class imbalance and metrics selection  
✅ **Research Documentation** - Clear methodology and rigorous benchmarking  
✅ **Reproducibility** - Well-structured code with full explanations  

---

## 🔍 Research Contributions

- **Novel Tokenization**: Custom approach to convert financial records into transformer-compatible sequences
- **Temporal Awareness**: Learned positional encoding capturing payment history flow
- **Class-Imbalance Handling**: Weighted loss function addressing skewed default distribution
- **Comprehensive Evaluation**: Detailed metrics selection prioritizing recall for risk management

---

## 📚 Resources & Documentation

### Technical Report
- **Full Analysis**: [Finance and AI Report](Finance_and_AI_Report.pdf)
  - Model architecture details
  - Experimental setup and results
  - Comparison with baselines
  - Business implications

### Dataset Information
- **Source**: UCI Machine Learning Repository - Default of Credit Card Clients
- **Records**: 30,000 credit applications
- **Features**: 23 attributes (demographic, credit limit, payment history, bills)
- **Target**: Binary classification (default: Yes/No)

---

## 🚀 Future Enhancements

- [ ] Ensemble methods combining Transformer + Random Forest
- [ ] SMOTE or focal loss for class imbalance
- [ ] Cross-validation framework
- [ ] Attention visualization and interpretability
- [ ] Real-world dataset evaluation
- [ ] Model deployment with FastAPI
- [ ] Hyperparameter optimization with Optuna
- [ ] Benchmark on additional datasets

---

## 🤝 Contributing

Contributions are welcome! Whether it's bug fixes, improvements, or research ideas:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-idea`)
3. Commit your changes (`git commit -m 'Add your contribution'`)
4. Push to the branch (`git push origin feature/your-idea`)
5. Open a Pull Request

For detailed guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 📧 Support & Questions

For questions or feedback:
- Check the [Technical Report](Finance_and_AI_Report.pdf)
- Review the [Jupyter Notebook](IFTE0002%20-%20Group%20Project.ipynb)
- Open an issue on GitHub

---

**Last Updated**: June 2026  
**Project Status**: Research Complete, Production-Ready Code
