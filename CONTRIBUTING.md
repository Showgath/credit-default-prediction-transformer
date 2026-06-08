# Contributing to Credit Default Prediction Transformer

Thank you for your interest in contributing! This document provides guidelines for contributing to this research project.

## 🤝 How to Contribute

### 1. Fork and Clone
```bash
git clone https://github.com/YourUsername/credit-default-prediction-transformer.git
cd credit-default-prediction-transformer
```

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-ensemble-model`
- `bugfix/fix-data-preprocessing`
- `docs/improve-setup-instructions`

### 3. Make Your Changes
- Write clean, readable code following PEP 8
- Add detailed comments for complex neural network logic
- Include docstrings for functions and classes
- Test your changes thoroughly

### 4. Commit Your Changes
```bash
git commit -m "Brief description of changes"
```

Use clear commit messages:
- ✅ Good: `feat: implement attention visualization`
- ✅ Good: `fix: correct positional encoding logic`
- ❌ Bad: `update code`

### 5. Push and Open a Pull Request
```bash
git push origin feature/your-feature-name
```

Then open a Pull Request with:
- Clear title describing the change
- Detailed description of what and why
- Reference to any related issues

---

## 📋 Before You Submit

- [ ] Code follows PEP 8 style guide
- [ ] Changes are tested and working
- [ ] Documentation is updated if needed
- [ ] No experimental data or models committed
- [ ] Notebook outputs are cleared before commit

---

## 🎯 Areas for Contribution

### High Priority
- [ ] Model interpretability and visualization
- [ ] Hyperparameter optimization
- [ ] Additional baseline comparisons
- [ ] Documentation improvements
- [ ] Bug fixes

### Medium Priority
- [ ] Performance optimizations
- [ ] Alternative architectures
- [ ] Attention mechanism analysis
- [ ] Code refactoring

### Research Ideas
- [ ] Ensemble methods combining Transformer + Random Forest
- [ ] Class imbalance techniques (SMOTE, focal loss)
- [ ] Cross-validation framework
- [ ] Model deployment strategies
- [ ] Real-world dataset evaluation

---

## 💻 Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook "IFTE0002 - Group Project.ipynb"
```

---

## 📝 Code Style Guidelines

### Python
- Follow PEP 8
- Use meaningful variable names
- Add docstrings following NumPy format

### Example Function:
```python
def create_temporal_sequence(customer_record: dict) -> torch.Tensor:
    """
    Convert customer record to temporal token sequence.
    
    Args:
        customer_record (dict): Dictionary containing customer financial data
    
    Returns:
        torch.Tensor: 11-token sequence [S1-S5, T1-T6]
    """
    # Implementation here
    pass
```

---

## 🐛 Reporting Issues

If you find a bug or have a suggestion:

1. Check existing issues first
2. Create a new issue with:
   - Clear title
   - Detailed description
   - Steps to reproduce (if applicable)
   - Expected vs. actual behavior
   - Environment details (Python version, PyTorch version, etc.)

---

## 📚 Documentation

When adding features:
- Update README.md if needed
- Add docstrings to new classes/functions
- Include usage examples
- Document model changes

---

## 🚀 Getting Help

- Review the [Technical Report](Finance_and_AI_Report.pdf)
- Check the notebook: `IFTE0002 - Group Project.ipynb`
- Read the main [README.md](README.md)
- Ask questions in Issues

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
