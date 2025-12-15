# XAI Consistency Analysis: Credit Card Fraud Detection
Comprehensive comparison of Explainable AI methods (SHAP, LIME, DALEX) for fraud detection models

> **Comprehensive comparison of Explainable AI methods (SHAP, LIME, DALEX) for fraud detection models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Complete](https://img.shields.io/badge/status-complete-success.svg)]()

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Overview

This project investigates **whether different XAI methods provide consistent explanations** for credit card fraud detection models. We compare:

- **SHAP**
- **LIME**
- **DALEX**
- **RF Native** 

### Research Question

**Do different XAI methods agree on what makes a transaction fraudulent?**

---

## Key Findings

### High Consistency: SHAP vs RF Native

```
Spearman ρ = 0.943 (p < 0.001)
Jaccard Similarity = 0.765
Common Features: 13/15
```

**Conclusion:** SHAP and RF Native see the **same fraud patterns**. RF Native can serve as a fast approximation of SHAP.

### Good Consistency: SHAP vs LIME (Local)

```
Jaccard Similarity: 0.25 - 1.00
Direction Agreement: 100%
Best case: 5/5 features match (perfect agreement)
```

**Conclusion:** LIME agrees with SHAP on **direction of impact** (100% of the time). Agreement strongest for clear-cut fraud cases.

### Poor Consistency: DALEX

```
Spearman ρ = 0.064 (no correlation)
```

**Conclusion:** DALEX struggles with extreme class imbalance (0.17% fraud). Requires special handling or larger samples.

### Consensus Fraud Indicators

All methods agree these features are critical:

1. **V14** - Most predictive (dominant across all methods)
2. **V10** - Second most important
3. **V4, V12, V17** - Strong supporting indicators
4. **V3, V11, V16** - Additional signals

---

## Dataset

**Credit Card Fraud Detection Dataset**

- **Source:** [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Features:** 30 (V1-V28 PCA-transformed, Time, Amount)
- **Target:** Class (0=Legitimate, 1=Fraud)
- **Imbalance:** 0.17% fraud (492 fraudulent transactions)

**Key Characteristics:**
- Highly imbalanced (typical of real-world fraud data)
- PCA-transformed features (privacy protection)
- European cardholders, September 2013

---

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Setup

```bash
# Clone repository
git clone https://github.com/misiakfilip/xai-consistency-analysis.git
cd xai-consistency-analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
imbalanced-learn>=0.11.0
shap>=0.43.0
lime>=0.2.0.1
dalex>=1.7.0
matplotlib>=3.7.0
seaborn>=0.12.0
kagglehub>=0.1.0
```

---

### Quick Start

```bash
# Run full analysis
python xai-consistency-analysis.py
```

---

## Project Structure

```
xai-consistency-analysis/
│
├── data/
│   └── creditcard.csv           # Dataset (download separately)
│
├── notebooks/
│   └── xai-consistency-analysis.ipynb   # jupyter notebook
│
├── src/
│   └── xai-consistency-analysis.py # python script
│
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── LICENSE                      # MIT License
```

---

## Results

### Model Performance

| Model | ROC-AUC | F1-Score (Fraud) | Precision | Recall |
|-------|---------|------------------|-----------|--------|
| **Random Forest** | **0.9834** | **0.7778** | 0.71 | 0.86 |
| XGBoost | 0.9717 | 0.7167 | 0.61 | 0.88 |

### Global Consistency

| Comparison | Spearman ρ | Jaccard | Common (Top 15) | Status |
|------------|-----------|---------|-----------------|--------|
| **SHAP vs RF Native** | **0.943** | 0.765 | 13/15 | Excellent |
| SHAP vs DALEX | 0.064 | 0.364 | 8/15 | Poor |
| DALEX vs RF Native | 0.013 | 0.304 | 7/15 | Poor |

### Local Consistency (SHAP vs LIME)

| Instance Type | Jaccard | Direction Agreement | Status |
|---------------|---------|---------------------|--------|
| Clear Fraud (low amount) | **1.000** | **5/5 (100%)** | Perfect |
| Fraud (high amount) | 0.250 | 2/2 (100%) | Good |
| True Legitimate | 0.429 | 3/3 (100%) | Good |
| False Positive | 0.429 | 3/3 (100%) | Good |

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{xai-consistency-analysis,
  author = {[Filip Misiak]},
  title = {XAI Consistency Analysis: Credit Card Fraud Detection},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/misiakfilip/xai-consistency-analysis}
}
```
---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Areas for contribution:**
- Improved DALEX implementation for imbalanced data
- Additional XAI methods (Integrated Gradients, Attention)
- Other fraud detection datasets
- Production deployment examples

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Dataset:** [Machine Learning Group - ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **SHAP:** [Scott Lundberg](https://github.com/slundberg/shap)
- **LIME:** [Marco Tulio Ribeiro](https://github.com/marcotcr/lime)
- **DALEX:** [Przemyslaw Biecek](https://github.com/ModelOriented/DALEX)

---

## Contact

**Author:** Filip Misiak 

**Email:** filip.misiak11@gmail.com

**GitHub:** [@misiakfilip](https://github.com/misiakfilip)

---

## Related Projects

- [SHAP Official Repository](https://github.com/slundberg/shap)
- [LIME Official Repository](https://github.com/marcotcr/lime)
- [DALEX Official Repository](https://github.com/ModelOriented/DALEX)
  
---

## Project Stats

![GitHub stars](https://img.shields.io/github/stars/misiakfilip/xai-consistency-analysis?style=social)
![GitHub forks](https://img.shields.io/github/forks/misiakfilip/xai-consistency-analysis?style=social)
![GitHub issues](https://img.shields.io/github/issues/misiakfilip/xai-consistency-analysis)
![GitHub last commit](https://img.shields.io/github/last-commit/misiakfilip/xai-consistency-analysis)

---

** If you find this project useful, please consider giving it a star!**
