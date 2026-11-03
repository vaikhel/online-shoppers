# Online Shoppers Purchasing Intention

Predicting whether a user will make a purchase during an online session.

## Dataset
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- Instances: 12,330
- Target: `Revenue` (binary)
- Class balance: ~84.5% False / 15.5% True

## Project Structure

```
online-shoppers/
├── data/
├── notebooks/
│   ├── eda.ipynb
│   └── modeling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── requirements.txt
└── README.md
```

## Models
- Logistic Regression
- Decision Tree
- MLP (PyTorch)
- CatBoost

## Results

CatBoost showed the best performance.

| Model              | Accuracy | F1-Score | ROC-AUC |
|--------------------|----------|----------|---------|
| CatBoost (test)    | 0.89     | 0.67     | 0.92    |

## Key Findings
- CatBoost achieved the highest F1-score and strong ROC-AUC
- Cross-validation results were consistent with the test set
- Class imbalance was handled with `class_weight` / `auto_class_weights`

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook
```