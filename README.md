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
- MLP
- CatBoost
