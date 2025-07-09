# CART Machine Learning Project

This project explores Classification and Regression Tree (CART) algorithms using Python's Scikit-Learn library. It implements both classification and regression modeling using decision trees and random forests.

![CART Algorithm Exploration and Evaluation - visual selection](https://github.com/user-attachments/assets/92dac774-e27b-4c76-a531-b147f4ce46c5)


## 👥 Authors

- Ahmad Mukhtar
- Muhammad Abubakar Tahir
- Muhammad Ehtesham
- Nobukhosi Sibanda

## 🎯 Project Overview

The project demonstrates the implementation and comparison of different CART algorithms on two distinct datasets:

### Classification Task
- **Dataset**: Handwritten Digits Dataset (sklearn)
- **Models**: 
  - DecisionTreeClassifier
  - RandomForestClassifier
- **Metrics**: Confusion Matrix, Accuracy Score

### Regression Task
- **Dataset**: California Housing Dataset
- **Models**:
  - DecisionTreeRegressor
  - RandomForestRegressor
- **Metrics**: Mean Squared Error (MSE), R² Score

## 📊 Results

### Classification Results (Digits Dataset)

| Model | Accuracy |
|-------|----------|
| Default Decision Tree | 0.8426 |
| Default Random Forest | 0.9759 |
| Tuned Decision Tree | 0.8574 |
| Tuned Random Forest | 0.9685 |

### Regression Results (California Housing)

| Model | MSE ↓ | R² ↑ |
|-------|-------|------|
| Default Decision Tree | 0.5280 | 0.5977 |
| Default Random Forest | 0.2565 | 0.8046 |
| Tuned Decision Tree | 0.4060 | 0.6906 |
| Tuned Random Forest | 0.2569 | 0.8043 |

## 🛠️ Implementation Details

### Libraries Used
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
```

### Model Tuning
- **Decision Trees**: Optimized using GridSearchCV with parameters:
  - max_depth: [5, 10, 20]
  - min_samples_split: [2, 5, 10]

- **Random Forests**: Optimized using GridSearchCV with parameters:
  - n_estimators: [50, 100]
  - max_depth: [10, 20]

## 📝 Key Findings

### Classification Task
1. Random Forest consistently outperformed Decision Trees
2. Default Random Forest achieved the highest accuracy (97.59%)
3. Hyperparameter tuning improved Decision Tree performance
4. Feature importance analysis showed focused patterns in digit recognition

### Regression Task
1. Random Forest models showed superior performance
2. Significant improvement in R² score with Random Forest (≈0.80)
3. MSE reduced by approximately 50% using Random Forest
4. Tuning provided marginal improvements over default Random Forest

## 🔄 Data Processing

### California Housing Dataset
- Outlier detection and removal using IQR method
- Feature correlation analysis
- VIF calculation for multicollinearity detection
- Standardization of features

### Digits Dataset
- 8x8 pixel images (64 features)
- 1,797 total samples
- Target classes: 0-9
- 70-30 train-test split

## 📈 Visualizations

The notebook includes comprehensive visualizations:
- Confusion matrices for classification results
- Feature importance heatmaps
- Correlation matrices
- Outlier detection plots
- Sample digits visualization

![CART Algorithm Exploration and Evaluation - visual selection (1)](https://github.com/user-attachments/assets/5c9fbc2f-dbe9-42dc-81b1-2cdd2fa18b67)
## 🚀 Getting Started

1. Clone this repository
2. Install required packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```
3. Open and run the Jupyter notebook

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- [Digits Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
