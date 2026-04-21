# Apple Quality Classification

A machine learning project that predicts whether an apple is **good** or **bad** quality based on physical and sensory features.

## Dataset

The [Apple Quality dataset](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality) contains 4,000 samples with 7 features:

| Feature | Description |
|---|---|
| Size | Physical size of the apple |
| Weight | Weight of the apple |
| Sweetness | Sweetness level |
| Crunchiness | Texture/crunchiness |
| Juiciness | Juiciness level |
| Ripeness | Ripeness level |
| Acidity | Acidity level |

Target: `Quality` — `good` (1) or `bad` (0)

## Project Structure

```
ML_Classification_apples/
├── apple_ml.ipynb   # Main notebook
└── README.md
```

## How to Run

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib
```

### 2. Download the dataset

Download `apple_quality.csv` from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality) and update the file path in cell-0 of the notebook:

```python
df = pd.read_csv(r"path/to/apple_quality.csv")
```

### 3. Run the notebook

Open `apple_ml.ipynb` in Jupyter and run all cells top to bottom.

```bash
jupyter notebook apple_ml.ipynb
```

## Approach

Three classifiers are compared, each wrapped in a `sklearn Pipeline` (StandardScaler → model) to ensure consistent scaling and prevent data leakage:

1. **Logistic Regression** — baseline model
2. **Random Forest** — tuned with GridSearchCV (`n_estimators`, `max_depth`, `min_samples_split`)
3. **SVM (RBF kernel)** — tuned with GridSearchCV (`C`, `gamma`)

All models are evaluated with 5-fold cross-validation on the training set, then assessed on a held-out test set (20% split).

## Class Balance

The dataset is **balanced**: 2,000 good apples and 2,000 bad apples (50/50 split). This means accuracy, precision, recall, and F1 are all valid evaluation metrics — no oversampling or class-weight adjustments are needed.

## Results

| Model | Test Accuracy | ROC-AUC | CV Accuracy (5-fold) |
|---|---|---|---|
| Logistic Regression | 0.76 | 0.82 | 0.743 ± 0.014 |
| Random Forest (tuned) | 0.90 | 0.96 | 0.878 ± 0.008 |
| SVM (tuned, C=100) | 0.92 | 0.97 | 0.902 ± 0.005 |

**Best model:** SVM with RBF kernel (`C=100`, `gamma='scale'`), achieving **92% accuracy** and **0.97 ROC-AUC** on the test set.

The Random Forest feature importance analysis shows which apple attributes most strongly predict quality.
