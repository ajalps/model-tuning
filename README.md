# Supervised Classification Modeling & Tuning — Credit Default Risk

Trains and compares **4 model architectures** (Logistic Regression, Random Forest,
XGBoost, SVM) on the preprocessed credit-default dataset, tunes each with
`GridSearchCV` + stratified 5-fold cross-validation, evaluates all four on the same
held-out test set, and serializes the champion model with `joblib`.

## Contents

| File | Purpose |
|---|---|
| [`model_tuning.ipynb`](model_tuning.ipynb) | Train, tune, compare, and select the champion model |
| `requirements.txt` | Python dependencies |
| `champion_model.pkl` | Produced when you run the notebook — the saved champion pipeline |

## Approach

- **Models:** Logistic Regression, Random Forest, XGBoost, SVM (RBF kernel). If
  `xgboost` isn't installed, the notebook automatically falls back to scikit-learn's
  own Gradient Boosting so it still runs end to end — install `xgboost` (already in
  `requirements.txt`) to get the real thing.
- **Tuning:** `GridSearchCV` with `StratifiedKFold(n_splits=5)`, scored on ROC-AUC.
- **Evaluation:** precision, recall, F1, ROC-AUC, and a confusion matrix on a held-out
  test set, plus an overlaid ROC curve comparing all four models.
- **Champion selection:** highest test-set ROC-AUC (swap this rule for recall/precision
  if your use case has asymmetric costs — see the problem-framing memo from the earlier
  stage of this project for how to think about that trade-off).
- **Serialization:** the full pipeline (preprocessing + model) is saved together as
  `champion_model.pkl`, so it can be loaded and called on new raw data directly.

## Running the notebook

```bash
pip install -r requirements.txt
jupyter notebook model_tuning.ipynb
```

Run all cells top to bottom (Kernel → Restart & Run All). This can take a few minutes —
`GridSearchCV` is fitting many model/hyperparameter combinations across 5 folds each.

## Reusing the saved model

```python
import joblib
model = joblib.load("champion_model.pkl")
model.predict(new_raw_dataframe)        # same raw columns as the training data
model.predict_proba(new_raw_dataframe)
```
