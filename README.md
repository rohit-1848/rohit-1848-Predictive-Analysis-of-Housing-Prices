# 🏠 Predictive Analysis of Housing Price

End‑to‑end ML pipeline that predicts the sale price of homes in **Ames, Iowa**.  
All code and analysis live in **`Predictive Analysis of Housing Price.ipynb`**.

---

## 📁 Repository Layout
```text
├── train.csv # 1 460 × 81 – training data 
├── test.csv # 1 459 × 80 – evaluation data (no SalePrice)
├── data_description.txt # original feature definitions
├── Untitled.ipynb (Predictive Analysis of Housing Price.ipynb)
├── Prediction.csv # output: Id, SalePrice
└── README.md
```
---

## 🔧 Workflow Summary

| Stage | Highlights |
|-------|------------|
| **EDA** | shape checks, missing‑value matrix, correlation heat‑map, target distribution |
| **Pre‑processing** | • `SimpleImputer` (median / mode) • log‑transform highly‑skewed numeric features • ordinal / one‑hot encoding • standard scaling for numeric |
| **Model selection** | Hyper‑parameter tuning with `GridSearchCV` for:<br>  • Gradient Boosting Regressor (GBR)<br>  • XGBRegressor<br>  • CatBoostRegressor<br>  • LightGBMRegressor<br>  • Random Forest Regressor<br>  • Ridge / Lasso / ElasticNet (baseline) |
| **Ensembles** | 1. **VotingRegressor** (`vr`) blending GBR + XGB + Ridge (weights 2 : 3 : 1)<br>2. **StackingRegressor** (`stackreg`) with the tuned GBR, XGB, Cat, LGBM, RFR as base estimators and the `vr` voting model as meta‑learner |
| **Validation** | 80 / 20 split → metrics: **RMSE (log‑scale)** & **R²** |
| **Inference** | Predictions on `test.csv`, exponentiated via `np.exp`, saved to `prediction.csv` |

---

## 📊 Results (Validation Split, log‑target)

| Model | RMSE | R² |
|-------|------|----|
| CatBoost | **0.1005** | 0.924 |
| Gradient Boosting | 0.1021 | 0.922 |
| XGBoost | 0.1039 | 0.919 |
| LightGBM | 0.1125 | 0.905 |
| Random Forest | 0.1186 | 0.894 |
| **Stacked Ensemble** | 0.1056 | 0.916 |

> CatBoost slightly out‑performed the stacked ensemble, showing that a well‑tuned single model can still be top.

---

## 🚀 Quick Start

### 1️⃣  Clone the repository and enter the project folder
```bash
git clone https://github.com/ABTG716/Housing_price.git
cd Housing_price
```

### 2️⃣  Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```
#### ➜ Windows: .venv\Scripts\activate

### 3️⃣  Install dependencies
```bash
pip install -r requirements.txt
```
#### pandas, numpy, scikit‑learn, xgboost, lightgbm, catboost, seaborn, matplotlib

### 4️⃣  Launch the notebook (GUI)
```bash
jupyter notebook "Predictive Analysis of Housing Price.ipynb"
```

###     ── or run it headless ──
```bash
jupyter nbconvert --to notebook --execute "Predictive Analysis of Housing Price.ipynb"
```

### 5️⃣  Verify the submission file
```bash
ls -l prediction.csv
```
#### should contain:  Id,  SalePrice

---

## 📦 Dependencies
All required packages (and specific versions) are listed in requirements.txt.
