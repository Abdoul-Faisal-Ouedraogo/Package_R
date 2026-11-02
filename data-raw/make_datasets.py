"""Provenance of the example datasets shipped in inst/extdata/ (tab-separated, '.' decimal).

  wine.tsv          UCI Wine (178 x 13, 3 cultivars), copy bundled with scikit-learn
  breast_cancer.tsv UCI Wisconsin Diagnostic Breast Cancer (569 x 30, 2 classes), idem
  customers.tsv     SYNTHETIC mixed-type data (numeric + categorical predictors, some NA)
Run from the package root:  python data-raw/make_datasets.py   (needs numpy, pandas, scikit-learn)
"""
import numpy as np, pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer
out = "inst/extdata/"

w = load_wine(as_frame=True)
d = w.data.copy()
d.columns = [c.replace("od280/od315_of_diluted_wines", "od280_od315") for c in d.columns]
d["cultivar"] = pd.Series(w.target).map({0: "cultivar_1", 1: "cultivar_2", 2: "cultivar_3"})
d.to_csv(out + "wine.tsv", sep="\t", index=False, na_rep="NA")

b = load_breast_cancer(as_frame=True)
d = b.data.copy()
d.columns = [c.replace(" ", "_") for c in d.columns]
d["diagnosis"] = pd.Series(b.target).map({0: "malignant", 1: "benign"})
d.to_csv(out + "breast_cancer.tsv", sep="\t", index=False, na_rep="NA")

rng = np.random.default_rng(2026); n = 1500
age = rng.normal(45, 13, n).clip(18, 85).round()
region = rng.choice(["North", "South", "East", "West"], n, p=[.3, .25, .25, .2])
contract = rng.choice(["Monthly", "Annual", "TwoYear"], n, p=[.55, .25, .2])
payment = rng.choice(["Card", "Transfer", "Cash"], n, p=[.5, .35, .15])
options = rng.choice(["Yes", "No"], n, p=[.4, .6])
tenure = rng.gamma(2.0, 14, n).clip(1, 72).round()
monthly = (35 + 25 * (options == "Yes") + 0.15 * age + rng.normal(0, 8, n)).clip(15, 140).round(2)
income = (np.exp(rng.normal(10.5, 0.4, n)) / 100).round() * 100
logit = (-0.6 + 0.9 * (contract == "Monthly") - 0.9 * (contract == "TwoYear") - 0.035 * tenure
         + 0.02 * (monthly - 65) + 0.5 * (payment == "Cash") - 0.3 * (options == "Yes")
         + 0.25 * (region == "West") - 0.01 * (age - 45))
churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
d = pd.DataFrame(dict(age=age, income=income, tenure_months=tenure, monthly_charges=monthly,
                      region=region, contract=contract, payment=payment, has_options=options, churn=churn))
d.loc[rng.random(n) < 0.015, "income"] = np.nan          # a few missing values on purpose
d.to_csv(out + "customers.tsv", sep="\t", index=False, na_rep="NA")
