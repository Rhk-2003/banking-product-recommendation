"""Train the two-stage next-month product acquisition model."""
import json, sys, gc
sys.path.insert(0, "src")
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features_reco import CAT_COLS, PRODUCTS, feature_columns
from recommendation import mapk, rank_products

FEATS = feature_columns()
YCOLS = [f"y_{p}" for p in PRODUCTS]
MIN_POS = 50     # a product needs this many training acquisitions to be modelled

tr = pd.read_parquet("data/reco_train.parquet")
va = pd.read_parquet("data/reco_valid.parquet")
for d in (tr, va):
    for c in CAT_COLS:
        d[c] = d[c].astype("category")
# align category levels between train and valid
for c in CAT_COLS:
    cats = tr[c].cat.categories.union(va[c].cat.categories)
    tr[c] = tr[c].cat.set_categories(cats)
    va[c] = va[c].cat.set_categories(cats)

# ---------------- Stage 1: will the customer acquire anything? ----------------
print("stage 1: propensity to acquire any new product")
s1 = lgb.LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.05,
                        num_leaves=63, min_child_samples=100, subsample=0.8,
                        subsample_freq=1, colsample_bytree=0.8, n_jobs=2,
                        random_state=42, verbose=-1)
s1.fit(tr[FEATS], tr["y_any"], categorical_feature=CAT_COLS,
       eval_set=[(va[FEATS], va["y_any"])], eval_metric="average_precision",
       callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(100)])
p1_va = s1.predict_proba(va[FEATS])[:, 1]
auc = roc_auc_score(va["y_any"], p1_va)
ap = average_precision_score(va["y_any"], p1_va)
print(f"  stage1 valid ROC-AUC={auc:.4f}  PR-AUC={ap:.4f}  (base rate={va.y_any.mean():.4f})")

# ---------------- Stage 2: which product, given they acquire ------------------
print("stage 2: which product, among customers who acquired something")


def expand(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One row per (customer, newly acquired product) pair."""
    adopters = d[d.y_any == 1]
    ys = adopters[YCOLS].to_numpy()
    rows, labels = np.nonzero(ys)
    # reset_index: a customer adding two products yields two rows that would
    # otherwise share an index label, which makes .loc expand them later
    X = adopters.iloc[rows][FEATS].reset_index(drop=True)
    y = pd.Series(np.array(PRODUCTS)[labels], name="target")
    return X, y


Xtr, ytr = expand(tr)
counts = ytr.value_counts()
keep = counts[counts >= MIN_POS].index.tolist()
print("  product classes kept:", len(keep), "of", len(PRODUCTS))
print("  dropped (too few training acquisitions):",
      {k: int(v) for k, v in counts[counts < MIN_POS].items()})
mask = ytr.isin(keep).to_numpy()
Xtr, ytr = Xtr[mask].reset_index(drop=True), ytr[mask].reset_index(drop=True)
print("  stage2 train rows:", len(Xtr))

classes = sorted(keep)
code = {c: i for i, c in enumerate(classes)}
s2 = lgb.LGBMClassifier(objective="multiclass", num_class=len(classes),
                        n_estimators=400, learning_rate=0.07, num_leaves=63,
                        min_child_samples=50, subsample=0.9, subsample_freq=1,
                        colsample_bytree=0.8, n_jobs=2, random_state=42, verbose=-1)
Xva, yva = expand(va)
vmask = yva.isin(classes).to_numpy()
s2.fit(Xtr, ytr.map(code), categorical_feature=CAT_COLS,
       eval_set=[(Xva[vmask], yva[vmask].map(code))], eval_metric="multi_logloss",
       callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(100)])

# ---------------- MAP@7 on the full validation month -------------------------
print("evaluating MAP@7 on 2016-05 acquisitions")
owned = va[[f"own_{p}" for p in PRODUCTS]].to_numpy()
p2_va = s2.predict_proba(va[FEATS])
recs, _ = rank_products(p2_va, classes, owned, stage1_proba=None, top_n=7)
actual = [[PRODUCTS[j] for j in np.nonzero(r)[0]] for r in va[YCOLS].to_numpy()]

has = [i for i, a in enumerate(actual) if a]
map7_adopters = mapk([actual[i] for i in has], [recs[i] for i in has], 7)
map7_all = mapk(actual, recs, 7)

# baseline: recommend the globally most-acquired products, excluding held ones
pop = pd.Series(tr[YCOLS].sum().to_numpy(), index=PRODUCTS).sort_values(ascending=False)
pop_order = pop.index.tolist()
prod_idx = {p: i for i, p in enumerate(PRODUCTS)}
base_recs = [[p for p in pop_order if owned[i, prod_idx[p]] == 0][:7] for i in range(len(va))]
base_map7 = mapk([actual[i] for i in has], [base_recs[i] for i in has], 7)

print(f"  MAP@7 (adopters only)   model={map7_adopters:.4f}   popularity baseline={base_map7:.4f}")
print(f"  MAP@7 (all customers)   model={map7_all:.4f}")

joblib.dump({"model": s1, "features": FEATS, "cat_cols": CAT_COLS}, "models/stage1_propensity.pkl")
joblib.dump({"model": s2, "classes": classes, "features": FEATS, "cat_cols": CAT_COLS},
            "models/stage2_product.pkl")

cat_levels = {c: [str(x) for x in tr[c].cat.categories] for c in CAT_COLS}
meta = {
    "stage1": {"roc_auc": float(auc), "pr_auc": float(ap), "base_rate": float(va.y_any.mean()),
               "best_iteration": int(s1.best_iteration_ or s1.n_estimators)},
    "stage2": {"classes": classes, "n_train_rows": int(len(Xtr)),
               "best_iteration": int(s2.best_iteration_ or s2.n_estimators)},
    "map7_adopters": map7_adopters, "map7_all": map7_all, "map7_baseline": base_map7,
    "train_months": ["2016-02-28", "2016-03-28"], "valid_month": "2016-04-28",
    "label_month": "2016-05-28",
    "cat_levels": cat_levels,
    "num_defaults": {c: (float(tr[c].median()) if tr[c].notna().any() else 0.0)
                     for c in ["age", "antiguedad", "renta", "cod_prov"]},
}
with open("models/reco_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("saved models + meta")
