"""Inference surface for the recommendation app, so the dashboard and the training
script can never drift apart in how a customer is encoded."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features_reco import CAT_COLS, NUM_COLS, PRODUCT_NAMES, PRODUCTS, feature_columns
from recommendation import rank_products

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def load_reco():
    s1 = joblib.load(MODELS / "stage1_propensity.pkl")
    s2 = joblib.load(MODELS / "stage2_product.pkl")
    meta = json.loads((MODELS / "reco_meta.json").read_text())
    return s1, s2, meta


def build_customer_row(profile: dict, owned: list, meta: dict) -> pd.DataFrame:
    """
    Turn dashboard inputs into exactly the feature frame the models were trained
    on: same columns, same order, same categorical level sets.
    """
    feats = feature_columns()
    row = {}
    for c in NUM_COLS:
        row[c] = profile.get(c, np.nan)
    for c in CAT_COLS:
        row[c] = profile.get(c, "__NA__")
    for p in PRODUCTS:
        row[f"own_{p}"] = 1 if p in owned else 0
        # no history supplied from the UI -> same month-ago state, which is the
        # honest default for a what-if (we are not inventing an acquisition)
        row[f"lag_{p}"] = 1 if p in owned else 0
    row["n_products"] = len(owned)
    row["n_products_lag"] = float(len(owned))
    row["added_last_month"] = profile.get("added_last_month", 0)
    row["dropped_last_month"] = profile.get("dropped_last_month", 0)
    row["is_new_customer"] = profile.get("is_new_customer", 0)

    X = pd.DataFrame([row])[feats]
    # Training cast every column in CAT_COLS to pandas 'category' (indrel_1mes
    # included, even though it is numeric). LightGBM records how many category
    # columns it saw, so inference has to cast exactly the same set or it raises
    # "train and valid dataset categorical_feature do not match".
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c].astype(str), categories=meta["cat_levels"][c])
    for c in NUM_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def recommend(s1, s2, meta, X: pd.DataFrame, owned: list, top_n: int = 7) -> dict:
    p1 = s1["model"].predict_proba(X)[:, 1]
    p2 = s2["model"].predict_proba(X)
    owned_mask = np.array([[1 if p in owned else 0 for p in PRODUCTS]])
    recs, scores = rank_products(p2, s2["classes"], owned_mask,
                                 stage1_proba=None, top_n=top_n)
    tbl = pd.DataFrame({
        "product_code": s2["classes"],
        "product": [PRODUCT_NAMES[c] for c in s2["classes"]],
        "p_given_acquires": p2[0],
        "already_held": [c in owned for c in s2["classes"]],
    })
    tbl["expected_propensity"] = tbl["p_given_acquires"] * p1[0]
    tbl = tbl[~tbl.already_held].sort_values("p_given_acquires", ascending=False)
    return {"p_acquires_any": float(p1[0]), "ranked": tbl.head(top_n).reset_index(drop=True),
            "all_scores": tbl.reset_index(drop=True)}
