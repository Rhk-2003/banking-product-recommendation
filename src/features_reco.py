"""
Module 2 feature engineering: Santander monthly customer snapshots -> supervised
next-month product-acquisition dataset.

Framing. The label is a *new* product acquisition: the customer did not hold
product p in month t but does hold it in month t+1. Plain ownership is far easier
to predict and commercially useless (a sales officer cannot sell a product the
customer already has), so ownership at t is used as a feature, never as a label.

Only 3.01% of customers acquire anything in a given month, so the task is split
in two (see train_reco.py):
  Stage 1  P(acquires any new product next month)   - binary, all customers
  Stage 2  P(product = p | acquires something)      - multiclass, adopters only
A product's score is the product of the two, which keeps the ranking calibrated
against an absolute propensity instead of only a relative one.
"""
import numpy as np
import pandas as pd

PRODUCTS = [
    "ind_ahor_fin_ult1", "ind_aval_fin_ult1", "ind_cco_fin_ult1", "ind_cder_fin_ult1",
    "ind_cno_fin_ult1", "ind_ctju_fin_ult1", "ind_ctma_fin_ult1", "ind_ctop_fin_ult1",
    "ind_ctpp_fin_ult1", "ind_deco_fin_ult1", "ind_deme_fin_ult1", "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1", "ind_fond_fin_ult1", "ind_hip_fin_ult1", "ind_plan_fin_ult1",
    "ind_pres_fin_ult1", "ind_reca_fin_ult1", "ind_tjcr_fin_ult1", "ind_valo_fin_ult1",
    "ind_viv_fin_ult1", "ind_nomina_ult1", "ind_nom_pens_ult1", "ind_recibo_ult1",
]

# Plain-English names, so the dashboard never shows a raw Spanish column code.
PRODUCT_NAMES = {
    "ind_ahor_fin_ult1": "Savings Account",
    "ind_aval_fin_ult1": "Guarantees",
    "ind_cco_fin_ult1": "Current Account",
    "ind_cder_fin_ult1": "Derivada Account",
    "ind_cno_fin_ult1": "Payroll Account",
    "ind_ctju_fin_ult1": "Junior Account",
    "ind_ctma_fin_ult1": "Más Particular Account",
    "ind_ctop_fin_ult1": "Particular Account",
    "ind_ctpp_fin_ult1": "Particular Plus Account",
    "ind_deco_fin_ult1": "Short-Term Deposit",
    "ind_deme_fin_ult1": "Medium-Term Deposit",
    "ind_dela_fin_ult1": "Long-Term Deposit",
    "ind_ecue_fin_ult1": "e-Account",
    "ind_fond_fin_ult1": "Funds",
    "ind_hip_fin_ult1": "Mortgage",
    "ind_plan_fin_ult1": "Pension Plan",
    "ind_pres_fin_ult1": "Loan",
    "ind_reca_fin_ult1": "Taxes Account",
    "ind_tjcr_fin_ult1": "Credit Card",
    "ind_valo_fin_ult1": "Securities",
    "ind_viv_fin_ult1": "Home Account",
    "ind_nomina_ult1": "Payroll",
    "ind_nom_pens_ult1": "Pensions",
    "ind_recibo_ult1": "Direct Debit",
}

CAT_COLS = ["ind_empleado", "sexo", "indrel_1mes", "tiprel_1mes", "indresi",
            "indext", "indfall", "segmento", "canal_entrada", "nomprov"]
NUM_COLS = ["age", "antiguedad", "renta", "ind_nuevo", "ind_actividad_cliente",
            "indrel", "cod_prov"]

MONTHS = ["2016-01-28", "2016-02-28", "2016-03-28", "2016-04-28", "2016-05-28"]


def clean_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes and the known dirty encodings in the raw monthly file."""
    df = df.copy()
    for c in ["age", "antiguedad", "renta", "ind_nuevo", "ind_actividad_cliente",
              "indrel", "cod_prov"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # antiguedad (months with the bank) uses -999999 as a missing marker
    df.loc[df["antiguedad"] < 0, "antiguedad"] = np.nan
    df.loc[(df["age"] < 18) | (df["age"] > 100), "age"] = np.nan

    # indrel_1mes arrives as a mix of '1' and '1.0'
    df["indrel_1mes"] = pd.to_numeric(df["indrel_1mes"], errors="coerce").astype("float32")

    for c in CAT_COLS:
        if c == "indrel_1mes":
            continue
        df[c] = df[c].astype("string").fillna("__NA__").astype("category")

    for p in PRODUCTS:
        df[p] = pd.to_numeric(df[p], errors="coerce").fillna(0).astype("int8")
    return df


def build_pair(snap: dict, t: str, t_next: str, t_lag: str | None) -> pd.DataFrame:
    """
    One supervised block: customer state at month `t`, labels from month `t_next`.

    Returns one row per customer present in both t and t_next, with:
      - demographics at t
      - ownership of all 24 products at t                 (own_*)
      - ownership one month earlier at t_lag              (lag_*)
      - churn/add activity between t_lag and t            (added_last_month etc.)
      - the acquisition label per product at t_next       (y_*)
    """
    cur, nxt = snap[t], snap[t_next]
    cur = cur.set_index("ncodpers")
    nxt = nxt.set_index("ncodpers")
    common = cur.index.intersection(nxt.index)
    cur, nxt = cur.loc[common], nxt.loc[common]

    out = cur[NUM_COLS + CAT_COLS].copy()
    own = cur[PRODUCTS].astype("int8")
    out[[f"own_{p}" for p in PRODUCTS]] = own.to_numpy()
    out["n_products"] = own.sum(axis=1).astype("int8")

    if t_lag is not None:
        lag = snap[t_lag].set_index("ncodpers")[PRODUCTS]
        lag = lag.reindex(common)          # customers new this month get NaN
        lag_arr = lag.to_numpy(dtype="float32")
        out[[f"lag_{p}" for p in PRODUCTS]] = lag_arr
        delta = own.to_numpy(dtype="float32") - lag_arr
        out["added_last_month"] = np.nansum(delta == 1, axis=1).astype("int8")
        out["dropped_last_month"] = np.nansum(delta == -1, axis=1).astype("int8")
        out["n_products_lag"] = np.nansum(lag_arr, axis=1).astype("float32")
        out["is_new_customer"] = lag.isna().all(axis=1).astype("int8").to_numpy()
    else:
        for p in PRODUCTS:
            out[f"lag_{p}"] = np.nan
        out["added_last_month"] = 0
        out["dropped_last_month"] = 0
        out["n_products_lag"] = np.nan
        out["is_new_customer"] = 1

    # labels: held next month, not held this month
    y = ((nxt[PRODUCTS].to_numpy(dtype="int8") - own.to_numpy(dtype="int8")) == 1).astype("int8")
    out[[f"y_{p}" for p in PRODUCTS]] = y
    out["y_any"] = (y.sum(axis=1) > 0).astype("int8")
    out["month"] = t
    return out.reset_index()


def feature_columns() -> list[str]:
    return (NUM_COLS + CAT_COLS
            + [f"own_{p}" for p in PRODUCTS]
            + [f"lag_{p}" for p in PRODUCTS]
            + ["n_products", "n_products_lag", "added_last_month",
               "dropped_last_month", "is_new_customer"])


# Plain-English names for the non-product features, so charts never show a raw
# Spanish column code.
FEATURE_NAMES = {
    "age": "Age",
    "antiguedad": "Tenure with bank (months)",
    "renta": "Household income",
    "ind_nuevo": "New customer flag",
    "ind_actividad_cliente": "Activity flag",
    "indrel": "Primary customer flag",
    "indrel_1mes": "Customer type (month start)",
    "tiprel_1mes": "Relationship type",
    "cod_prov": "Province code",
    "nomprov": "Province",
    "ind_empleado": "Employee status",
    "sexo": "Sex",
    "indresi": "Residence flag",
    "indext": "Foreigner flag",
    "indfall": "Deceased flag",
    "canal_entrada": "Acquisition channel",
    "n_products": "Products held",
    "n_products_lag": "Products held (prior month)",
    "added_last_month": "Products added last month",
    "dropped_last_month": "Products dropped last month",
    "is_new_customer": "No history last month",
}


def pretty_feature(f: str) -> str:
    """Render any model feature name for display."""
    if f.startswith("own_"):
        return f"{PRODUCT_NAMES.get(f[4:], f[4:])} (held)"
    if f.startswith("lag_"):
        return f"{PRODUCT_NAMES.get(f[4:], f[4:])} (held, prior month)"
    return FEATURE_NAMES.get(f, f)
