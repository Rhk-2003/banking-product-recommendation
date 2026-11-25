"""Step 3: pre-compute the small artefacts the Streamlit app reads, so it starts fast."""
import sys
sys.path.insert(0, "src")
import joblib
import pandas as pd
from features_reco import PRODUCT_NAMES, PRODUCTS

for name, path in [("s1", "models/stage1_propensity.pkl"), ("s2", "models/stage2_product.pkl")]:
    b = joblib.load(path)
    imp = pd.DataFrame({"feature": b["features"],
                        "gain": b["model"].booster_.feature_importance("gain")})
    imp["pretty"] = imp["feature"].apply(
        lambda f: PRODUCT_NAMES.get(f.replace("own_", "").replace("lag_", ""), f)
        + (" (held)" if f.startswith("own_") else
           " (held, prior month)" if f.startswith("lag_") else ""))
    imp.sort_values("gain", ascending=False).head(25).to_csv(
        f"reports/feature_importance_{name}.csv", index=False)

va = pd.read_parquet("data/reco_valid.parquet")
ycols = [f"y_{p}" for p in PRODUCTS]

# a readable demo sample: half real adopters, half not
sample = pd.concat([va[va.y_any == 1].sample(150, random_state=1),
                    va[va.y_any == 0].sample(150, random_state=1)]) \
    .sample(frac=1, random_state=2).reset_index(drop=True)
sample.to_parquet("data/sample_customers.parquet", index=False)

acq = pd.Series(va[ycols].sum().to_numpy(), index=PRODUCTS).sort_values(ascending=False)
pd.DataFrame({"product_code": acq.index,
              "product": [PRODUCT_NAMES[p] for p in acq.index],
              "acquisitions": acq.to_numpy(),
              "ownership_rate": va[[f"own_{p}" for p in acq.index]].mean().to_numpy()}) \
    .to_csv("reports/product_acquisition_stats.csv", index=False)
print("dashboard assets written:", sample.shape)
