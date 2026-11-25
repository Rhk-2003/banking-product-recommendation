"""Materialise the train/validation blocks for Module 2."""
import sys, gc
sys.path.insert(0, "src")
import pandas as pd
from features_reco import (CAT_COLS, MONTHS, NUM_COLS, PRODUCTS, build_pair,
                           clean_snapshot)


def load_months(path: str) -> pd.DataFrame:
    """
    Stream the 2.3 GB Santander CSV and keep only the five months we model.
    Chunked because the full file does not fit comfortably in memory.
    """
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    keep = set(MONTHS)
    cols = list(dict.fromkeys(["fecha_dato", "ncodpers"] + NUM_COLS + CAT_COLS + PRODUCTS))
    out = []
    for i, ch in enumerate(pd.read_csv(path, usecols=cols, chunksize=1_000_000,
                                       low_memory=False)):
        ch = ch[ch.fecha_dato.isin(keep)]
        if len(ch):
            out.append(ch)
        print(f"  chunk {i} kept {len(ch):,}", flush=True)
    return pd.concat(out, ignore_index=True)


RAW = sys.argv[1] if len(sys.argv) > 1 else "data/train_ver2.csv"
raw = load_months(RAW)
snap = {}
for m in MONTHS:
    snap[m] = clean_snapshot(raw[raw.fecha_dato == m].copy())
    print("cleaned", m, snap[m].shape, flush=True)
del raw; gc.collect()

# features at t, labels at t+1, lag at t-1
blocks = [("2016-02-28", "2016-03-28", "2016-01-28"),
          ("2016-03-28", "2016-04-28", "2016-02-28")]
val    =  ("2016-04-28", "2016-05-28", "2016-03-28")

tr = pd.concat([build_pair(snap, *b) for b in blocks], ignore_index=True)
print("train", tr.shape, "| y_any rate", round(tr.y_any.mean(), 4))
tr.to_parquet("data/reco_train.parquet", index=False); del tr; gc.collect()

va = build_pair(snap, *val)
print("valid", va.shape, "| y_any rate", round(va.y_any.mean(), 4))
va.to_parquet("data/reco_valid.parquet", index=False)
