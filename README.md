# Banking Product Recommendation

Predicts which banking product a customer will **newly acquire next month**, from
Santander's **13.6M monthly customer snapshots** across 24 products — served through a
Streamlit app that scores any customer profile in real time and produces a ranked sales
worklist.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-two--stage-green)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)

**MAP@7 0.877** (popularity baseline 0.645) · **ROC-AUC 0.916** · **PR-AUC 0.420** against a 3.01% base rate

---

## The framing decision that defines this project

The obvious label is *does this customer own product X?* It is also useless. A sales
officer cannot sell a product the customer already has, and ownership is trivially
predictable from last month's ownership — a model that scores 0.99 and helps nobody.

So the label here is a **new acquisition**: the customer did **not** hold product `p` in
month *t*, and **does** hold it in month *t+1*. Ownership at *t* becomes a feature.
That makes the problem genuinely hard — only **3.01%** of customers acquire anything in
a given month.

## Two stages, because there are two questions

```
Stage 1   P(acquires any new product)     binary · 1.84M training rows · all customers
          → "is this customer worth calling at all?"

Stage 2   P(product = p | acquires)       multiclass · 68k adopter rows · 16 products
          → "if they buy, what do they buy?"

score(p) = Stage1 × Stage2  →  absolute propensity, for ranking a call list
```

A single model cannot serve both. Ranking products well says nothing about whether the
customer will buy; a propensity score says nothing about which product to pitch. The app
shows both numbers and explains which to use for which decision.

## Results

| Metric | Model | Baseline | Note |
|---|---|---|---|
| MAP@7 (adopters) | **0.8772** | 0.6448 | +36% relative over popularity |
| Stage 1 ROC-AUC | **0.9157** | — | |
| Stage 1 PR-AUC | **0.4195** | 0.0301 | ~14× the base rate |

**Read the baseline honestly.** The baseline is not random — it recommends the globally
most-acquired products, skipping anything already held, and scores 0.645 on its own,
because acquisitions are heavily concentrated in Direct Debit, Pensions and Payroll. The
honest headline is the **36% relative gain**, not the raw 0.877.

MAP@7 is measured **over adopters only**. Across all customers it is 0.026, because 97%
of them buy nothing and score zero by construction.

## Temporal validation

| Block | Features from | Labels from |
|---|---|---|
| Train | Feb 2016, Mar 2016 | Mar 2016, Apr 2016 |
| Validation | **Apr 2016** | **May 2016** |

Lag features come from *t−1*. Nothing from the label month reaches the features, and the
validation month is strictly after every training month — so the score is a forecast,
not a reconstruction.

## What the app does

- **Recommend products** — set age, tenure, income, segment, relationship type and
  current holdings; get a ranked list of what they'd take next. Or load one of 300
  held-out real customers and see the model's prediction against what they *actually*
  bought in May 2016, hits and misses both shown.
- **Model performance** — metrics, the baseline comparison, which products are modelled
  and which are not, and feature importance for both stages.
- **Sales worklist** — score a batch, rank by propensity, attach the recommended product,
  export to CSV. Includes a decile chart: the real test of a ranking model is whether the
  customers it scores highest actually buy more often.

## Findings worth discussing

**`dropped_last_month` is the strongest propensity feature.** It is *not* leakage — it
compares *t−1* to *t*, both strictly before the label month. It is real behaviour:
several Santander flags (notably Payroll and Pensions) flicker off and back on, so a
customer who dropped a product last month is unusually likely to "acquire" one next
month. Part of what this model predicts is therefore administrative churn, not genuine
demand. That belongs in the handover note, not buried in a notebook.

**8 of 24 products are not modelled.** They had fewer than 50 training acquisitions —
four (Guarantees, Short/Medium-Term Deposit, Savings) had effectively zero. A classifier
trained on 7 positive examples produces a number, not a prediction, so those products are
excluded rather than quietly scored. Ask this model about a mortgage and it declines to
answer — which is the correct answer.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Trained models and dashboard assets are committed, so the app runs immediately.

### Rebuilding from the raw data

Download `train_ver2.csv` from the Kaggle competition into `data/`, then:

```bash
python src/build_reco_data.py data/train_ver2.csv   # stream 2.3 GB → train/valid blocks
python src/train_reco.py                            # fit both stages + evaluate
python src/prepare_dashboard_assets.py              # app assets
```

The CSV is read in 1M-row chunks — the full file does not fit comfortably in memory.

## Layout

```
├── app.py                            Streamlit app
├── src/
│   ├── features_reco.py              cleaning, acquisition labels, feature builder
│   ├── recommendation.py             MAP@7 + ranking rule
│   ├── build_reco_data.py            step 1
│   ├── train_reco.py                 step 2
│   ├── prepare_dashboard_assets.py   step 3
│   ├── inference.py                  shared by app + scripts
│   └── viz.py                        chart theme
├── models/                           both stages + metadata
├── reports/                          feature importance, acquisition stats
└── data/                             300-customer demo sample (raw CSV gitignored)
```

## Limitations

- **Part of the signal is flag-flicker**, not demand — see `dropped_last_month` above.
- **8 of 24 products cannot be modelled** from this window, including mortgages and loans.
- **MAP@7 0.877 is over adopters only**, against a 0.645 baseline — quote the gain.
- **Spanish retail banking, 2015–16.** Not transferable to another bank or market
  without refitting.
- **These are propensity scores, not eligibility decisions.** Real offers need consent,
  suitability and credit checks.

## Data source

[Santander Product Recommendation (Kaggle)](https://www.kaggle.com/c/santander-product-recommendation) — 13,647,309 customer-months, Jan 2015 – May 2016, 956,645 customers, 24 products.

## Push to GitHub

```bash
git remote add origin https://github.com/Rhk-2003/banking-product-recommendation.git
git branch -M main
git push -u origin main
```

The 2.3 GB Kaggle CSV and the large regenerable intermediates are gitignored; the trained
models and dashboard assets are committed, so anyone cloning the repo can run the app
without downloading the dataset.
