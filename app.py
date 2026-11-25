"""
Banking Product Recommendation - Streamlit app.

Predicts which banking product a customer will NEWLY acquire next month, from
Santander's monthly customer snapshots (13.6M customer-months, 24 products).

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import inference as inf
import viz
from features_reco import CAT_COLS, PRODUCT_NAMES, PRODUCTS

st.set_page_config(page_title="Banking Product Recommendation",
                   page_icon="▤", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1400px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  [data-testid="stMetricValue"] {font-size: 1.6rem;}
  .note {color:#666666; font-size:13px; line-height:1.5;}
  .card {border:1px solid #e8e8e8; border-radius:10px; padding:16px 18px; background:#ffffff;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_models():
    return inf.load_reco()


@st.cache_data(show_spinner=False)
def load_tables():
    return {
        "imp_s1": pd.read_csv(ROOT / "reports/feature_importance_s1.csv"),
        "imp_s2": pd.read_csv(ROOT / "reports/feature_importance_s2.csv"),
        "acq": pd.read_csv(ROOT / "reports/product_acquisition_stats.csv"),
    }


@st.cache_data(show_spinner=False)
def load_sample_customers():
    return pd.read_parquet(ROOT / "data/sample_customers.parquet")


try:
    S1, S2, RECO_META = load_models()
    T = load_tables()
except FileNotFoundError as e:
    st.error(f"Model artefacts missing: {e}\n\nRun the pipeline first - see README.md.")
    st.stop()

st.sidebar.markdown("### Product Recommendation")
PAGE = st.sidebar.radio("Page", ["Recommend products", "Model performance",
                                 "Sales worklist", "About the data"],
                        label_visibility="collapsed", key="nav_page")
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div class='note'>LightGBM, two-stage<br>"
    f"ROC-AUC {RECO_META['stage1']['roc_auc']:.3f}<br>"
    f"MAP@7 {RECO_META['map7_adopters']:.3f}<br>"
    f"{len(RECO_META['stage2']['classes'])} products modelled</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Page 1: interactive recommendation
# --------------------------------------------------------------------------- #
if PAGE == "Recommend products":
    st.title("What would this customer buy next month?")
    st.markdown("<div class='note'>Set the customer profile and tick what they already "
                "hold. The model ranks the products they do <i>not</i> hold by how likely "
                "each is to be their next acquisition.</div>", unsafe_allow_html=True)
    st.write("")

    lv = RECO_META["cat_levels"]
    nd = RECO_META["num_defaults"]

    mode = st.radio("Input mode", ["Set parameters manually", "Load a real validation customer"],
                    horizontal=True, label_visibility="collapsed", key="reco_mode")

    loaded = None
    if mode == "Load a real validation customer":
        sc = load_sample_customers()
        idx = st.selectbox(
            "Customer (300 held-out customers from May 2016, half of them real adopters)",
            range(len(sc)),
            format_func=lambda i: (f"#{int(sc.ncodpers.iloc[i])} · age {sc.age.iloc[i]:.0f} · "
                                   f"{sc.segmento.iloc[i]} · {int(sc.n_products.iloc[i])} products"
                                   f"{' · acquired next month' if sc.y_any.iloc[i] == 1 else ''}"))
        loaded = sc.iloc[int(idx)]

    cL, cR = st.columns([1, 1.4], gap="large")

    with cL:
        st.subheader("Customer profile")
        if loaded is not None:
            age_v = float(loaded.age) if pd.notna(loaded.age) else nd["age"]
            ant_v = float(loaded.antiguedad) if pd.notna(loaded.antiguedad) else nd["antiguedad"]
            ren_v = float(loaded.renta) if pd.notna(loaded.renta) else nd["renta"]
            seg_v, sex_v = str(loaded.segmento), str(loaded.sexo)
            rel_v, ch_v = str(loaded.tiprel_1mes), str(loaded.canal_entrada)
            act_v = int(loaded.ind_actividad_cliente) if pd.notna(loaded.ind_actividad_cliente) else 1
            prov_v = str(loaded.nomprov)
            owned_default = [p for p in PRODUCTS if loaded[f"own_{p}"] == 1]
        else:
            age_v, ant_v, ren_v = nd["age"], nd["antiguedad"], nd["renta"]
            seg_v, sex_v, rel_v, ch_v, act_v = "02 - PARTICULARES", "V", "A", "KHE", 1
            prov_v = "MADRID"
            owned_default = ["ind_cco_fin_ult1"]

        segopts = [c for c in lv["segmento"] if c != "__NA__"]
        relopts = [c for c in lv["tiprel_1mes"] if c != "__NA__"]
        sexopts = [c for c in lv["sexo"] if c != "__NA__"]

        age2 = st.slider("Age", 18, 100, int(age_v))
        ant2 = st.slider("Tenure with bank (months)", 0, 300, int(min(ant_v, 300)))
        ren2 = st.number_input("Gross household income (€)", 0.0, 2_000_000.0,
                               float(ren_v), step=1000.0)
        seg2 = st.selectbox("Segment", segopts,
                            index=segopts.index(seg_v) if seg_v in segopts else 0)
        rel2 = st.selectbox("Relationship type", relopts,
                            index=relopts.index(rel_v) if rel_v in relopts else 0,
                            help="A = active, I = inactive, P = former, R = potential")
        act2 = st.selectbox("Activity flag", [1, 0], index=0 if act_v == 1 else 1,
                            format_func=lambda v: "Active customer" if v == 1 else "Inactive")
        with st.expander("More attributes"):
            sex2 = st.selectbox("Sex", sexopts,
                                index=sexopts.index(sex_v) if sex_v in sexopts else 0)
            ch2 = st.selectbox("Acquisition channel", lv["canal_entrada"],
                               index=lv["canal_entrada"].index(ch_v)
                               if ch_v in lv["canal_entrada"] else 0)
            prov2 = st.selectbox("Province", lv["nomprov"],
                                 index=lv["nomprov"].index(prov_v)
                                 if prov_v in lv["nomprov"] else 0)
            drop2 = st.slider("Products dropped last month", 0, 5, 0,
                              help="Top driver of the stage-1 propensity model: customers "
                                   "who dropped a product often re-acquire the next month.")

        st.subheader("Products already held")
        owned = st.multiselect("Current holdings", PRODUCTS, default=owned_default,
                               format_func=lambda p: PRODUCT_NAMES[p])
        st.caption(f"{len(owned)} of {len(PRODUCTS)} products held. "
                   "Held products are excluded from the recommendation list.")

    profile = {
        "age": age2, "antiguedad": ant2, "renta": ren2, "segmento": seg2,
        "sexo": sex2, "tiprel_1mes": rel2, "canal_entrada": ch2, "nomprov": prov2,
        "ind_actividad_cliente": act2, "indrel": 1, "indrel_1mes": "1.0",
        "ind_nuevo": 1 if ant2 <= 6 else 0, "cod_prov": nd["cod_prov"],
        "ind_empleado": "N", "indresi": "S", "indext": "N", "indfall": "N",
        "added_last_month": 0, "dropped_last_month": drop2,
        "is_new_customer": 1 if ant2 <= 1 else 0,
    }
    X = inf.build_customer_row(profile, owned, RECO_META)
    out = inf.recommend(S1, S2, RECO_META, X, owned, top_n=7)
    ranked = out["ranked"]

    with cR:
        st.subheader("Recommendation")
        p_any, base = out["p_acquires_any"], RECO_META["stage1"]["base_rate"]
        m1, m2, m3 = st.columns(3)
        m1.metric("P(buys anything next month)", f"{p_any * 100:.2f}%",
                  f"{p_any / base:.1f}× base rate", delta_color="off")
        m2.metric("Top recommendation", ranked.iloc[0]["product"] if len(ranked) else "—")
        m3.metric("Confidence in top pick",
                  f"{ranked.iloc[0]['p_given_acquires'] * 100:.1f}%" if len(ranked) else "—")

        if p_any >= 3 * base:
            st.success("High-propensity customer — worth an outbound call this cycle.")
        elif p_any >= base:
            st.info("Around or above the portfolio average. Suitable for a campaign list, "
                    "not a priority call.")
        else:
            st.warning("Below the portfolio average propensity. The ranking below still "
                       "shows the best-fit product, but expect a low conversion rate.")

        if len(ranked) == 0:
            st.error("This customer already holds every modelled product — nothing to recommend.")
        else:
            disp = ranked.head(7).copy()
            st.plotly_chart(
                viz.hbar(disp["product"], (disp["p_given_acquires"] * 100).round(2),
                         "Next-product ranking",
                         "P(product | customer acquires something) %",
                         color=viz.SERIES_1,
                         texts=[f"{v*100:.1f}%" for v in disp["p_given_acquires"]]),
                width="stretch")
            tbl = disp[["product", "p_given_acquires", "expected_propensity"]].copy()
            tbl.columns = ["Product", "P(product | acquires)", "Expected propensity"]
            st.dataframe(tbl.style.format({"P(product | acquires)": "{:.2%}",
                                           "Expected propensity": "{:.3%}"}),
                         width="stretch", hide_index=True)
            st.caption("**P(product | acquires)** ranks *which* product — it sums to ~1 "
                       "across all candidates. **Expected propensity** multiplies it by "
                       "P(buys anything), giving the absolute chance this customer takes "
                       "that product next month. Use the first to choose the pitch, the "
                       "second to decide whether to call at all.")

        if loaded is not None:
            actual = [PRODUCT_NAMES[p] for p in PRODUCTS if loaded[f"y_{p}"] == 1]
            st.markdown("---")
            if actual:
                top7 = disp["product"].tolist() if len(ranked) else []
                hit = [a for a in actual if a in top7]
                st.markdown(f"**Ground truth (May 2016):** this customer actually acquired "
                            f"{', '.join(actual)}.")
                if hit:
                    st.success(f"Model ranked {', '.join(hit)} in its top 7 — "
                               f"position {[top7.index(h) + 1 for h in hit]}.")
                else:
                    st.error("None of the actual acquisitions made the model's top 7 — "
                             "a miss, shown here rather than hidden.")
            else:
                st.markdown("**Ground truth (May 2016):** this customer acquired nothing.")
                st.caption("The ranking above is still a valid 'if they buy, this is what' "
                           "answer; the propensity metric is what says they probably won't.")


# --------------------------------------------------------------------------- #
# Page 2: model performance
# --------------------------------------------------------------------------- #
elif PAGE == "Model performance":
    st.title("Model performance")
    st.markdown("<div class='note'>Trained on Feb–Mar 2016 state with Mar–Apr labels; "
                "evaluated on April 2016 state against <b>May 2016</b> acquisitions — a "
                "strict forward-in-time split, so no future information reaches the "
                "features.</div>", unsafe_allow_html=True)
    st.write("")

    s1m, s2m = RECO_META["stage1"], RECO_META["stage2"]
    c = st.columns(4)
    c[0].metric("Stage 1 ROC-AUC", f"{s1m['roc_auc']:.4f}")
    c[1].metric("Stage 1 PR-AUC", f"{s1m['pr_auc']:.4f}",
                f"{s1m['pr_auc'] / s1m['base_rate']:.0f}× base rate", delta_color="off")
    c[2].metric("MAP@7 (adopters)", f"{RECO_META['map7_adopters']:.4f}",
                f"+{RECO_META['map7_adopters'] - RECO_META['map7_baseline']:.4f} vs baseline")
    c[3].metric("Acquisition base rate", f"{s1m['base_rate'] * 100:.2f}%")

    st.write("")
    fig = go.Figure(go.Bar(
        x=["Popularity baseline", "Two-stage LightGBM"],
        y=[RECO_META["map7_baseline"], RECO_META["map7_adopters"]],
        marker=dict(color=[viz.SERIES_2, viz.SERIES_1], cornerradius=4),
        text=[f"{RECO_META['map7_baseline']:.3f}", f"{RECO_META['map7_adopters']:.3f}"],
        textposition="outside", textfont=dict(size=13, color=viz.TEXT_SECONDARY),
        hovertemplate="%{x}<br>MAP@7 %{y:.4f}<extra></extra>"))
    fig.update_yaxes(title="MAP@7", range=[0, 1.08])
    fig.update_xaxes(title=None)
    fig.update_layout(title="MAP@7 — model vs popularity baseline", bargap=0.55)
    st.plotly_chart(viz.style(fig, height=330, showlegend=False), width="stretch")
    st.markdown(f"""
**Reading the baseline honestly.** The baseline is not random — it recommends the
globally most-acquired products, skipping anything the customer already holds, and
that alone scores **{RECO_META['map7_baseline']:.3f}**. Acquisitions are heavily
concentrated (Direct Debit, Pensions and Payroll account for most of them), so a
popularity list is already strong. The model's **{RECO_META['map7_adopters']:.3f}** is a
{100 * (RECO_META['map7_adopters'] / RECO_META['map7_baseline'] - 1):.0f}% relative
improvement over that, which is the number worth quoting — not the raw MAP@7.

**MAP@7 is measured over adopters only** ({RECO_META['map7_adopters']:.3f}). Across *all*
customers it is {RECO_META['map7_all']:.4f}, because 97% of them buy nothing and score
zero by construction. Ranking quality and purchase likelihood are two different
questions, which is exactly why the model has two stages.
""")

    st.write("")
    st.subheader("Which products the model can and cannot predict")
    acq = T["acq"].copy()
    acq["modelled"] = acq.product_code.isin(s2m["classes"])
    acq["plot_val"] = acq["acquisitions"].clip(lower=0.5)   # log axis cannot show 0
    fig = go.Figure()
    for flag, color, name in [(True, viz.SERIES_1, "Modelled"),
                              (False, "#b4b4b4", "Too few acquisitions to model")]:
        sub = acq[acq.modelled == flag]
        fig.add_trace(go.Bar(
            y=sub["product"], x=sub["plot_val"], orientation="h", name=name,
            marker=dict(color=color, cornerradius=4),
            customdata=sub["acquisitions"],
            hovertemplate="%{y}<br>%{customdata:,} acquisitions<extra></extra>"))
    fig.update_xaxes(title="New acquisitions in May 2016 (log scale; 0 shown at the axis floor)",
                     type="log")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(title="Acquisition volume by product", barmode="stack", bargap=0.3)
    st.plotly_chart(viz.style(fig, height=640), width="stretch")
    st.markdown(f"""
**{len(s2m['classes'])} of 24 products are modelled.** The other 8 are excluded because
they had fewer than 50 training acquisitions — four of them (Guarantees, Short- and
Medium-Term Deposit, Savings Account) had effectively **zero**. A classifier trained on
7 positive examples would produce a number, not a prediction, so those products are
dropped rather than quietly scored. A sales officer asking this model about a mortgage
gets no answer, and that is the correct answer.
""")

    st.write("")
    st.subheader("What drives the predictions")
    t1, t2 = st.tabs(["Stage 1 · will they buy?", "Stage 2 · which product?"])
    with t1:
        i1 = T["imp_s1"].head(12)
        st.plotly_chart(viz.hbar(i1["pretty"], i1["gain"].round(0),
                                 "Top features — acquisition propensity", "LightGBM gain",
                                 color=viz.SERIES_1,
                                 texts=[f"{v:,.0f}" for v in i1["gain"]]), width="stretch")
        st.markdown("""
`dropped_last_month` dominating is worth a second look rather than a shrug. It is
**not leakage** — it compares last month to the current month, both strictly before
the label month. It is real behaviour: several Santander flags (notably Payroll and
Pensions) flicker off and back on, so a customer who dropped a product last month is
unusually likely to "acquire" one next month. A model shipped on this signal would
partly be predicting administrative churn, not genuine demand — the kind of caveat
that belongs in the handover note, not buried in a notebook.
""")
    with t2:
        i2 = T["imp_s2"].head(12)
        st.plotly_chart(viz.hbar(i2["pretty"], i2["gain"].round(0),
                                 "Top features — product choice", "LightGBM gain",
                                 color=viz.SERIES_1,
                                 texts=[f"{v:,.0f}" for v in i2["gain"]]), width="stretch")
        st.markdown("""
Product *choice* is driven by life stage — age, tenure and income — plus how many
products the customer already holds. That is the intuitive result and the one a sales
officer can act on: a 25-year-old with one account and a 55-year-old with six get
different pitches for structural reasons, not because of a spurious correlation.
""")


# --------------------------------------------------------------------------- #
# Page 3: sales worklist
# --------------------------------------------------------------------------- #
elif PAGE == "Sales worklist":
    st.title("Sales worklist")
    st.markdown("<div class='note'>The operational view: score a batch of held-out "
                "customers, rank them by acquisition propensity, and attach the "
                "recommended product to each. This is what a Client Acquisition "
                "officer would actually be handed.</div>", unsafe_allow_html=True)
    st.write("")

    sc = load_sample_customers()
    FEATS = S1["features"]
    for c in CAT_COLS:
        sc[c] = pd.Categorical(sc[c].astype(str), categories=RECO_META["cat_levels"][c])

    n = st.slider("Customers to score", 50, len(sc), 200, 50)
    batch = sc.head(n).copy()

    p1 = S1["model"].predict_proba(batch[FEATS])[:, 1]
    p2 = S2["model"].predict_proba(batch[FEATS])
    classes = S2["classes"]
    owned_mask = batch[[f"own_{p}" for p in PRODUCTS]].to_numpy()
    prod_idx = {p: i for i, p in enumerate(PRODUCTS)}

    scores = p2.copy()
    for j, cls in enumerate(classes):
        scores[:, j] = np.where(owned_mask[:, prod_idx[cls]] == 1, -1.0, scores[:, j])
    best = scores.argmax(axis=1)

    work = pd.DataFrame({
        "Customer": batch.ncodpers.to_numpy(),
        "Age": batch.age.to_numpy(),
        "Tenure (mo)": batch.antiguedad.to_numpy(),
        "Products held": batch.n_products.to_numpy(),
        "P(buys next month)": p1,
        "Recommended product": [PRODUCT_NAMES[classes[b]] for b in best],
        "Product confidence": scores[np.arange(len(batch)), best],
        "Actually bought": ["yes" if v == 1 else "no" for v in batch.y_any.to_numpy()],
    }).sort_values("P(buys next month)", ascending=False).reset_index(drop=True)
    work.index += 1

    top_decile = work.head(max(1, len(work) // 10))
    hit_top = (top_decile["Actually bought"] == "yes").mean()
    hit_all = (work["Actually bought"] == "yes").mean()

    k = st.columns(4)
    k[0].metric("Customers scored", f"{len(work):,}")
    k[1].metric("Top-decile hit rate", f"{hit_top * 100:.0f}%")
    k[2].metric("Overall hit rate", f"{hit_all * 100:.0f}%")
    k[3].metric("Lift in top decile", f"{hit_top / hit_all:.1f}×" if hit_all else "—")

    st.caption("This 300-customer sample is deliberately balanced 50/50 between real "
               "adopters and non-adopters so the table is readable, so the hit rates "
               "here are far above the true 3% base rate. The **lift** is the "
               "meaningful figure — it survives the resampling; the absolute rates do not.")

    st.write("")
    st.dataframe(
        work.style.format({"Age": "{:.0f}", "Tenure (mo)": "{:.0f}",
                           "P(buys next month)": "{:.2%}", "Product confidence": "{:.1%}"})
        .apply(lambda col: viz.blue_gradient_styles(col), subset=["P(buys next month)"]),
        width="stretch", height=460)
    st.download_button("Download worklist as CSV", work.to_csv(index=False).encode(),
                       "sales_worklist.csv", "text/csv")

    st.write("")
    st.subheader("Does the ranking actually work?")
    work["decile"] = pd.qcut(work["P(buys next month)"].rank(method="first", ascending=False),
                             10, labels=range(1, 11))
    # column-wise mean rather than groupby.apply: the include_groups argument
    # needed to silence apply's deprecation only exists on pandas >= 2.2
    work["_hit"] = (work["Actually bought"] == "yes").astype(float)
    g = work.groupby("decile", observed=True)["_hit"].mean()
    fig = go.Figure(go.Bar(
        x=[f"D{i}" for i in g.index], y=(g.to_numpy() * 100).round(1),
        marker=dict(color=viz.SERIES_1, cornerradius=4),
        text=[f"{v*100:.0f}%" for v in g.to_numpy()], textposition="outside",
        textfont=dict(size=12, color=viz.TEXT_SECONDARY),
        hovertemplate="Decile %{x}<br>%{y:.1f}% bought<extra></extra>"))
    fig.add_hline(y=hit_all * 100, line=dict(color=viz.SERIES_2, width=2, dash="dash"),
                  annotation_text="sample average", annotation_position="top right",
                  annotation_font=dict(size=11, color=viz.SERIES_2))
    fig.update_xaxes(title="Propensity decile (D1 = highest scored)")
    fig.update_yaxes(title="% who actually bought", range=[0, max(g.to_numpy() * 100) * 1.25])
    fig.update_layout(title="Actual purchase rate by predicted propensity decile", bargap=0.35)
    st.plotly_chart(viz.style(fig, height=360, showlegend=False), width="stretch")
    st.markdown("A monotone decline left to right is the test of a ranking model: the "
                "customers it scores highest really do buy more often. If this chart were "
                "flat, the ROC-AUC on the previous page would be meaningless.")


# --------------------------------------------------------------------------- #
# Page 4: about
# --------------------------------------------------------------------------- #
else:
    st.title("About this project")
    c = st.columns(4)
    c[0].metric("Customer-months available", "13.6M")
    c[1].metric("Loaded for modelling", "4.6M")
    c[2].metric("Products tracked", "24")
    c[3].metric("Products modelled", len(RECO_META["stage2"]["classes"]))

    st.markdown("""
### The framing decision that defines this project

The obvious label is *does this customer own product X?* It is also useless: a sales
officer cannot sell a product the customer already has, and ownership is trivially
predictable from last month's ownership.

So the label here is a **new acquisition** — the customer did **not** hold product `p`
in month *t*, and **does** hold it in month *t+1*. Ownership at *t* becomes a feature
instead. That makes the problem genuinely hard: only **3.01%** of customers acquire
anything in a given month.

### Two stages, because they answer two different questions

```
Stage 1   P(acquires any new product)      binary, all 926k customers
          → "is this customer worth calling at all?"

Stage 2   P(product = p | acquires)        multiclass, 68k adopter rows
          → "if they buy, what do they buy?"

score(p) = Stage1 × Stage2   →  absolute propensity for ranking a call list
```

A single model cannot serve both: ranking products well says nothing about whether
the customer will buy, and a propensity score says nothing about which product to pitch.

### Temporal validation

| Block | Features from | Labels from |
|---|---|---|
| Train | Feb 2016, Mar 2016 | Mar 2016, Apr 2016 |
| Validation | **Apr 2016** | **May 2016** |

Lag features come from *t−1*. Nothing from the label month ever reaches the features,
and the validation month is strictly after every training month — so the score is a
forecast, not a reconstruction.

### Data quality defects handled

| Defect | Handling |
|---|---|
| `antiguedad` (tenure) uses `-999999` as a missing marker | Set to null |
| `renta` (income) missing on 24% of rows | Left null — LightGBM splits on missingness natively |
| `indrel_1mes` mixes `'1'` and `'1.0'` | Coerced to numeric |
| Ages outside 18–100 | Set to null |
| 8 products with <50 training acquisitions | Excluded from the model entirely |

### Honest limitations

- **`dropped_last_month` is the strongest propensity feature**, and part of what it
  captures is administrative flag-flicker rather than real demand. See *Model performance*.
- **MAP@7 of 0.877 is measured over adopters only.** A popularity baseline already
  scores 0.645 on the same set; the honest headline is the 36% relative gain.
- **8 of 24 products cannot be modelled** from this window — including mortgages and loans.
- **Spanish retail banking, 2015–16.** The patterns are not transferable to another
  bank or market without refitting.
- **Recommendations are propensity scores, not eligibility decisions.** Real product
  offers need consent, suitability and credit checks.
""")
    st.caption("Source: Santander Product Recommendation (Kaggle) — 13.6M monthly "
               "customer snapshots, Jan 2015 – May 2016.")
