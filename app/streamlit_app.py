from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st

# ---------- make sure "src/" is importable ----------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infer import load_artifacts, prepare_features, predict_df  # noqa: E402
# ----------------------------------------------------

# ---------- light, compact theme tweaks ----------
st.set_page_config(page_title="Closing Auction Return Forecaster", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top:1.4rem; padding-bottom:3rem;}
      [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
      [data-testid="stDataFrame"] div[role="table"] {font-size:0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Closing Auction Return Forecaster")

# ---------- load v3 artifacts ----------
# prefer artifacts/; if you keep a v3 folder, this also works
ART_V3 = (ROOT / "artifacts") if (ROOT / "artifacts").exists() else (ROOT / "artifacts_v3")

@st.cache_resource(show_spinner=False)
def _load_v3():
    return load_artifacts(ART_V3)

model, FEATURES, cv_meta = _load_v3()

# top chips
m1, m2, m3 = st.columns([1.4, 1.2, 5])

with m1:
    # Render the model name as a compact HTML chip (not st.metric)
    st.markdown(
        """
        <div style="font-size:0.9rem;color:#a8a8a8;margin-bottom:0.15rem;">Model</div>
        <div style="
            display:inline-block;
            padding:0.28rem 0.65rem;
            border-radius:999px;
            background:rgba(180,200,255,0.12);
            border:1px solid rgba(180,200,255,0.35);
            font-weight:600;
            font-size:1.05rem;
            white-space:nowrap;">
            LightGBM&nbsp;v3
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.metric(
        "CV MAE (bps)",
        f"{cv_meta.get('cv_mae_mean', float('nan')):.3f}",
        f"±{cv_meta.get('cv_mae_std', float('nan')):.3f}",
    )

with m3:
    st.caption(f"Features: {len(FEATURES)} • Best fold: {cv_meta.get('best_fold','?')}")

st.markdown(
    """
    <style>
      .block-container {padding-top:1.4rem; padding-bottom:3rem;}
      /* make metric values slightly smaller and tabular for alignment */
      [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums; font-size: 2.0rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- controls (no sidebar) ----------
cL, cR = st.columns([3, 2])
with cL:
    uploaded = st.file_uploader("Upload CSV (raw Optiver columns)", type=["csv"])
with cR:
    nrows = st.number_input("Rows to read (sample)", min_value=1_000, max_value=200_000,
                            step=1_000, value=20_000)

# ---------- load RAW data ----------
if uploaded is not None:
    df_raw = pd.read_csv(uploaded, nrows=int(nrows))
else:
    sample_path = ROOT / "data" / "train.csv"
    if not sample_path.exists():
        st.warning("Upload a CSV or place a sample at data/train.csv")
        st.stop()
    df_raw = pd.read_csv(sample_path, nrows=int(nrows))

st.caption(f"Raw shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")

# ---------- build features & predict ----------
with st.spinner("Building features and predicting..."):
    df_feat, feats_built = prepare_features(df_raw)
    df_pred = predict_df(model, df_feat.copy(), FEATURES)  # aligns to model feature list

# display-safe (no duplicate column names)
df_show = df_pred.loc[:, ~df_pred.columns.duplicated()]

# ---------- tabs ----------
tab_overview, tab_timeline, tab_groups, tab_explain, tab_whatif, tab_export = st.tabs(
    ["Overview", "Timeline", "Group metrics", "Explain", "What-if", "Export"]
)

# ===== OVERVIEW =====
with tab_overview:
    if "abs_error" in df_pred.columns:
        st.metric("MAE on this data (bps)", f"{df_pred['abs_error'].mean():.3f}")

    st.subheader("Preview (first 10 rows)")
    st.dataframe(df_show.head(10), use_container_width=True)

    # residuals
    if "abs_error" in df_pred.columns:
        st.subheader("Residual histogram (|target - prediction|, bps)")
        hist, bins = np.histogram(df_pred["abs_error"].astype(float), bins=50)
        centers = 0.5 * (bins[1:] + bins[:-1])
        st.area_chart(pd.DataFrame({"count": hist, "bin_center": centers}).set_index("bin_center"))

    # scatter sample
    if "target" in df_pred.columns:
        st.subheader("Prediction vs Target (sample)")
        samp = df_pred.sample(min(5000, len(df_pred)), random_state=0)
        st.scatter_chart(
            samp.rename(columns={"prediction": "Prediction", "target": "Target"})[
                ["Prediction", "Target"]
            ]
        )

    # feature importance (gain)
    try:
        gain = model.booster_.feature_importance("gain")
        names = model.booster_.feature_name()
        fi = pd.DataFrame({"feature": names, "gain": gain}).sort_values("gain", ascending=False)
        st.subheader("Feature importance (gain)")
        st.bar_chart(fi.set_index("feature").head(20))
    except Exception:
        pass

# ===== TIMELINE =====
with tab_timeline:
    st.write("Per-stock timeline across the last 10 minutes.")
    needed = ["stock_id", "date_id", "seconds_in_bucket"]
    if all(c in df_pred.columns for c in needed):
        a, b = st.columns(2)
        stock_options = sorted(df_pred["stock_id"].dropna().unique().tolist())
        sid = a.selectbox("stock_id", stock_options, index=0)

        date_options = sorted(df_pred.loc[df_pred["stock_id"] == sid, "date_id"].dropna().unique().tolist())
        did = b.selectbox("date_id", date_options, index=0)

        sub = df_pred[(df_pred["stock_id"] == sid) & (df_pred["date_id"] == did)].copy()
        sub = sub.sort_values("seconds_in_bucket")
        st.caption(f"Rows: {len(sub):,}")

        lines = {"Prediction": sub["prediction"].astype(float)}
        if "target" in sub.columns:
            lines["Target"] = sub["target"].astype(float)

        st.line_chart(pd.DataFrame(lines, index=sub["seconds_in_bucket"].astype(int)), height=280)
    else:
        st.info("Needs columns: stock_id, date_id, seconds_in_bucket")

# ===== GROUP METRICS =====
with tab_groups:
    if "abs_error" in df_pred.columns:
        g1, g2 = st.columns(2)
        worst_stocks = (
            df_pred.groupby("stock_id")["abs_error"].mean()
            .sort_values(ascending=False).reset_index(name="mae").head(20)
        )
        worst_dates = (
            df_pred.groupby("date_id")["abs_error"].mean()
            .sort_values(ascending=False).reset_index(name="mae").head(20)
        )
        with g1:
            st.write("Top-20 worst **stocks** by MAE")
            st.dataframe(worst_stocks, use_container_width=True)
        with g2:
            st.write("Top-20 worst **dates** by MAE")
            st.dataframe(worst_dates, use_container_width=True)
    else:
        st.info("This tab needs a 'target' column to compute errors.")

# ===== EXPLAIN (optional SHAP) =====
with tab_explain:
    st.write("Local explanation for a single row (top SHAP contributions).")
    try:
        import shap  # install with: pip install shap
        explainer = shap.TreeExplainer(model.booster_)
        idx = st.number_input("Row index to explain", min_value=0, max_value=len(df_feat)-1, step=1, value=0)
        x = df_feat.loc[idx, FEATURES].astype("float32").values.reshape(1, -1)
        shap_vals = explainer.shap_values(x)[0]
        contrib = pd.DataFrame({"feature": FEATURES, "shap": shap_vals}).sort_values(
            "shap", key=np.abs, ascending=False
        )
        k = st.slider("Top features to show", 5, min(20, len(FEATURES)), 10)
        st.bar_chart(contrib.head(k).set_index("feature")["shap"])
        st.caption("Positive SHAP ↑ prediction; negative SHAP ↓ prediction.")
    except Exception as e:
        st.info("Install SHAP to enable this tab: `pip install shap`")
        st.caption(f"(Reason: {e})")

# ===== WHAT-IF =====
with tab_whatif:
    st.write("Pick a row, tweak a few features, and see the new prediction.")
    idx = st.number_input("Row index to edit", min_value=0, max_value=len(df_feat)-1, step=1, value=0)
    row = df_feat.loc[idx, :].copy()
    baseline_pred = float(df_pred.loc[idx, "prediction"])

    editable = [f for f in [
        "signed_imbalance", "near_price_minus_ref", "far_price_minus_ref",
        "relative_spread", "si_roll_mean_60", "ret1s_roll_std_60"
    ] if f in FEATURES]

    if not editable:
        st.info("No editable features found in this model.")
    else:
        cols = st.columns(min(3, len(editable)))
        new_vals = {}
        for i, feat in enumerate(editable):
            col = cols[i % len(cols)]
            v = float(row.get(feat, 0.0))
            qlo, qhi = np.nanpercentile(df_feat[feat].astype(float), [5, 95])
            new_vals[feat] = col.slider(feat, float(qlo), float(qhi), float(v))
        tweaked = row.copy()
        for k2, v2 in new_vals.items():
            tweaked[k2] = v2

        for c in FEATURES:
            if c not in tweaked.index:
                tweaked[c] = 0.0
        X1 = tweaked[FEATURES].astype("float32").values.reshape(1, -1)
        new_pred = float(model.predict(X1)[0])

        w1, w2 = st.columns(2)
        w1.metric("Baseline prediction (bps)", f"{baseline_pred:.3f}")
        w2.metric("What-if prediction (bps)", f"{new_pred:.3f}", f"{new_pred - baseline_pred:+.3f}")

# ===== EXPORT =====
with tab_export:
    st.write("Download predictions with ids & (if available) target.")
    st.download_button(
        "Download predictions.csv",
        data=df_pred.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )
