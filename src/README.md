# CloseMove — Closing Auction Forecaster

Predicts **last-10-minute, index-neutral price moves** (in bps) for NASDAQ-listed stocks using a blend of
closing-auction and top-of-book signals. Trained with cross-validation and served with a Streamlit app.

![app](artifacts/screenshot.png) <!-- optional: add a screenshot path or delete this line -->

---

## Highlights

- **LightGBM v3** model with strong offline CV (≈ **5.24 bps MAE** on my splits).
- Feature set mixes microstructure signals (imbalance, near/far vs reference, spread) + two **leak-safe rolling** stats.
- Clean, resume-ready **Streamlit app**:
  - overview + residuals
  - per-stock **timeline** for the last 10 minutes
  - **group metrics** (worst stocks/dates)
  - **Explain** (SHAP) + **What-if** sliders for scenario testing
  - one-click **CSV export** of predictions

---

## Repo structure

