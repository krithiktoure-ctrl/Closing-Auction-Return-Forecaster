import pandas as pd
import numpy as np

def build_min_features(df, include_bucket=True):
    """
    Build a tiny, intuitive feature set.

    Returns:
        df_feat  : DataFrame with new feature columns added
        FEATURES : list of feature column names to train on
    """
    out = df.copy()

    # Spread & mid
    if {"ask_price", "bid_price"}.issubset(out.columns):
        out["bid_ask_spread"] = (out["ask_price"] - out["bid_price"]).clip(lower=0)
        out["mid_price"] = 0.5 * (out["ask_price"] + out["bid_price"])
    else:
        out["bid_ask_spread"] = 0.0
        out["mid_price"] = out.get("reference_price", 0.0)

    # Signed imbalance
    if {"imbalance_size", "imbalance_buy_sell_flag"}.issubset(out.columns):
        out["signed_imbalance"] = out["imbalance_size"] * out["imbalance_buy_sell_flag"]
    else:
        out["signed_imbalance"] = 0.0

    # Auction prices relative to reference
    for c in ["near_price", "far_price"]:
        new_col = f"{c}_minus_ref"
        if {c, "reference_price"}.issubset(out.columns):
            out[new_col] = out[c] - out["reference_price"]
        else:
            out[new_col] = 0.0

    # Optional: coarse time bucket (0 early, 1 middle, 2 late)
    if include_bucket:
        if "seconds_in_bucket" in out.columns:
            sib = out["seconds_in_bucket"]
            out["sec_bucket_group"] = ((sib >= 300).astype("int8") + (sib >= 480).astype("int8")).astype("int8")
        else:
            out["sec_bucket_group"] = 0

    # Final feature list (keep only existing columns)
    FEATURES = [
        c for c in [
            "seconds_in_bucket",
            "sec_bucket_group" if include_bucket else None,
            "bid_ask_spread", "mid_price", "signed_imbalance",
            "near_price_minus_ref", "far_price_minus_ref",
            "matched_size", "imbalance_size", "bid_size", "ask_size",
            "reference_price", "wap",
        ] if c and c in out.columns
    ]

    return out, FEATURES

def build_features_v2(df, include_bucket=True):
    """
    Returns (df_out, feature_list) using the baseline features plus:
      - depth_imbalance_ratio = (bid_size - ask_size) / (bid_size + ask_size)
      - relative_spread      = bid_ask_spread / mid_price
      - late_signed_imbalance = signed_imbalance if seconds_in_bucket >= 480 else 0
    """
    out, base_feats = build_min_features(df, include_bucket=include_bucket)

    # depth_imbalance_ratio in [-1, 1]
    if {"bid_size", "ask_size"}.issubset(out.columns):
        denom = (out["bid_size"] + out["ask_size"]).replace(0, np.nan)
        out["depth_imbalance_ratio"] = ((out["bid_size"] - out["ask_size"]) / denom).fillna(0.0).clip(-1.0, 1.0)
    else:
        out["depth_imbalance_ratio"] = 0.0

    # relative spread (scale-free)
    if {"bid_ask_spread", "mid_price"}.issubset(out.columns):
        eps = 1e-12
        out["relative_spread"] = out["bid_ask_spread"] / (out["mid_price"].abs() + eps)
        out["relative_spread"] = out["relative_spread"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    else:
        out["relative_spread"] = 0.0

    # late_signed_imbalance (only matters near the close)
    if {"signed_imbalance", "seconds_in_bucket"}.issubset(out.columns):
        late = (out["seconds_in_bucket"] >= 480).astype("int8")
        out["late_signed_imbalance"] = out["signed_imbalance"] * late
    else:
        out["late_signed_imbalance"] = 0.0

    feats = base_feats + ["depth_imbalance_ratio", "relative_spread", "late_signed_imbalance"]
    feats = [c for c in feats if c in out.columns]   # keep only existing columns
    return out, feats


def build_features_v3(df, include_bucket: bool = True):
    """
    Returns (df_out, feature_list) using v2 features plus:
      - si_roll_mean_60:  past-60s rolling mean of signed_imbalance
      - ret1s_roll_std_60: past-60s rolling std of 1-second WAP returns (in bps)
    All rolls are past-only (shift(1)) to avoid leakage.
    """
    # start from v2 baseline
    out, feats = build_features_v2(df, include_bucket=include_bucket)

    # sort so rolling goes forward in time within each day/stock
    out = out.sort_values(["date_id", "stock_id", "seconds_in_bucket"]).reset_index(drop=True)

    # group per (day, stock) so rolls don't mix different stocks/days
    grp = out.groupby(["date_id", "stock_id"], sort=False)

    # 1) rolling mean of signed_imbalance over last 60s (PAST ONLY)
    if "signed_imbalance" in out.columns:
        out["si_roll_mean_60"] = (
            grp["signed_imbalance"]
            .transform(lambda s: s.shift(1).rolling(window=60, min_periods=5).mean())
            .fillna(0.0).astype("float32")
        )
    else:
        out["si_roll_mean_60"] = 0.0

    # 2) rolling std of 1-second WAP returns (in bps), last 60s (PAST ONLY)
    if "wap" in out.columns:
        wap_prev = grp["wap"].shift(1)
        ret1s_bps = ((out["wap"] - wap_prev) / wap_prev) * 1e4
        out["wap_ret1s"] = ret1s_bps.fillna(0.0).astype("float32")

        out["ret1s_roll_std_60"] = (
            out.groupby(["date_id", "stock_id"])["wap_ret1s"]
            .transform(lambda s: s.shift(1).rolling(window=60, min_periods=5).std())
            .fillna(0.0).astype("float32")
        )
    else:
        out["ret1s_roll_std_60"] = 0.0

    # final feature list (drop helper if you don't want it as a feature)
    feats_v3 = feats + ["si_roll_mean_60", "ret1s_roll_std_60"]
    feats_v3 = [c for c in feats_v3 if c in out.columns]
    return out, feats_v3