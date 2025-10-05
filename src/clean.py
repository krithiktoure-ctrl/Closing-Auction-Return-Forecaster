import pandas as pd               


NUMERIC_LIKE = [
    "stock_id", "date_id", "seconds_in_bucket",
    "imbalance_size", "imbalance_buy_sell_flag",
    "reference_price", "matched_size",
    "far_price", "near_price",
    "bid_price", "bid_size", "ask_price", "ask_size",
    "wap", "target"
]

def basic_clean(
    df: pd.DataFrame,
    target_col: str = "target",
    impute: str = "zero") -> pd.DataFrame:
    """
    Make a simple, safe cleaned table:
      1) turn numeric-like columns into real numbers (bad strings -> NaN),
      2) drop rows with no target (for training),
      3) fill other numeric NaNs using the chosen method,
      4) clip sizes so they are >= 0,
      5) sort rows by date -> stock -> seconds (chronological order).
    """
    out = df.copy()

    
    for c in NUMERIC_LIKE:
        if c in out.columns:     
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if target_col in out.columns:
        out = out.dropna(subset=[target_col])

    
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    if impute == "zero":
        fill_values = {c: 0.0 for c in num_cols}
    elif impute == "mean":
        fill_values = out[num_cols].mean(numeric_only=True).to_dict()
    elif impute == "median":
        fill_values = out[num_cols].median(numeric_only=True).to_dict()
    elif impute == "mode":
        mode_row = out[num_cols].mode(dropna=True).iloc[0]
        fill_values = mode_row.to_dict()
    else:
        raise ValueError("impute must be 'zero', 'mean', 'median', or 'mode'")
    out[num_cols] = out[num_cols].fillna(fill_values)

    
    for c in ["matched_size", "imbalance_size", "bid_size", "ask_size"]:
        if c in out.columns:
            out[c] = out[c].clip(lower=0)

    sort_cols = [c for c in ["date_id", "stock_id", "seconds_in_bucket"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(by=sort_cols).reset_index(drop=True)

    return out
