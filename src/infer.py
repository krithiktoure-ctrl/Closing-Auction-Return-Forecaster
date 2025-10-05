# src/infer.py
from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from .features import build_features_v3

ART = Path(__file__).resolve().parents[1] / "artifacts"

def load_artifacts(art_dir: Path = ART):
    meta_path = art_dir / "cv_meta_v3.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Re-save artifacts from the notebook.")
    meta = json.loads(meta_path.read_text())
    best_fold = int(meta.get("best_fold", 1))

    model_path = next((p for p in art_dir.glob(f"lgbm_v3_best_fold{best_fold}.pkl")), None)
    if model_path is None:
        picks = sorted(art_dir.glob("lgbm_v3_best_fold*.pkl"))
        if not picks:
            raise FileNotFoundError("No v3 model artifact found in artifacts/.")
        model_path = picks[0]

    model = joblib.load(model_path)

    # Prefer the JSON feature list IF its length matches the model feature count
    n_model_feats = None
    try:
        n_model_feats = int(model.booster_.num_feature())
    except Exception:
        pass

    feat_json = art_dir / "features_v3.json"
    if feat_json.exists():
        features_json = json.loads(feat_json.read_text())
        if n_model_feats is None or len(features_json) == n_model_feats:
            features = list(features_json)
        else:
            # fall back to model's own feature names
            features = list(model.booster_.feature_name())
    else:
        features = list(model.booster_.feature_name())

    return model, features, meta



def prepare_features(df_raw: pd.DataFrame):
    """
    Build v3 features from raw Optiver columns.
    Returns (df_feat, feature_names_from_builder).
    Keeps id columns (date_id, stock_id, seconds_in_bucket, target) if present.
    Ensures NO duplicate column names.
    """
    df_feat, feats = build_features_v3(df_raw, include_bucket=True)

    keep = [c for c in ["date_id", "stock_id", "seconds_in_bucket", "target"] if c in df_feat.columns]
    # avoid duplicates: remove any features that are also in 'keep'
    feats_no_dup = [c for c in feats if c not in keep]

    if keep:
        df_feat = pd.concat([df_feat[keep], df_feat[feats_no_dup]], axis=1)
    else:
        df_feat = df_feat[feats_no_dup]

    # final safety: drop duplicated column names, keep first occurrence
    df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()]

    return df_feat, feats_no_dup


def predict_df(model, df_feat: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """
    Align df_feat to the exact training feature list 'feats':
      - add any missing columns as 0.0
      - preserve exact order for model input
    Adds 'prediction' (and 'abs_error' if 'target' exists).
    """
    for c in feats:
        if c not in df_feat.columns:
            df_feat[c] = 0.0

    X = df_feat[feats].astype("float32").values  # exact order expected by the model
    preds = model.predict(X)

    out = df_feat.copy()
    out["prediction"] = preds.astype("float32")
    if "target" in out.columns:
        out["abs_error"] = (out["target"] - out["prediction"]).abs()

    return out
