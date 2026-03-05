from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd

TARGET_COL = "Churn"

def load_telco_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Convert TotalCharges to numeric; blanks become NaN
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace({"": None}), errors="coerce")
    # Map target to 0/1
    df[TARGET_COL] = (df[TARGET_COL].astype(str).str.strip().str.lower() == "yes").astype(int)
    return df

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return X, y
