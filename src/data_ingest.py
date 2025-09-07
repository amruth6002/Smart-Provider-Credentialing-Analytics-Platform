import pandas as pd
import re
from typing import Dict, List
from .config import COLUMN_SYNONYMS, DATE_COLUMNS

def _normalize_columns(df: pd.DataFrame, synonyms: Dict[str, List[str]]) -> pd.DataFrame:
    lower_map = {c.lower(): c for c in df.columns}
    rename = {}
    for std_col, syns in synonyms.items():
        for s in syns:
            if s.lower() in lower_map:
                rename[lower_map[s.lower()]] = std_col
                break
    out = df.rename(columns=rename).copy()
    # If no full_name, synthesize
    if "full_name" not in out.columns:
        if {"first_name", "last_name"}.issubset(out.columns):
            out["full_name"] = (out["first_name"].fillna("") + " " + out["last_name"].fillna("")).str.strip()
    return out

def _to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    for c in DATE_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True).dt.date
    return df

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_and_normalize(path: str) -> pd.DataFrame:
    df = load_csv(path)
    df = _normalize_columns(df, COLUMN_SYNONYMS)
    df = _to_datetime(df)
    return df

def add_state_tag(df: pd.DataFrame, state: str) -> pd.DataFrame:
    df = df.copy()
    df["validation_source_state"] = state
    return df