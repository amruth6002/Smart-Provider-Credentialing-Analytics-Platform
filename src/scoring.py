import pandas as pd
from .config import SCORING_WEIGHTS

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Booleans default to False if missing
    for col in ["license_expired", "license_found", "npi_missing", "phone_issue", "duplicate_suspect", "license_state_mismatch"]:
        if col not in out.columns:
            out[col] = False

    penalties = (
        SCORING_WEIGHTS["license"] * ((~out["license_found"]) | (out["license_expired"]) | (out["license_state_mismatch"])) +
        SCORING_WEIGHTS["npi"] * (out["npi_missing"]) +
        SCORING_WEIGHTS["duplicates"] * (out["duplicate_suspect"]) +
        SCORING_WEIGHTS["contact_format"] * (out["phone_issue"]) +
        SCORING_WEIGHTS["mismatches"] * 0  # placeholder for extra mismatch rules
    )

    out["dq_score"] = (100 - penalties).clip(lower=0)
    return out

def overall_score(df: pd.DataFrame) -> float:
    if "dq_score" not in df.columns or df["dq_score"].empty:
        return 0.0
    return float(df["dq_score"].mean())