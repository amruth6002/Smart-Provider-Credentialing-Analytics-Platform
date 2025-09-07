import pandas as pd
from typing import Tuple

def validate_licenses(roster: pd.DataFrame, ny: pd.DataFrame, ca: pd.DataFrame) -> pd.DataFrame:
    """
    Validates roster licenses against NY/CA databases.
    - Flags:
      license_found, license_expired, license_state_mismatch
    """
    df = roster.copy()
    # Consolidate state DBs
    ny_src = ny.copy(); ny_src["validation_state"] = "NY"
    ca_src = ca.copy(); ca_src["validation_state"] = "CA"

    # Normalize expected columns in state DBs
    for s in (ny_src, ca_src):
        if "license_expiration_date" not in s.columns:
            # fallbacks commonly seen in state data
            for c in ["expiration_date", "exp_date", "license_exp"]:
                if c in s.columns:
                    s["license_expiration_date"] = pd.to_datetime(s[c], errors="coerce", utc=True).dt.date
                    break

    state_db = pd.concat([ny_src, ca_src], ignore_index=True)
    
    # Remove duplicates by license_number to prevent row multiplication
    # Keep the first occurrence of each license number
    state_db = state_db.drop_duplicates(subset=["license_number"], keep="first")

    # Join by license_number where available
    joined = df.merge(
        state_db[["license_number", "validation_state", "license_expiration_date"]],
        on="license_number", how="left", suffixes=("", "_state")
    )

    # Heuristic state match
    if "license_state" in joined.columns:
        joined["license_state_mismatch"] = (
            (joined["validation_state"].notna()) &
            (joined["license_state"].notna()) &
            (joined["validation_state"] != joined["license_state"])
        )
    else:
        joined["license_state_mismatch"] = False

    # Found flag
    joined["license_found"] = joined["validation_state"].notna()

    # Expired flag: prefer state expiration if present, else roster field
    date_roster = joined.get("license_expiration_date")
    date_state = joined.get("license_expiration_date_state")
    # In case merge created only one column
    if "license_expiration_date_state" not in joined.columns and "license_expiration_date" in state_db.columns:
        # Avoid overriding roster date; compute state date by re-merge
        # Use the deduplicated state_db to prevent row multiplication
        tmp = df[["license_number"]].merge(
            state_db[["license_number", "license_expiration_date"]],
            on="license_number", how="left"
        )["license_expiration_date"]
        date_state = tmp
    today = pd.Timestamp("today").date()
    best_exp = date_state.fillna(date_roster) if date_state is not None else date_roster
    joined["license_expired"] = best_exp.apply(lambda d: bool(d and d < today))

    return joined

def validate_npi(roster: pd.DataFrame, npi: pd.DataFrame) -> pd.DataFrame:
    """
    Validates NPI presence and linkage by direct join.
    Flags: npi_missing, npi_found
    """
    df = roster.copy()
    result = df.merge(npi[["npi"]].drop_duplicates(), on="npi", how="left", indicator=True)
    result["npi_missing"] = result["npi"].isna()
    result["npi_found"] = (result["_merge"] == "both")
    result = result.drop(columns=["_merge"])
    return result