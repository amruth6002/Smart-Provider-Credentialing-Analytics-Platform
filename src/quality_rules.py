import pandas as pd

def rule_phone_format(df: pd.DataFrame) -> pd.Series:
    """
    Flag phone numbers with formatting issues.
    This includes:
    1. Numbers that couldn't be cleaned (phone_clean is null)
    2. Numbers with non-standard formatting (missing parentheses, spaces, etc.)
    """
    if "phone" not in df.columns:
        return pd.Series(False, index=df.index)
    
    # Start with numbers that couldn't be cleaned
    issues = df["phone_clean"].isna()
    
    # Also flag numbers with non-standard formatting patterns
    phone_series = df["phone"].fillna("").astype(str)
    
    # Flag phones that don't match standard US patterns but were still cleanable
    # Standard patterns: (XXX) XXX-XXXX, XXX-XXX-XXXX, XXXXXXXXXX
    import re
    standard_patterns = [
        r'^\(\d{3}\)\s*\d{3}[-\s]?\d{4}$',  # (XXX) XXX-XXXX
        r'^\d{3}[-\s]\d{3}[-\s]\d{4}$',     # XXX-XXX-XXXX or XXX XXX XXXX
        r'^\d{10}$'                          # XXXXXXXXXX
    ]
    
    def has_standard_format(phone_str):
        if not phone_str or phone_str.strip() == "":
            return False
        phone_str = phone_str.strip()
        return any(re.match(pattern, phone_str) for pattern in standard_patterns)
    
    # Flag phones that don't match standard patterns (but might still be cleanable)
    non_standard = ~phone_series.apply(has_standard_format) & phone_series.str.strip().astype(bool)
    
    return issues | non_standard

def rule_missing_npi(df: pd.DataFrame) -> pd.Series:
    return df["npi"].isna()

def rule_specialty_missing(df: pd.DataFrame) -> pd.Series:
    return ~df["specialty"].astype(str).str.strip().astype(bool)

def rule_multi_state_single_license(df: pd.DataFrame) -> pd.Series:
    """
    Flag providers appearing in multiple address_state values but having only one license_number.
    Group by provider (NPI preferred, fall back to full_name).
    """
    gkey = df["npi"].fillna(df["full_name_clean"])
    states_per = df.groupby(gkey)["address_state"].nunique()
    lic_per = df.groupby(gkey)["license_number"].nunique()
    flags = (states_per > 1) & (lic_per <= 1)
    return gkey.map(flags).fillna(False)

def summarize_by_state(issues_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = issues_df.select_dtypes(include="bool").columns.tolist()
    grp = issues_df.groupby("address_state")[numeric_cols].sum().reset_index()
    grp["total_records"] = issues_df.groupby("address_state").size().values
    return grp