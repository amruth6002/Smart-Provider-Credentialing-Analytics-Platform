import pandas as pd
import phonenumbers
from dateutil import parser

def clean_phone(s: pd.Series, region="US") -> pd.Series:
    def _fmt(x):
        if pd.isna(x) or str(x).strip()=="":
            return None
        try:
            num = phonenumbers.parse(str(x), region)
            if phonenumbers.is_valid_number(num):
                return phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.E164)
            return None
        except Exception:
            return None
    return s.apply(_fmt)

def clean_email(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().where(lambda x: x.str.contains(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", regex=True), None)

def clean_name(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

def clean_zip(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d{5})")[0]

def ensure_dates(s: pd.Series) -> pd.Series:
    def _parse(x):
        if pd.isna(x) or str(x).strip()=="":
            return None
        try:
            return parser.parse(str(x)).date()
        except Exception:
            return None
    return s.apply(_parse)

def standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "phone" in out.columns:
        out["phone_clean"] = clean_phone(out["phone"])
    if "email" in out.columns:
        out["email_clean"] = clean_email(out["email"])
    if "full_name" in out.columns:
        out["full_name_clean"] = clean_name(out["full_name"])
    if "address_zip" in out.columns:
        out["address_zip5"] = clean_zip(out["address_zip"])
    return out