import pandas as pd
from functools import lru_cache
from typing import Dict, Any
from .data_ingest import load_and_normalize
from .standardize import standardize_frame
from .entity_resolution import find_duplicates
from .validation import validate_licenses, validate_npi
from .quality_rules import rule_phone_format, summarize_by_state, rule_multi_state_single_license
from .scoring import compute_scores, overall_score
from .data_context import DataContextBuilder

class ProviderDQEngine:
    def __init__(self):
        self.roster = None
        self.ny = None
        self.ca = None
        self.npi = None
        self.aug = None         # augmented with validations and rules
        self.dup_pairs = None   # duplicate pair table
        self.data_context_builder = None  # For generating data context

    def load_files(self, roster_path: str, ny_path: str=None, ca_path: str=None, npi_path: str=None):
        self.roster = load_and_normalize(roster_path)
        self.ny = load_and_normalize(ny_path) if ny_path else None
        self.ca = load_and_normalize(ca_path) if ca_path else None
        self.npi = load_and_normalize(npi_path) if npi_path else None
        self._build_index()
        # Initialize data context builder after data is loaded
        self.data_context_builder = DataContextBuilder(self)

    def _build_index(self):
        df = self.roster.copy()
        df = standardize_frame(df)

        # Ensure columns used by rules exist to avoid KeyErrors
        if "phone_clean" not in df.columns:
            df["phone_clean"] = None
        if "npi" not in df.columns:
            df["npi"] = pd.NA

        # License validations (avoid using "or" with DataFrames)
        if (self.ny is not None) or (self.ca is not None):
            ny_df = self.ny if self.ny is not None else pd.DataFrame()
            ca_df = self.ca if self.ca is not None else pd.DataFrame()
            df = validate_licenses(df, ny_df, ca_df)
        else:
            df["license_found"] = False
            df["license_expired"] = False
            df["license_state_mismatch"] = False

        # NPI validation
        if self.npi is not None:
            df = validate_npi(df, self.npi)
        else:
            df["npi_missing"] = df["npi"].isna()
            df["npi_found"] = False

        # Phone issues
        df["phone_issue"] = rule_phone_format(df)

        # Duplicate detection
        self.dup_pairs = find_duplicates(df)
        dup_idx = set(self.dup_pairs["idx_a"]).union(set(self.dup_pairs["idx_b"])) if self.dup_pairs is not None and not self.dup_pairs.empty else set()
        df = df.reset_index(drop=True)
        df["duplicate_suspect"] = df.index.isin(dup_idx)

        # Multi-state single license
        if "address_state" in df.columns:
            df["multi_state_single_license"] = rule_multi_state_single_license(df)
        else:
            df["multi_state_single_license"] = False

        # Scores
        df = compute_scores(df)
        self.aug = df

    # Query methods (deterministic, fast)
    def count_expired(self) -> int:
        return int(self.aug["license_expired"].sum())

    def list_phone_issues(self) -> pd.DataFrame:
        return self.aug[self.aug["phone_issue"]]

    def list_missing_npi(self) -> pd.DataFrame:
        return self.aug[self.aug["npi_missing"]]

    def list_duplicates(self) -> pd.DataFrame:
        if self.dup_pairs is None or self.dup_pairs.empty:
            return pd.DataFrame(columns=["idx_a","idx_b","score","cluster_id","name_a","name_b"])
        base = self.aug.reset_index(drop=True)
        out = self.dup_pairs.copy()
        name_col = "full_name_clean" if "full_name_clean" in base.columns else ("full_name" if "full_name" in base.columns else None)
        if name_col:
            out["name_a"] = out["idx_a"].apply(lambda i: base.loc[i, name_col])
            out["name_b"] = out["idx_b"].apply(lambda i: base.loc[i, name_col])
        else:
            out["name_a"] = out["idx_a"]
            out["name_b"] = out["idx_b"]
        return out.sort_values(["cluster_id","score"], ascending=[True, False])

    def get_quality_score(self) -> float:
        return overall_score(self.aug)

    def specialties_with_most_issues(self) -> pd.DataFrame:
        cols = ["license_expired","npi_missing","phone_issue","duplicate_suspect","license_state_mismatch"]
        tmp = self.aug.copy()
        exist_cols = [c for c in cols if c in tmp.columns]
        tmp["issues"] = tmp[exist_cols].sum(axis=1) if exist_cols else 0
        grp = tmp.groupby("specialty", dropna=False)["issues"].sum().reset_index().sort_values("issues", ascending=False)
        return grp

    def state_issue_summary(self) -> pd.DataFrame:
        if "address_state" not in self.aug.columns:
            return pd.DataFrame()
        return summarize_by_state(self.aug)

    def compliance_report_expired(self) -> pd.DataFrame:
        cols = [c for c in ["full_name","npi","license_number","license_state","license_expiration_date","email_clean","phone_clean","address_city","address_state","address_zip5"] if c in self.aug.columns]
        return self.aug[self.aug["license_expired"]][cols].sort_values("license_expiration_date")

    def filter_by_expiration_window(self, days: int=90) -> pd.DataFrame:
        if "license_expiration_date" not in self.aug.columns:
            return pd.DataFrame()
        today = pd.Timestamp("today").date()
        end = today + pd.Timedelta(days=days)
        mask = (self.aug["license_expiration_date"].notna()) & (self.aug["license_expiration_date"] >= today) & (self.aug["license_expiration_date"] <= end)
        return self.aug[mask].copy()

    def multi_state_single_license(self) -> pd.DataFrame:
        return self.aug[self.aug["multi_state_single_license"]].copy()

    def export_update_list(self) -> pd.DataFrame:
        cols = ["license_expired","npi_missing","phone_issue","duplicate_suspect","license_state_mismatch"]
        exist_cols = [c for c in cols if c in self.aug.columns]
        mask = self.aug[exist_cols].any(axis=1) if exist_cols else pd.Series(False, index=self.aug.index)
        keep = [c for c in ["full_name","npi","license_number","address_state","email_clean","phone_clean","specialty"] if c in self.aug.columns]
        return self.aug[mask][keep].copy()

    def search_provider_by_name(self, name: str) -> pd.DataFrame:
        """Search for providers by name (fuzzy matching)"""
        if not name or self.aug is None:
            return pd.DataFrame()
        
        name = name.strip().lower()
        
        # Search in full_name column if available
        if "full_name" in self.aug.columns:
            mask = self.aug["full_name"].str.lower().str.contains(name, na=False, regex=False)
            if mask.any():
                cols = [c for c in ["full_name", "npi", "primary_specialty", "practice_city", "practice_state", "license_number", "license_state", "practice_phone"] if c in self.aug.columns]
                return self.aug[mask][cols].copy()
        
        # Alternative search in first_name and last_name if full_name not available or no match
        if "first_name" in self.aug.columns and "last_name" in self.aug.columns:
            name_parts = name.split()
            if len(name_parts) >= 2:
                first_name = name_parts[0]
                last_name = name_parts[-1]
                
                mask = (
                    self.aug["first_name"].str.lower().str.contains(first_name, na=False, regex=False) &
                    self.aug["last_name"].str.lower().str.contains(last_name, na=False, regex=False)
                )
                
                if mask.any():
                    cols = [c for c in ["first_name", "last_name", "npi", "primary_specialty", "practice_city", "practice_state", "license_number", "license_state", "practice_phone"] if c in self.aug.columns]
                    return self.aug[mask][cols].copy()
        
        # Return empty DataFrame if no matches found
        return pd.DataFrame()

    # Dispatcher from intents
    def run_query(self, intent: str, params: Dict[str, Any]) -> Any:
        mapping = {
            "expired_license_count": self.count_expired,
            "phone_format_issues": self.list_phone_issues,
            "missing_npi": self.list_missing_npi,
            "duplicate_records": self.list_duplicates,
            "overall_quality_score": self.get_quality_score,
            "specialties_with_most_issues": self.specialties_with_most_issues,
            "state_issue_summary": self.state_issue_summary,
            "compliance_report_expired": self.compliance_report_expired,
            "filter_by_expiration_window": lambda: self.filter_by_expiration_window(params.get("days", 90)),
            "multi_state_single_license": self.multi_state_single_license,
            "export_update_list": self.export_update_list,
            "search_provider_by_name": lambda: self.search_provider_by_name(params.get("name", "")),
        }
        fn = mapping.get(intent)
        if fn is None:
            raise ValueError(f"Unknown intent: {intent}")
        return fn()
    
    def get_data_context_for_query(self, intent: str, result: Any, params: Dict[str, Any] = None):
        """Get rich data context for LLM to generate intelligent responses"""
        if self.data_context_builder is None:
            return None
        return self.data_context_builder.build_context_for_query(intent, result, params)