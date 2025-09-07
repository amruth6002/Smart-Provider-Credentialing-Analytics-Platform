"""
Data context module for providing rich data information to the LLM.
This enables the AI to generate more intelligent, data-aware responses.
"""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class DataContext:
    """Container for data context information to enhance LLM responses"""
    
    # Dataset information
    total_providers: int = 0
    dataset_columns: List[str] = None
    data_sample: pd.DataFrame = None
    
    # Quality metrics
    overall_quality_score: float = 0.0
    quality_breakdown: Dict[str, float] = None
    
    # Processing context
    query_type: str = ""
    processing_steps: List[str] = None
    
    # Data patterns and statistics
    data_stats: Dict[str, Any] = None
    key_findings: List[str] = None
    
    def __post_init__(self):
        if self.dataset_columns is None:
            self.dataset_columns = []
        if self.quality_breakdown is None:
            self.quality_breakdown = {}
        if self.processing_steps is None:
            self.processing_steps = []
        if self.data_stats is None:
            self.data_stats = {}
        if self.key_findings is None:
            self.key_findings = []


class DataContextBuilder:
    """Builder class for creating rich data context from engine data"""
    
    def __init__(self, engine):
        self.engine = engine
        
    def build_context_for_query(self, intent: str, result: Any, params: Dict = None) -> DataContext:
        """Build comprehensive data context for a specific query"""
        context = DataContext()
        
        # Basic dataset information
        if self.engine.aug is not None:
            context.total_providers = len(self.engine.aug)
            context.dataset_columns = self.engine.aug.columns.tolist()
            
            # Provide a small sample of relevant data (first 3 rows, key columns only)
            key_columns = self._get_key_columns_for_intent(intent)
            available_columns = [col for col in key_columns if col in self.engine.aug.columns]
            if available_columns:
                context.data_sample = self.engine.aug[available_columns].head(3).copy()
            
            # Calculate quality metrics
            context.overall_quality_score = round(self.engine.get_quality_score(), 1)
            context.quality_breakdown = self._calculate_quality_breakdown()
            
            # Add processing context
            context.query_type = intent
            context.processing_steps = self._get_processing_steps_for_intent(intent)
            
            # Generate data statistics
            context.data_stats = self._generate_data_stats(intent, result)
            
            # Extract key findings
            context.key_findings = self._extract_key_findings(intent, result)
            
        return context
    
    def _get_key_columns_for_intent(self, intent: str) -> List[str]:
        """Get relevant columns to show for different query types"""
        column_mapping = {
            "expired_license_count": ["full_name", "license_expiration_date", "license_state", "specialty"],
            "phone_format_issues": ["full_name", "practice_phone", "phone_clean", "specialty"],
            "missing_npi": ["full_name", "npi", "specialty", "practice_state"],
            "duplicate_records": ["full_name", "npi", "specialty", "practice_city"],
            "specialties_with_most_issues": ["specialty", "license_expired", "npi_missing", "phone_issue"],
            "state_issue_summary": ["practice_state", "license_expired", "license_found", "npi_missing"],
            "compliance_report_expired": ["full_name", "license_number", "license_expiration_date", "license_state"],
            "filter_by_expiration_window": ["full_name", "license_expiration_date", "specialty", "practice_state"],
            "multi_state_single_license": ["full_name", "practice_state", "license_state", "specialty"],
            "export_update_list": ["full_name", "npi", "specialty", "license_expired", "npi_missing"],
        }
        
        # Default columns if intent not found
        default_columns = ["full_name", "specialty", "practice_state", "npi"]
        
        return column_mapping.get(intent, default_columns)
    
    def _get_processing_steps_for_intent(self, intent: str) -> List[str]:
        """Describe what processing steps were performed for this query"""
        steps = ["Loaded and normalized provider data"]
        
        if intent == "expired_license_count":
            steps.extend([
                "Validated licenses against state databases",
                "Checked license expiration dates",
                "Counted providers with expired licenses"
            ])
        elif intent == "phone_format_issues":
            steps.extend([
                "Standardized phone number formats",
                "Applied phone validation rules",
                "Identified formatting issues"
            ])
        elif intent == "missing_npi":
            steps.extend([
                "Cross-referenced with NPI registry",
                "Identified missing NPI numbers"
            ])
        elif intent == "duplicate_records":
            steps.extend([
                "Applied entity resolution algorithms",
                "Identified potential duplicate providers"
            ])
        elif intent == "specialties_with_most_issues":
            steps.extend([
                "Aggregated quality issues by specialty",
                "Calculated issue counts per specialty",
                "Ranked specialties by issue frequency"
            ])
        
        return steps
    
    def _calculate_quality_breakdown(self) -> Dict[str, float]:
        """Calculate breakdown of quality issues"""
        if self.engine.aug is None:
            return {}
        
        breakdown = {}
        total = len(self.engine.aug)
        
        if total == 0:
            return breakdown
        
        # Calculate percentages for each issue type
        if "license_expired" in self.engine.aug.columns:
            breakdown["expired_licenses"] = round((self.engine.aug["license_expired"].sum() / total) * 100, 1)
        
        if "npi_missing" in self.engine.aug.columns:
            breakdown["missing_npi"] = round((self.engine.aug["npi_missing"].sum() / total) * 100, 1)
        
        if "phone_issue" in self.engine.aug.columns:
            breakdown["phone_issues"] = round((self.engine.aug["phone_issue"].sum() / total) * 100, 1)
        
        if "duplicate_suspect" in self.engine.aug.columns:
            breakdown["duplicates"] = round((self.engine.aug["duplicate_suspect"].sum() / total) * 100, 1)
        
        return breakdown
    
    def _generate_data_stats(self, intent: str, result: Any) -> Dict[str, Any]:
        """Generate relevant statistics for the query"""
        stats = {}
        
        if self.engine.aug is not None:
            total = len(self.engine.aug)
            stats["total_providers"] = total
            
            # Add specific stats based on intent
            if intent == "expired_license_count" and isinstance(result, (int, float)):
                stats["expired_count"] = int(result)
                stats["expired_percentage"] = round((result / total) * 100, 1) if total > 0 else 0
                
            elif intent == "specialties_with_most_issues" and isinstance(result, pd.DataFrame):
                if not result.empty:
                    stats["top_specialty"] = result.iloc[0]["specialty"] if "specialty" in result.columns else "Unknown"
                    stats["top_specialty_issues"] = int(result.iloc[0]["issues"]) if "issues" in result.columns else 0
                    stats["specialties_analyzed"] = len(result)
            
            elif isinstance(result, pd.DataFrame):
                stats["result_count"] = len(result)
                stats["result_percentage"] = round((len(result) / total) * 100, 1) if total > 0 else 0
        
        return stats
    
    def _extract_key_findings(self, intent: str, result: Any) -> List[str]:
        """Extract key insights from the data for this query"""
        findings = []
        
        if self.engine.aug is None:
            return findings
        
        total = len(self.engine.aug)
        
        # Intent-specific findings
        if intent == "expired_license_count" and isinstance(result, (int, float)):
            if result > 0:
                pct = (result / total) * 100 if total > 0 else 0
                if pct > 20:
                    findings.append(f"High rate of expired licenses ({pct:.1f}%) indicates urgent renewal needed")
                elif pct > 10:
                    findings.append(f"Moderate expired license rate ({pct:.1f}%) requires attention")
                else:
                    findings.append(f"Low expired license rate ({pct:.1f}%) shows good compliance")
                
        elif intent == "specialties_with_most_issues" and isinstance(result, pd.DataFrame):
            if not result.empty and "specialty" in result.columns and "issues" in result.columns:
                top_specialty = result.iloc[0]["specialty"]
                top_issues = result.iloc[0]["issues"]
                findings.append(f"'{top_specialty}' specialty has the most quality issues ({top_issues})")
                
                if len(result) > 1:
                    second_specialty = result.iloc[1]["specialty"]
                    findings.append(f"Secondary concern: '{second_specialty}' specialty also shows significant issues")
        
        # Add general quality findings
        quality_score = self.engine.get_quality_score()
        if quality_score < 70:
            findings.append("Overall data quality is below acceptable threshold")
        elif quality_score > 85:
            findings.append("Data quality is in excellent condition")
        
        return findings