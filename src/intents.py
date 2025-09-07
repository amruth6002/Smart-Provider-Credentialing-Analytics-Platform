from typing import Dict, List, Pattern
import re

# Simple rule-based patterns for hackathon-level NLU
INTENT_PATTERNS: Dict[str, List[Pattern]] = {
    "expired_license_count": [
        re.compile(r"\bhow many\b.*\bexpired license", re.I),
        re.compile(r"\bexpired licenses\b.*\bcount\b", re.I),
    ],
    "phone_format_issues": [
        re.compile(r"\bphone\b.*(format|invalid|issue|problem)", re.I),
    ],
    "missing_npi": [
        re.compile(r"\bmissing\b.*\bnpi\b", re.I),
        re.compile(r"\bwhich\b.*\bnpi\b.*\bmissing\b", re.I),
    ],
    "duplicate_records": [
        re.compile(r"\bduplicate\b.*(record|provider)", re.I),
        re.compile(r"\bpotential duplicate", re.I),
    ],
    "overall_quality_score": [
        re.compile(r"\boverall\b.*\bquality score\b", re.I),
        re.compile(r"\bdata quality score\b", re.I),
    ],
    "specialties_with_most_issues": [
        re.compile(r"\bspecialt(y|ies)\b.*\bmost\b.*(issue|problem)", re.I),
    ],
    "state_issue_summary": [
        re.compile(r"\bsummary\b.*\b(state|by state)\b", re.I),
    ],
    "compliance_report_expired": [
        re.compile(r"\bcompliance report\b.*\bexpired\b", re.I),
    ],
    "filter_by_expiration_window": [
        re.compile(r"\bfilter\b.*\bexpiration\b.*\b(\d+)\s*days\b", re.I),
        re.compile(r"\bexpire(s|d)?\b.*\bnext\b.*\b(\d+)\b\s*days", re.I),
    ],
    "multi_state_single_license": [
        re.compile(r"\bmultiple states\b.*single license\b", re.I),
        re.compile(r"\bpracticing\b.*\bmultiple states\b", re.I),
        re.compile(r"\bproviders?\b.*\bmultiple states\b.*\bsingle license", re.I),
        re.compile(r"\bsingle license\b.*\bmultiple states\b", re.I),
    ],
    "export_update_list": [
        re.compile(r"\bexport\b.*(update|credential)", re.I),
        re.compile(r"\blist\b.*\bproviders?\b.*\brequiring\b.*\b(update|credential)", re.I),
        re.compile(r"\bproviders?\b.*\brequiring\b.*\b(update|credential)", re.I),
        re.compile(r"\bcredential\b.*\bupdate", re.I),
    ],
    "search_provider_by_name": [
        re.compile(r"\bis\s+(their|there|they)\s*(anyone|anybody|somebody|someone)\s+.*\b([a-zA-Z]+\s+[a-zA-Z]+)\b.*\bname", re.I),
        re.compile(r"\bfind\b.*\bprovider\b.*\bnamed?\b.*\b([a-zA-Z]+\s+[a-zA-Z]+)", re.I),
        re.compile(r"\bsearch\b.*\bfor\b.*\b([a-zA-Z]+\s+[a-zA-Z]+)", re.I),
        re.compile(r"\blook\s+for\b.*\b([a-zA-Z]+\s+[a-zA-Z]+)", re.I),
        re.compile(r"\bdo\s+we\s+have\b.*\b([a-zA-Z]+\s+[a-zA-Z]+)", re.I),
        re.compile(r"\bshow\s+me\b.*\b([a-zA-Z]+\s+[a-zA-Z]+)", re.I),
        re.compile(r"^([a-zA-Z]+\s+[a-zA-Z]+)\s+in\s+(the\s+)?(dataset|data)", re.I),
        re.compile(r"\b([a-zA-Z]+\s+[a-zA-Z]+)\b.*\bin\s+(the\s+)?(dataset|data)", re.I),
    ],
}

def extract_params(intent: str, text: str):
    # Only one param used in this demo: days window for expiration
    if intent == "filter_by_expiration_window":
        m = re.search(r"(\d+)\s*days", text, re.I)
        if m:
            return {"days": int(m.group(1))}
        return {"days": 90}
    elif intent == "search_provider_by_name":
        # Extract name patterns from various query formats
        patterns = [
            r"\bis\s+(?:their|there|they)\s*(?:anyone|anybody|somebody|someone)\s+.*\b([a-zA-Z]+\s+[a-zA-Z]+)\b.*\bname",
            r"\bfind\b.*\bprovider\b.*\bnamed?\b.*\b([a-zA-Z]+\s+[a-zA-Z]+)",
            r"\bsearch\b.*\bfor\b.*\b([a-zA-Z]+\s+[a-zA-Z]+)",
            r"\blook\s+for\b.*\b([a-zA-Z]+\s+[a-zA-Z]+)",
            r"\bdo\s+we\s+have\b.*\b([a-zA-Z]+\s+[a-zA-Z]+)",
            r"\bshow\s+me\b.*\b([a-zA-Z]+\s+[a-zA-Z]+)",
            r"^([a-zA-Z]+\s+[a-zA-Z]+)\s+in\s+(?:the\s+)?(?:dataset|data)",
            r"\b([a-zA-Z]+\s+[a-zA-Z]+)\b.*\bin\s+(?:the\s+)?(?:dataset|data)",
        ]
        
        for pattern in patterns:
            m = re.search(pattern, text, re.I)
            if m:
                name = m.group(1).strip()
                return {"name": name}
        
        # Fallback: try to extract any two consecutive words that could be a name
        m = re.search(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", text)
        if m:
            return {"name": m.group(1)}
        
        return {"name": ""}
    return {}