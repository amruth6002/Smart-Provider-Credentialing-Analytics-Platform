from typing import Tuple, Dict
from .intents import INTENT_PATTERNS, extract_params

def parse_intent(text: str) -> Tuple[str, Dict]:
    for intent, patterns in INTENT_PATTERNS.items():
        for p in patterns:
            if p.search(text):
                params = extract_params(intent, text)
                return intent, params
    # Fallbacks: simple keywords
    t = text.lower()
    if "expired" in t and "license" in t and "how many" in t:
        return "expired_license_count", {}
    if "duplicate" in t:
        return "duplicate_records", {}
    if "quality score" in t:
        return "overall_quality_score", {}
    if "phone" in t and ("issue" in t or "format" in t):
        return "phone_format_issues", {}
    return "overall_quality_score", {}