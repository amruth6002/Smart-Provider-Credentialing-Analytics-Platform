from dataclasses import dataclass
from typing import Dict, List

# Synonym mapping so we can normalize variable roster schemas.
COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "provider_id": ["provider_id", "id", "prv_id", "provider_identifier"],
    "first_name": ["first_name", "fname", "given_name", "provider_first_name"],
    "last_name": ["last_name", "lname", "surname", "provider_last_name"],
    "full_name": ["full_name", "name", "provider_name"],
    "npi": ["npi", "npi_number", "provider_npi"],
    "license_number": ["license_number", "lic_no", "license", "provider_license_number"],
    "license_state": ["license_state", "state_license", "lic_state", "issuing_state"],
    "license_expiration_date": ["license_expiration_date", "license_expiration", "expiration_date", "expiry", "exp_date"],
    "specialty": ["specialty", "primary_specialty", "taxonomy"],
    "phone": ["phone", "phone_number", "telephone", "contact_phone", "practice_phone"],
    "email": ["email", "email_address"],
    "address_line1": ["address_line1", "address1", "street", "practice_address_line1"],
    "address_city": ["address_city", "city", "practice_city"],
    "address_state": ["address_state", "state", "practice_state"],
    "address_zip": ["address_zip", "zip", "zipcode", "postal_code", "practice_zip"],
}

DATE_COLUMNS = ["license_expiration_date"]

# Scoring weights (sum to 100)
SCORING_WEIGHTS = {
    "license": 35,
    "npi": 25,
    "duplicates": 15,
    "contact_format": 15,
    "mismatches": 10,
}

@dataclass
class Thresholds:
    # Entity resolution
    name_similarity_min: float = 85.0  # 0-100 rapidfuzz ratio
    block_key_len_min: int = 2
    # Phone validity
    require_phone_region: str = "US"

THRESHOLDS = Thresholds()