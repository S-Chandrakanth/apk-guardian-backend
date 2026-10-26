"""Shared constants for feature extraction, dataset validation, and training.

Training and scanning import the SAME definitions from here so the feature
order and meaning cannot drift between the two.
"""

from __future__ import annotations

# The eight static features, in the exact column order used for training and prediction.
FEATURE_NAMES: tuple[str, ...] = (
    "permissions_count",
    "activities_count",
    "services_count",
    "receivers_count",
    "api_calls_count",
    "strings_count",
    "dangerous_permissions",
    "suspicious_api_calls",
)

# Human-readable meaning of each feature (also written into feature_schema.json).
FEATURE_DEFINITIONS: dict[str, str] = {
    "permissions_count": "Unique permission names in <uses-permission>/<uses-permission-sdk-23>.",
    "activities_count": "Unique <activity> and <activity-alias> names declared in the manifest.",
    "services_count": "Unique <service> names declared in the manifest.",
    "receivers_count": "Unique <receiver> names declared in the manifest.",
    "api_calls_count": (
        "Distinct EXTERNAL method references in the DEX files (a static reference count; "
        "NOT the number of calls executed at runtime)."
    ),
    "strings_count": "Unique entries in the DEX string pool(s), across all classes*.dex files.",
    "dangerous_permissions": "Requested permissions that appear in the pinned dangerous-permission list.",
    "suspicious_api_calls": "Distinct external method references matching the pinned suspicious-API list.",
}

# Features that need the DEX files / the manifest respectively (used for PARTIAL status).
MANIFEST_FEATURES: tuple[str, ...] = (
    "permissions_count",
    "activities_count",
    "services_count",
    "receivers_count",
    "dangerous_permissions",
)
DEX_FEATURES: tuple[str, ...] = ("api_calls_count", "strings_count", "suspicious_api_calls")

# Base extractor version. The full version string also carries the pinned-list versions
# (see app.ml.feature_lists.extractor_version) so changing a list changes the version.
EXTRACTOR_BASE_VERSION = "1.0.0"

# Training labels.
LABEL_BENIGN = "BENIGN"
LABEL_MALWARE = "MALWARE"
VALID_LABELS: tuple[str, ...] = (LABEL_BENIGN, LABEL_MALWARE)

# Extraction statuses.
STATUS_OK = "OK"
STATUS_PARTIAL = "PARTIAL"
STATUS_FAILED = "FAILED"
VALID_STATUSES: tuple[str, ...] = (STATUS_OK, STATUS_PARTIAL, STATUS_FAILED)

# Manifest (dataset table) layout.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "sha256",
    "label",
    "original_category",
    "label_source",
    "extractor_version",
    "extraction_status",
)
MANIFEST_COLUMNS: tuple[str, ...] = (*IDENTITY_COLUMNS, *FEATURE_NAMES, "error_code", "error_message")
REQUIRED_COLUMNS: tuple[str, ...] = (*IDENTITY_COLUMNS, *FEATURE_NAMES)

# Label sources that mark data as test-only. Training refuses these unless explicitly allowed.
SYNTHETIC_LABEL_SOURCE_PREFIX = "UNIT_TEST_SYNTHETIC"
