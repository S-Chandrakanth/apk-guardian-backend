"""SQLAlchemy models. Constraints are enforced by the database itself, not only by the API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

SCAN_STATUSES = (
    "COMPLETED",           # extraction OK and a compatible model produced a prediction
    "MODEL_UNAVAILABLE",   # extraction OK, but no usable trained model: no prediction made
    "EXTRACTION_PARTIAL",  # manifest read, DEX unreadable: features incomplete, no prediction
    "EXTRACTION_FAILED",   # file could not be analysed
    "PREDICTION_FAILED",   # model present but prediction raised an error
)
CLASSIFICATIONS = ("BENIGN", "SUSPICIOUS", "MALICIOUS", "NEEDS REVIEW")
RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
EXTRACTION_STATUSES = ("OK", "PARTIAL", "FAILED")
SCHEMA_VERSION = 1


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(254), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    scans: Mapped[list["Scan"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        CheckConstraint("length(trim(name)) > 0", name="ck_users_name_not_blank"),
        CheckConstraint("email = lower(email) AND length(email) > 3", name="ck_users_email_lowercase"),
    )


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    apk_name: Mapped[str] = mapped_column(String(255), nullable=False)
    package_name: Mapped[Optional[str]] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    scan_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    scan_status: Mapped[str] = mapped_column(String(24), nullable=False)
    extraction_status: Mapped[Optional[str]] = mapped_column(String(8))
    error_code: Mapped[Optional[str]] = mapped_column(String(40))

    classification: Mapped[Optional[str]] = mapped_column(String(16))
    prediction: Mapped[Optional[str]] = mapped_column(String(8))  # raw model label, not the final classification
    benign_probability: Mapped[Optional[float]] = mapped_column(Float)
    malicious_probability: Mapped[Optional[float]] = mapped_column(Float)
    risk_score: Mapped[Optional[int]] = mapped_column(Integer)
    risk_level: Mapped[Optional[str]] = mapped_column(String(10))
    risk_basis: Mapped[Optional[str]] = mapped_column(String(24))

    # Identity of everything a stored result depends on (used to decide if it can be reused).
    extractor_version: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_schema_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    model_sha256: Mapped[Optional[str]] = mapped_column(String(64))
    risk_logic_version: Mapped[str] = mapped_column(String(32), nullable=False)
    reused_previous_result: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Internal JSON: extraction result + model output. Not returned to clients as-is.
    result_json: Mapped[str] = mapped_column(Text, nullable=False)

    user: Mapped[User] = relationship(back_populates="scans")
    report: Mapped[Optional["Report"]] = relationship(
        back_populates="scan", cascade="all, delete-orphan", passive_deletes=True, uselist=False
    )

    __table_args__ = (
        CheckConstraint("length(sha256) = 64", name="ck_scans_sha256_length"),
        CheckConstraint("file_size >= 0", name="ck_scans_file_size"),
        CheckConstraint(_in("scan_status", SCAN_STATUSES), name="ck_scans_status"),
        CheckConstraint(
            "classification IS NULL OR " + _in("classification", CLASSIFICATIONS), name="ck_scans_classification"
        ),
        CheckConstraint(
            "risk_level IS NULL OR " + _in("risk_level", RISK_LEVELS), name="ck_scans_risk_level"
        ),
        CheckConstraint(
            "extraction_status IS NULL OR " + _in("extraction_status", EXTRACTION_STATUSES),
            name="ck_scans_extraction_status",
        ),
        CheckConstraint("prediction IS NULL OR prediction IN ('BENIGN', 'MALWARE')", name="ck_scans_prediction"),
        CheckConstraint("risk_score IS NULL OR (risk_score >= 0 AND risk_score <= 100)", name="ck_scans_risk_score"),
        CheckConstraint(
            "(benign_probability IS NULL OR (benign_probability >= 0 AND benign_probability <= 1)) AND "
            "(malicious_probability IS NULL OR (malicious_probability >= 0 AND malicious_probability <= 1))",
            name="ck_scans_probabilities",
        ),
        Index("ix_scans_user_date", "user_id", "scan_date"),
        Index("ix_scans_sha256_status", "sha256", "scan_status"),
    )


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scan_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("scans.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    explanation_source: Mapped[str] = mapped_column(String(16), nullable=False, default="rule_based")
    findings: Mapped[str] = mapped_column(Text, nullable=False)          # JSON list
    recommendations: Mapped[str] = mapped_column(Text, nullable=False)   # JSON list
    limitations: Mapped[str] = mapped_column(Text, nullable=False)       # JSON list

    scan: Mapped[Scan] = relationship(back_populates="report")

    __table_args__ = (
        CheckConstraint("explanation_source IN ('rule_based', 'ai')", name="ck_reports_source"),
    )


class LibrarySample(Base):
    __tablename__ = "library_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    relative_path: Mapped[str] = mapped_column(String(500), nullable=False)
    ground_truth_label: Mapped[str] = mapped_column(String(32), nullable=False, default="unclassified")
    label_source: Mapped[str] = mapped_column(String(64), nullable=False, default="unclassified")
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    # Last scan details (synced whenever the library sample is scanned)
    last_scan_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("scans.id", ondelete="SET NULL"), nullable=True)
    last_scan_status: Mapped[str] = mapped_column(String(32), nullable=False, default="NOT_SCANNED")
    last_prediction: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    last_risk_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    last_risk_level: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    last_scan_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint("length(sha256) = 64", name="ck_library_sha256_length"),
        CheckConstraint("file_size >= 0", name="ck_library_file_size"),
    )

