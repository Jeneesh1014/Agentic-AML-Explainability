"""Tests for Auditor node and BaFinComplianceSchema validation."""

import pytest
from pydantic import ValidationError

from src.nodes.auditor import BaFinComplianceSchema, audit_explanation


def test_valid_suspicious_schema():
    """Verify that a properly structured SUSPICIOUS payload validates successfully."""
    payload = {
        "verdict": "SUSPICIOUS",
        "confidence_score": 0.85,
        "rule_violated": "Smurfing",
        "evidence_list": [
            "Amount of 9950 EUR is just below the 10000 EUR threshold.",
            "Transaction timestamp is abnormal."
        ]
    }
    schema = BaFinComplianceSchema(**payload)
    assert schema.verdict == "SUSPICIOUS"
    assert schema.confidence_score == 0.85
    assert schema.rule_violated == "Smurfing"
    assert len(schema.evidence_list) == 2


def test_valid_clean_schema():
    """Verify that a properly structured CLEAN payload validates successfully."""
    payload = {
        "verdict": "CLEAN",
        "confidence_score": 1.0,
        "rule_violated": "None",
        "evidence_list": []
    }
    schema = BaFinComplianceSchema(**payload)
    assert schema.verdict == "CLEAN"
    assert schema.confidence_score == 1.0
    assert schema.rule_violated == "None"
    assert schema.evidence_list == []


def test_invalid_verdict():
    """Verify that an unexpected verdict value raises a validation error."""
    payload = {
        "verdict": "FLAGGED",
        "confidence_score": 0.9,
        "rule_violated": "Smurfing",
        "evidence_list": ["Sample evidence"]
    }
    with pytest.raises(ValidationError):
        BaFinComplianceSchema(**payload)


def test_confidence_score_out_of_bounds():
    """Verify that a confidence score greater than 1.0 raises a validation error."""
    payload = {
        "verdict": "SUSPICIOUS",
        "confidence_score": 1.5,
        "rule_violated": "Velocity Fraud",
        "evidence_list": ["High velocity detected"]
    }
    with pytest.raises(ValidationError):
        BaFinComplianceSchema(**payload)


def test_audit_explanation_node_pass():
    """Verify that the auditor node returns revision_count 0 on valid payload."""
    state = {
        "transaction_data": {},
        "is_suspicious": True,
        "explanation_json": {
            "verdict": "SUSPICIOUS",
            "confidence_score": 0.88,
            "rule_violated": "Smurfing",
            "evidence_list": ["Evidence item 1"]
        },
        "validation_errors": None,
        "revision_count": 0,
        "model_name": "qwen2.5:7b"
    }
    result = audit_explanation(state)
    assert result["validation_errors"] is None
    assert result["revision_count"] == 0


def test_audit_explanation_node_missing_payload():
    """Verify that the auditor node rejects missing explanation data."""
    state = {
        "transaction_data": {},
        "is_suspicious": True,
        "explanation_json": None,
        "validation_errors": None,
        "revision_count": 0,
        "model_name": "qwen2.5:7b"
    }
    result = audit_explanation(state)
    assert result["validation_errors"] is not None
    assert result["revision_count"] == 1


def test_audit_explanation_node_schema_failure():
    """Verify that the auditor node catches schema validation failures."""
    state = {
        "transaction_data": {},
        "is_suspicious": True,
        "explanation_json": {
            "verdict": "INVALID_VERDICT",
            "confidence_score": 0.5
        },
        "validation_errors": None,
        "revision_count": 0,
        "model_name": "qwen2.5:7b"
    }
    result = audit_explanation(state)
    assert result["validation_errors"] is not None
    assert result["revision_count"] == 1
