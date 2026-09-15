"""Tests for graph routing logic and retry constraints."""

from langgraph.graph import END

from src.graph import route_after_auditor


def test_route_after_auditor_passed():
    """Verify routing ends when validation errors are absent."""
    state = {
        "transaction_data": {},
        "is_suspicious": True,
        "explanation_json": {"verdict": "SUSPICIOUS"},
        "validation_errors": None,
        "revision_count": 0,
        "model_name": "qwen2.5:7b"
    }
    assert route_after_auditor(state) == END


def test_route_after_auditor_retry():
    """Verify routing loops back to explainer when errors exist and revision limit not reached."""
    state = {
        "transaction_data": {},
        "is_suspicious": True,
        "explanation_json": {},
        "validation_errors": "Schema error",
        "revision_count": 1,
        "model_name": "qwen2.5:7b"
    }
    assert route_after_auditor(state) == "explainer"


def test_route_after_auditor_max_revisions():
    """Verify routing halts at END when revision limit of 3 is reached."""
    state = {
        "transaction_data": {},
        "is_suspicious": True,
        "explanation_json": {},
        "validation_errors": "Schema error",
        "revision_count": 3,
        "model_name": "qwen2.5:7b"
    }
    assert route_after_auditor(state) == END
