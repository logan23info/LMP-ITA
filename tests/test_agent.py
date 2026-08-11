"""
test_agent.py — Unit tests for audit tools
Run: pytest tests/ -v
"""

import pytest
import sys
sys.path.insert(0, ".")

from src.tools import check_segregation_of_duties, score_risk, check_completeness


class TestSoDChecker:
    def test_detects_developer_dba_conflict(self):
        result = check_segregation_of_duties.invoke({"user_roles": "Developer, Production DBA"})
        assert "CONFLICT" in result
        assert "FAIL" in result

    def test_detects_payment_conflict(self):
        result = check_segregation_of_duties.invoke({"user_roles": "Payment Initiator, Payment Approver"})
        assert "CONFLICT" in result
        assert "Critical" in result

    def test_no_conflict_clean_roles(self):
        result = check_segregation_of_duties.invoke({"user_roles": "Read Only User, Report Viewer"})
        assert "PASS" in result
        assert "No SoD conflicts" in result

    def test_multiple_conflicts_detected(self):
        result = check_segregation_of_duties.invoke({
            "user_roles": "Developer, Production DBA, Payment Initiator, Payment Approver"
        })
        assert "FAIL" in result


class TestRiskScorer:
    def test_critical_high_is_critical(self):
        result = score_risk.invoke({"impact": "Critical", "likelihood": "High"})
        assert "RISK RATING: Critical" in result

    def test_low_low_is_low(self):
        result = score_risk.invoke({"impact": "Low", "likelihood": "Low"})
        assert "RISK RATING: Low" in result

    def test_compensating_controls_downgrade(self):
        result = score_risk.invoke({
            "impact": "High",
            "likelihood": "High",
            "compensating_controls": "Manual review by CISO monthly"
        })
        assert "INHERENT RISK" in result
        assert "RESIDUAL RISK" in result

    def test_response_time_included(self):
        result = score_risk.invoke({"impact": "Critical", "likelihood": "High"})
        assert "business days" in result.lower() or "immediate" in result.lower()


class TestCompletenessChecker:
    def test_zero_exceptions_effective(self):
        result = check_completeness.invoke({"items_tested": 50, "exceptions_found": 0})
        assert "EFFECTIVE" in result
        assert "0.0%" in result

    def test_high_exception_rate_ineffective(self):
        result = check_completeness.invoke({"items_tested": 42, "exceptions_found": 15})
        assert "INEFFECTIVE" in result or "PARTIALLY" in result

    def test_exception_rate_calculation(self):
        result = check_completeness.invoke({"items_tested": 100, "exceptions_found": 10})
        assert "10.0%" in result

    def test_zero_items_tested_error(self):
        result = check_completeness.invoke({"items_tested": 0, "exceptions_found": 0})
        assert "ERROR" in result

    def test_within_threshold_substantially_effective(self):
        result = check_completeness.invoke({"items_tested": 100, "exceptions_found": 3})
        assert "SUBSTANTIALLY EFFECTIVE" in result or "EFFECTIVE" in result


class TestTrainingData:
    def test_jsonl_format_valid(self):
        import json
        with open("data/synthetic/audit_qa_pairs.jsonl") as f:
            records = [json.loads(line) for line in f if line.strip()]

        assert len(records) > 0, "Training data file is empty"
        for record in records:
            assert "control_id" in record
            assert "domain" in record
            assert "question" in record
            assert "answer" in record
            assert len(record["answer"]) > 100, f"Answer too short for {record['control_id']}"

    def test_control_ids_format(self):
        import json
        with open("data/synthetic/audit_qa_pairs.jsonl") as f:
            records = [json.loads(line) for line in f if line.strip()]

        for record in records:
            assert record["control_id"].startswith("ITGC-"), \
                f"Invalid control ID format: {record['control_id']}"
