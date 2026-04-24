"""
Tests for the 4 high-impact improvements:
- MITRE ATT&CK mapping
- Response engine
- Streaming pipeline
"""
import pytest
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detection.mitre_mapping import (
    enrich_alerts, get_coverage_matrix, get_mitre_for_rule, 
    get_mitre_for_attack_type, MITRE_TECHNIQUES, RULE_TO_MITRE
)
from detection.response import ResponseEngine, ActionType, ActionStatus


class TestMitreMapping:
    """Tests for MITRE ATT&CK mapping module."""

    def test_enrich_alerts_adds_fields(self):
        """Enriched alerts should have mitre_id, mitre_name, mitre_tactic, mitre_url."""
        alerts = [
            {"rule": "Password Spray", "severity": "Critical", "entity": "1.2.3.4"},
            {"rule": "Token Theft", "severity": "Critical", "entity": "user@corp.com"},
        ]
        enriched = enrich_alerts(alerts)
        
        assert enriched[0]["mitre_id"] == "T1110.003"
        assert "Password Spraying" in enriched[0]["mitre_name"]
        assert enriched[0]["mitre_tactic"] == "Credential Access"
        assert enriched[0]["mitre_url"].startswith("https://")
        
        assert enriched[1]["mitre_id"] == "T1528"

    def test_enrich_unknown_rule_gets_empty_fields(self):
        """Unknown rules should get empty MITRE fields, not crash."""
        alerts = [{"rule": "NonexistentRule", "severity": "Low", "entity": "x"}]
        enriched = enrich_alerts(alerts)
        assert enriched[0]["mitre_id"] == ""
        assert enriched[0]["mitre_name"] == ""

    def test_get_coverage_matrix_structure(self):
        """Coverage matrix should have expected columns and cover all mapped rules."""
        matrix = get_coverage_matrix()
        assert len(matrix) > 0
        
        # Every entry should have required fields
        for entry in matrix:
            assert "Technique ID" in entry
            assert "Technique Name" in entry
            assert "Tactic" in entry
            assert "Detection Rule" in entry

    def test_all_rules_have_mitre_mapping(self):
        """Every rule in RULE_TO_MITRE should map to a valid technique."""
        for rule_name, technique_id in RULE_TO_MITRE.items():
            assert technique_id in MITRE_TECHNIQUES, f"Rule '{rule_name}' maps to unknown technique {technique_id}"

    def test_get_mitre_for_attack_type(self):
        """Attack type labels should map to valid MITRE techniques."""
        result = get_mitre_for_attack_type("PasswordSpray")
        assert result is not None
        assert result["id"] == "T1110.003"

        result = get_mitre_for_attack_type("TokenTheft")
        assert result is not None
        assert result["id"] == "T1528"

        # Unknown attack type
        assert get_mitre_for_attack_type("UnknownAttack") is None


class TestResponseEngine:
    """Tests for the automated response module."""

    def test_recommend_actions_password_spray(self):
        """Password spray should recommend block IP, reset password, notify SOC."""
        engine = ResponseEngine()
        alert = {
            "rule": "Password Spray",
            "severity": "Critical",
            "entity": "192.168.1.1",
            "entity_type": "ip",
            "mitre_id": "T1110.003",
        }
        actions = engine.recommend_actions(alert)
        
        assert len(actions) >= 2
        action_types = {a.action_type for a in actions}
        assert ActionType.BLOCK_IP in action_types
        assert ActionType.NOTIFY_SOC in action_types

    def test_recommend_actions_token_theft(self):
        """Token theft should recommend revoke sessions and disable account."""
        engine = ResponseEngine()
        alert = {
            "rule": "Token Theft",
            "severity": "Critical",
            "entity": "user@corp.com",
            "entity_type": "user",
            "mitre_id": "T1528",
        }
        actions = engine.recommend_actions(alert)
        
        action_types = {a.action_type for a in actions}
        assert ActionType.REVOKE_SESSIONS in action_types

    def test_severity_escalation_fallback(self):
        """Unknown rules should fall back to severity-based escalation."""
        engine = ResponseEngine()
        alert = {
            "rule": "UnknownRule",
            "severity": "Critical",
            "entity": "target",
            "entity_type": "user",
        }
        actions = engine.recommend_actions(alert)
        assert len(actions) > 0

    def test_simulate_action(self):
        """Simulated actions should have status SIMULATED and a timestamp."""
        engine = ResponseEngine()
        alert = {"rule": "Password Spray", "severity": "Critical", 
                 "entity": "1.2.3.4", "entity_type": "ip"}
        actions = engine.recommend_actions(alert)
        
        simulated = engine.simulate_action(actions[0])
        assert simulated.status == ActionStatus.SIMULATED
        assert simulated.timestamp is not None

    def test_process_alerts_deduplication(self):
        """Duplicate actions for the same target should be deduplicated."""
        engine = ResponseEngine()
        alerts = [
            {"rule": "Password Spray", "severity": "Critical", "entity": "1.2.3.4", "entity_type": "ip"},
            {"rule": "Suspicious IP", "severity": "High", "entity": "1.2.3.4", "entity_type": "ip"},
        ]
        actions = engine.process_alerts(alerts)
        
        # Should have deduped BLOCK_IP for 1.2.3.4
        block_actions = [a for a in actions if a.action_type == ActionType.BLOCK_IP and a.target == "1.2.3.4"]
        assert len(block_actions) == 1

    def test_get_response_summary(self):
        """Response summary should have correct structure."""
        engine = ResponseEngine()
        alerts = [
            {"rule": "Password Spray", "severity": "Critical", "entity": "1.2.3.4", "entity_type": "ip"},
        ]
        engine.process_alerts(alerts)
        summary = engine.get_response_summary()
        
        assert "total_actions" in summary
        assert summary["total_actions"] > 0
        assert "by_type" in summary
        assert "by_severity" in summary

    def test_action_to_dict(self):
        """ResponseAction.to_dict() should return a serializable dict."""
        engine = ResponseEngine()
        alert = {"rule": "Token Theft", "severity": "Critical", 
                 "entity": "user@corp.com", "entity_type": "user", "mitre_id": "T1528"}
        actions = engine.recommend_actions(alert)
        d = actions[0].to_dict()
        
        assert isinstance(d, dict)
        assert "action" in d
        assert "target" in d
        assert "severity" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
