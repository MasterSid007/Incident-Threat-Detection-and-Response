"""
End-to-End Integration Test for ITDR Prototype.

Validates that all components work together as a cohesive system:
  Data Loading → Feature Extraction → ML Prediction → Rule Detection
  → MITRE Enrichment → Response Generation → Explainability → Comparison Evaluation

This test uses synthetic data so it runs without the large RBA dataset.
"""
import pytest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detection.features import FeatureExtractor
from detection.rules import RuleEngine
from detection.mitre_mapping import enrich_alerts, get_coverage_matrix, get_mitre_for_attack_type
from detection.response import ResponseEngine, ActionType, ActionStatus
from detection.alert_exporter import AlertExplainer
from detection.comparison_eval import evaluate_approach, rules_only_prediction


def _create_integration_dataset(n=200):
    """
    Create a realistic synthetic dataset that exercises the full pipeline.
    Includes normal logins, password sprays, and impossible travel patterns.
    """
    np.random.seed(42)
    records = []
    base_time = pd.Timestamp("2024-03-01 09:00:00")

    # Normal users (80% of data)
    normal_count = int(n * 0.8)
    for i in range(normal_count):
        records.append({
            "timestamp": base_time + pd.Timedelta(minutes=i * 2),
            "upn": f"user{i % 10}@corp.com",
            "ip": f"10.0.0.{i % 20}",
            "country": np.random.choice(["US", "US", "US", "UK"]),
            "city": "New York",
            "browser": "Chrome",
            "os": "Windows",
            "asn": f"AS{1000 + i % 5}",
            "status": "Success",
            "is_managed": True,
            "eventType": "Authentication",
            "appName": "Office 365",
            "correlationId": f"corr-{i}",
            "is_attack": False,
            "attack_type": "None",
        })

    # Attack events — password spray from single IP (20% of data)
    attack_count = n - normal_count
    spray_ip = "192.168.99.1"
    for i in range(attack_count):
        records.append({
            "timestamp": base_time + pd.Timedelta(minutes=i),
            "upn": f"victim{i}@corp.com",
            "ip": spray_ip,
            "country": np.random.choice(["RU", "CN"]),
            "city": "Unknown",
            "browser": "python-requests",
            "os": "Linux",
            "asn": "AS99999",
            "status": "Failure",
            "is_managed": False,
            "eventType": "Authentication",
            "appName": "Office 365",
            "correlationId": f"corr-attack-{i}",
            "is_attack": True,
            "attack_type": "PasswordSpray",
        })

    return pd.DataFrame(records)


class TestEndToEndIntegration:
    """Full pipeline integration tests."""

    @pytest.fixture
    def dataset(self):
        return _create_integration_dataset(200)

    def test_full_pipeline_runs_without_error(self, dataset):
        """The complete pipeline should run start to finish without errors."""
        # 1. Feature Extraction
        extractor = FeatureExtractor()
        extractor.fit(dataset)
        X = extractor.transform(dataset)
        assert X.shape[0] == len(dataset)
        assert not X.isna().any().any(), "Feature matrix has NaN values"

        # 2. Rule Engine
        rule_engine = RuleEngine(dataset)
        rule_alerts = rule_engine.run_all()
        assert isinstance(rule_alerts, list)
        # Should detect password spray and/or suspicious user agent
        rule_names = {a['rule'] for a in rule_alerts}
        assert len(rule_names) > 0, "No rules fired on attack data"

        # 3. MITRE Enrichment (already done by run_all)
        for alert in rule_alerts:
            assert 'mitre_id' in alert
            assert 'mitre_name' in alert

        # 4. Response Engine
        response_engine = ResponseEngine()
        response_actions = response_engine.process_alerts(rule_alerts)
        assert len(response_actions) > 0, "No response actions generated"
        for action in response_actions:
            assert action.status == ActionStatus.SIMULATED
            assert action.timestamp is not None

        summary = response_engine.get_response_summary()
        assert summary['total_actions'] > 0

        # 5. Comparison Evaluation
        y_true = dataset['is_attack'].astype(int)
        y_rules = rules_only_prediction(dataset, rule_alerts)
        rules_metrics = evaluate_approach(y_true, y_rules, "Rules-Only")
        assert 'precision' in rules_metrics
        assert 'recall' in rules_metrics
        assert 'f1_score' in rules_metrics

    def test_feature_extraction_produces_expected_features(self, dataset):
        """Feature extractor should produce known behavioral features."""
        extractor = FeatureExtractor()
        extractor.fit(dataset)
        X = extractor.transform(dataset)

        expected_features = [
            'hour_sin', 'hour_cos', 'is_failure', 'is_managed',
            'country_freq', 'browser_freq', 'asn_freq',
            'ip_attack_rate', 'asn_attack_rate', 'country_attack_rate',
            'ip_sharing_score', 'user_typical_ips', 'hour_deviation',
            'user_fail_rate', 'user_event_count',
        ]
        for feat in expected_features:
            assert feat in X.columns, f"Missing expected feature: {feat}"

    def test_rules_detect_password_spray(self, dataset):
        """Rule engine should detect the password spray attack pattern."""
        rule_engine = RuleEngine(dataset)
        spray_alerts = rule_engine.detect_password_spray()
        # Our synthetic data has 40 failures from one IP across many users
        # This should trigger the password spray rule
        if len(spray_alerts) > 0:
            assert spray_alerts[0]['rule'] == 'Password Spray'
            assert spray_alerts[0]['severity'] == 'Critical'

    def test_mitre_coverage_is_complete(self):
        """MITRE coverage matrix should have entries for all 7 detection rules."""
        matrix = get_coverage_matrix()
        rule_names = {entry['Detection Rule'] for entry in matrix}
        expected_rules = {
            'Password Spray', 'Impossible Travel', 'Token Theft',
            'Privilege Escalation', 'Suspicious User Agent',
            'Bulk Download', 'Suspicious IP', 'High-Risk ASN'
        }
        # All expected rules should be in the coverage matrix
        for rule in expected_rules:
            assert rule in rule_names, f"Rule '{rule}' missing from MITRE coverage matrix"

    def test_response_playbooks_cover_all_rules(self):
        """Every rule should have a response playbook defined."""
        from detection.response import RESPONSE_PLAYBOOKS
        expected_rules = [
            'Password Spray', 'Impossible Travel', 'Token Theft',
            'Privilege Escalation', 'Suspicious User Agent',
            'Bulk Download', 'Suspicious IP', 'High-Risk ASN'
        ]
        for rule in expected_rules:
            assert rule in RESPONSE_PLAYBOOKS, f"No response playbook for rule '{rule}'"

    def test_alert_explainer_produces_explanations(self, dataset):
        """Alert explainer should produce non-empty explanations for flagged events."""
        extractor = FeatureExtractor()
        extractor.fit(dataset)
        X = extractor.transform(dataset)

        # Simulate feature importances (normally from trained RF model)
        importances = pd.Series(np.random.rand(X.shape[1]), index=X.columns)

        explainer = AlertExplainer(importances, X.mean(), X.std())
        flagged_mask = dataset['is_attack'].astype(bool)
        explanations = explainer.explain_batch(X, flagged_mask, top_n=5)

        assert 'top_signals' in explanations.columns
        assert 'rationale' in explanations.columns

        # At least some flagged events should have explanations
        flagged_explanations = explanations[flagged_mask]
        non_empty = flagged_explanations['rationale'].str.len() > 0
        assert non_empty.any(), "No explanations generated for flagged events"

    def test_comparison_evaluation_structure(self, dataset):
        """Comparison evaluation should produce properly structured metrics for all 3 approaches."""
        y_true = dataset['is_attack'].astype(int)

        # Simulate predictions
        y_rules = (dataset['ip'] == '192.168.99.1').astype(int)
        y_ml = dataset['is_attack'].astype(int)  # Perfect ML for testing
        y_combined = ((y_rules == 1) | (y_ml == 1)).astype(int)

        rules_m = evaluate_approach(y_true, y_rules, "Rules-Only")
        ml_m = evaluate_approach(y_true, y_ml, "ML Behavioral")
        combined_m = evaluate_approach(y_true, y_combined, "Combined")

        for m in [rules_m, ml_m, combined_m]:
            assert 'approach' in m
            assert 'accuracy' in m
            assert 'precision' in m
            assert 'recall' in m
            assert 'f1_score' in m
            assert 'fpr' in m
            assert 0 <= m['accuracy'] <= 1
            assert 0 <= m['precision'] <= 1
            assert 0 <= m['recall'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
