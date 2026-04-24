"""
Alert Explainer & Exporter for ITDR Prototype.

Addresses proposal requirements FR4, FR5, NFR5:
- Per-alert top 3-5 contributing signals
- Human-readable rationale string
- Structured JSON alert export
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Optional


# Human-friendly names for ML features
# Must match actual features from FeatureExtractor.transform()
FEATURE_LABELS = {
    'hour_sin': 'unusual time of day',
    'hour_cos': 'unusual time of day',
    'day_of_week': 'unusual day of week',
    'is_weekend': 'weekend access',
    'is_business_hours': 'outside business hours',
    'is_failure': 'failed authentication',
    'is_new_device': 'new/unknown device',
    'is_managed': 'unmanaged device',
    'country_freq': 'rare country',
    'browser_freq': 'rare browser',
    'asn_freq': 'rare network (ASN)',
    'os_freq': 'rare operating system',
    'country_attack_rate': 'high-risk country',
    'asn_attack_rate': 'high-risk network (ASN)',
    'ip_attack_rate': 'high-risk IP address',
    'ip_sharing_score': 'IP shared across many users',
    'user_typical_ips': 'unusual IP diversity for user',
    'user_typical_countries': 'unusual country diversity for user',
    'country_anomaly': 'multi-country access pattern',
    'hour_deviation': 'login at unusual hour for user',
    'user_fail_rate': 'high failure rate for user',
    'user_event_count': 'high activity volume',
    'asn_attack_x_failure': 'risky network + failed login',
    'country_attack_x_hour_dev': 'risky country + unusual hour',
}


class AlertExplainer:
    """
    Generates per-alert explanations using feature contributions.
    
    For each flagged event, identifies the top contributing features
    and generates a one-sentence human-readable rationale.
    """
    
    def __init__(self, feature_importances: pd.Series, feature_means: pd.Series, feature_stds: pd.Series):
        """
        Args:
            feature_importances: Feature importance from the RF model
            feature_means: Mean values for each feature (baseline)
            feature_stds: Std dev for each feature (for z-score)
        """
        self.importances = feature_importances
        self.means = feature_means
        self.stds = feature_stds
    
    def explain_event(self, features: pd.Series, top_n: int = 5) -> Dict:
        """
        Generate explanation for a single event.
        
        Args:
            features: Feature values for this event
            top_n: Number of top signals to return
            
        Returns:
            Dict with 'signals' list and 'rationale' string
        """
        # Calculate feature contributions: |z-score| × importance
        contributions = {}
        for feat in features.index:
            if feat in self.importances.index and feat in self.stds.index:
                std = self.stds[feat]
                if std > 0:
                    z_score = abs((features[feat] - self.means[feat]) / std)
                else:
                    z_score = 0
                contribution = z_score * self.importances[feat]
                contributions[feat] = {
                    'feature': feat,
                    'value': float(features[feat]),
                    'z_score': round(float(z_score), 2),
                    'importance': round(float(self.importances[feat]), 4),
                    'contribution': round(float(contribution), 4),
                    'label': FEATURE_LABELS.get(feat, feat)
                }
        
        # Sort by contribution, take top N
        sorted_signals = sorted(
            contributions.values(), 
            key=lambda x: x['contribution'], 
            reverse=True
        )[:top_n]
        
        # Generate rationale string
        signal_phrases = []
        for s in sorted_signals[:3]:  # Use top 3 for the sentence
            label = s['label']
            if s['z_score'] > 2:
                signal_phrases.append(f"highly {label}")
            elif s['z_score'] > 1:
                signal_phrases.append(label)
        
        if signal_phrases:
            rationale = "Flagged due to: " + ", ".join(signal_phrases) + "."
        else:
            rationale = "Flagged by ML model based on combined feature analysis."
        
        return {
            'signals': sorted_signals,
            'rationale': rationale
        }
    
    def explain_batch(self, X: pd.DataFrame, flagged_mask: pd.Series, top_n: int = 5) -> pd.DataFrame:
        """
        Generate explanations for all flagged events.
        
        Args:
            X: Full feature matrix
            flagged_mask: Boolean mask of flagged events
            top_n: Signals per event
            
        Returns:
            DataFrame with 'top_signals' and 'rationale' columns
        """
        explanations = pd.DataFrame(index=X.index, columns=['top_signals', 'rationale'])
        explanations['top_signals'] = ''
        explanations['rationale'] = ''
        
        flagged_indices = X.index[flagged_mask]
        
        for idx in flagged_indices:
            exp = self.explain_event(X.loc[idx], top_n)
            # Compact signal summary for display
            signal_summary = "; ".join(
                [f"{s['label']} (z={s['z_score']})" for s in exp['signals'][:3]]
            )
            explanations.loc[idx, 'top_signals'] = signal_summary
            explanations.loc[idx, 'rationale'] = exp['rationale']
        
        return explanations


class AlertExporter:
    """
    Exports alerts as structured JSONL for evaluation and SOC integration.
    
    Each alert includes:
    - Identity metadata
    - Risk score and severity
    - Top 3-5 contributing signals
    - Human-readable rationale
    - Recommendation
    """
    
    @staticmethod
    def export_alerts(
        df: pd.DataFrame,
        explanations: pd.DataFrame,
        min_score: float = 50,
        output_file: str = "../alerts_export.jsonl"
    ) -> int:
        """
        Export high-risk alerts to JSONL file.
        
        Args:
            df: Scored DataFrame with risk scores
            explanations: DataFrame with 'top_signals' and 'rationale'
            min_score: Minimum risk score to export
            output_file: Output JSONL path
            
        Returns:
            Number of alerts exported
        """
        alerts = df[df['final_risk_score'] >= min_score].copy()
        alerts = alerts.sort_values('final_risk_score', ascending=False)
        
        count = 0
        with open(output_file, 'w') as f:
            for idx, row in alerts.iterrows():
                alert = {
                    "alert_id": f"ITDR-{count+1:06d}",
                    "timestamp": str(row.get('timestamp', '')),
                    "identity": {
                        "upn": str(row.get('upn', '')),
                        "ip": str(row.get('ip', '')),
                        "country": str(row.get('country', '')),
                        "device": str(row.get('device_name', ''))
                    },
                    "event_type": str(row.get('eventType', '')),
                    "risk_score": float(row.get('final_risk_score', 0)),
                    "severity": str(row.get('risk_level', 'Unknown')),
                    "is_attack": bool(row.get('is_attack', False)),
                    "detection": {
                        "rule_risk": float(row.get('rule_risk', 0)),
                        "ml_risk": float(row.get('ml_risk', 0)),
                        "rule_details": str(row.get('rule_details', '')),
                    },
                    "explanation": {
                        "top_signals": str(explanations.loc[idx, 'top_signals']) if idx in explanations.index else "",
                        "rationale": str(explanations.loc[idx, 'rationale']) if idx in explanations.index else "",
                    },
                    "recommendation": str(row.get('recommendation', 'Review and investigate'))
                }
                f.write(json.dumps(alert) + '\n')
                count += 1
        
        print(f"Exported {count:,} alerts to {output_file}")
        return count
