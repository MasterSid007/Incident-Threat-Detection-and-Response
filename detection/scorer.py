"""
Risk Scoring Module for ITDR.
Aggregates ML anomaly scores and rule-based alerts into a unified risk score.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskScorer:
    """
    Aggregates ML anomaly scores and rule-based detections into a final risk score.
    
    The scoring strategy:
    - Critical rules: 100 points (hard evidence)
    - High rules: 80 points
    - Medium rules: 60 points
    - ML anomaly: Scaled 0-70 based on deviation
    - Final score: Max(Rule Score, ML Score) with boosting for multiple signals
    """
    
    SEVERITY_SCORES = {
        "Critical": 100,
        "High": 80,
        "Medium": 60,
        "Low": 30
    }
    
    def __init__(
        self, 
        ml_weight: float = 0.4, 
        rule_weight: float = 0.6,
        multi_signal_boost: float = 1.2
    ):
        """
        Initialize the risk scorer.
        
        Args:
            ml_weight: Weight for ML scores in combined calculation
            rule_weight: Weight for rule scores in combined calculation
            multi_signal_boost: Multiplier when both ML and rules fire
        """
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        self.multi_signal_boost = multi_signal_boost

    def calculate_score(
        self, 
        df: pd.DataFrame, 
        rule_alerts: List[Dict],
        ae_results: Optional[pd.DataFrame] = None,
        user_deviation: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate unified risk scores combining all detection signals.
        
        Args:
            df: DataFrame with 'anomaly_score' from Isolation Forest
            rule_alerts: List of rule alert dicts
            ae_results: Optional DataFrame with autoencoder results
            user_deviation: Optional DataFrame with user deviation scores
            
        Returns:
            DataFrame with added risk score columns
        """
        df = df.copy()
        
        # === 1. Normalize ML Score (Isolation Forest) ===
        # ML is now a primary signal for real data (max 60 points)
        if 'anomaly_score' in df.columns:
            # Transform: negative = anomaly -> risk
            # Only score if clearly anomalous (score < -0.05)
            df['if_risk'] = 0.0
            anomaly_mask = df['anomaly_score'] < -0.05
            df.loc[anomaly_mask, 'if_risk'] = (0 - df.loc[anomaly_mask, 'anomaly_score']) * 100
            df['if_risk'] = df['if_risk'].clip(0, 60)  # Max 60 point ML signal
        else:
            df['if_risk'] = 0
            
        # === 2. Integrate Autoencoder Results ===
        if ae_results is not None and 'reconstruction_error' in ae_results.columns:
            # Only boost high reconstruction errors
            threshold = ae_results['reconstruction_error'].quantile(0.95)
            df['ae_risk'] = 0.0
            high_error = ae_results['reconstruction_error'] > threshold
            df.loc[high_error, 'ae_risk'] = 15  # Fixed boost for high errors
        else:
            df['ae_risk'] = 0
            
        # === 3. ML Boost Score (max 20) ===
        ml_cols = ['if_risk', 'ae_risk']
        available_ml = [col for col in ml_cols if col in df.columns]
        df['ml_risk'] = df[available_ml].max(axis=1) if available_ml else 0
            
        # === 4. Map Rule Alerts (PRIMARY SIGNAL) ===
        df['rule_risk'] = 0.0
        df['rule_details'] = ""
        df['recommendation'] = ""
        
        # Build lookup by timestamp + entity for faster matching
        for alert in rule_alerts:
            ts = alert['timestamp']
            severity = alert.get('severity', 'Medium')
            score = self.SEVERITY_SCORES.get(severity, 60)
            
            entity = alert.get('entity', '')
            entity_type = alert.get('entity_type', '')
            
            # Match by timestamp AND entity for precision
            if entity_type == 'ip' and 'ip' in df.columns:
                mask = (df['timestamp'] == ts) & (df['ip'] == entity)
            elif entity_type == 'user' and 'upn' in df.columns:
                mask = (df['timestamp'] == ts) & (df['upn'] == entity)
            else:
                mask = (df['timestamp'] == ts)
            
            if mask.sum() == 0:
                continue
            
            # Update with max rule score if multiple rules hit same event
            current_max = df.loc[mask, 'rule_risk'].max()
            if pd.isna(current_max) or score > current_max:
                df.loc[mask, 'rule_risk'] = score
                df.loc[mask, 'rule_details'] = f"{alert['rule']}: {alert['details']}"
                df.loc[mask, 'recommendation'] = alert.get('recommendation', '')
                
        # === 5. Integrate User Deviation Scores ===
        if user_deviation is not None and 'user_deviation_score' in user_deviation.columns:
            # High deviation from user baseline adds to risk
            df['user_deviation'] = (user_deviation['user_deviation_score'] * 10).clip(0, 30)
        else:
            df['user_deviation'] = 0
            
        # === 6. Calculate Final Risk Score ===
        # Strategy:
        # - If rule hit: Start with rule score
        # - Add ML evidence if significant
        # - Boost if multiple signals agree
        
        df['has_rule_hit'] = df['rule_risk'] > 0
        df['has_ml_flag'] = df['ml_risk'] > 10  # ML max is now 20, so 10 is significant
        
        # Base score is max of rule and ML
        df['base_score'] = df[['rule_risk', 'ml_risk']].max(axis=1)
        
        # Add user deviation
        df['base_score'] = df['base_score'] + df['user_deviation']
        
        # Boost when multiple signals agree
        multi_signal = df['has_rule_hit'] & df['has_ml_flag']
        df.loc[multi_signal, 'base_score'] = (
            df.loc[multi_signal, 'base_score'] * self.multi_signal_boost
        )
        
        # Final clipping
        df['final_risk_score'] = df['base_score'].clip(0, 100)
        
        # === 7. Risk Level Category ===
        df['risk_level'] = pd.cut(
            df['final_risk_score'],
            bins=[0, 30, 60, 80, 100],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        # Clean up intermediate columns
        df.drop(['has_rule_hit', 'has_ml_flag', 'base_score'], axis=1, inplace=True)
        
        logger.info(f"Scored {len(df)} events. Critical: {(df['risk_level'] == 'Critical').sum()}")
        
        return df
    
    def get_top_alerts(
        self, 
        df: pd.DataFrame, 
        min_score: float = 50, 
        limit: int = 20
    ) -> pd.DataFrame:
        """
        Extract top alerts above threshold.
        
        Args:
            df: Scored DataFrame
            min_score: Minimum risk score to include
            limit: Maximum alerts to return
            
        Returns:
            DataFrame of top alerts sorted by risk score
        """
        alerts = df[df['final_risk_score'] >= min_score].copy()
        alerts = alerts.sort_values('final_risk_score', ascending=False).head(limit)
        
        return alerts[[
            'timestamp', 'upn', 'eventType', 'final_risk_score', 
            'risk_level', 'rule_details', 'recommendation',
            'ml_risk', 'rule_risk', 'is_attack', 'attack_type'
        ]]


if __name__ == "__main__":
    from etl import LogLoader
    from features import FeatureExtractor
    from models import AnomalyDetector, Autoencoder, UserProfiler, SupervisedAttackClassifier
    from rules import RuleEngine
    from comparison_eval import run_comparison
    from alert_exporter import AlertExplainer, AlertExporter
    
    print("=" * 60)
    print("   ITDR Prototype - Full Integration Test")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1/8] Loading Log Data...")
    loader = LogLoader("../sample_logs.jsonl")
    df = loader.load_to_dataframe()
    print(f"      Loaded {len(df)} events")
    
    # 2. Baseline Rules
    print("\n[2/8] Running Rule Engine...")
    rule_engine = RuleEngine(df)
    rule_alerts = rule_engine.run_all()
    print(f"      Found {len(rule_alerts)} rule violations")
    
    # 3. Feature Extraction
    print("\n[3/8] Extracting Features...")
    extractor = FeatureExtractor()
    extractor.fit(df)
    X = extractor.transform(df)
    
    # 4. ML Models - Supervised classifier
    print("\n[4/8] Running ML Models...")
    y_labels = df['is_attack'].fillna(False).astype(bool)
    supervised = SupervisedAttackClassifier()
    accuracy = supervised.train(X, y_labels)
    supervised_results = supervised.predict(X)
    
    # Also run Isolation Forest (secondary signal)
    detector = AnomalyDetector(contamination=0.05)
    detector.train(X)
    if_results = detector.predict(X)
    
    # User Profiler
    profiler = UserProfiler()
    profiler.fit(df, X)
    user_dev = profiler.score(df, X)
    
    # 5. Final Results using Supervised Model
    print("\n[5/8] Calculating Final Results...")
    full_df = df.copy()
    full_df['attack_probability'] = supervised_results['attack_probability'].values
    full_df['predicted_attack'] = supervised_results['supervised_pred'].values
    full_df['final_risk_score'] = (full_df['attack_probability'] * 100).round(1)
    full_df['risk_level'] = full_df['final_risk_score'].apply(
        lambda x: 'Critical' if x >= 70 else ('High' if x >= 50 else ('Medium' if x >= 30 else 'Low'))
    )
    
    # 6. Per-Alert Explainability
    print("\n[6/8] Generating Alert Explanations...")
    rf_model = supervised.model.named_estimators_['rf']
    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    feature_means = X.mean()
    feature_stds = X.std()
    
    explainer = AlertExplainer(feature_importances, feature_means, feature_stds)
    flagged_mask = full_df['predicted_attack'].astype(bool)
    explanations = explainer.explain_batch(X, flagged_mask, top_n=5)
    
    # Add explanations to the dataframe
    full_df['top_signals'] = explanations['top_signals']
    full_df['rationale'] = explanations['rationale']
    
    # Export alerts to JSONL
    full_df['rule_risk'] = 0
    full_df['rule_details'] = ''
    full_df['ml_risk'] = (full_df['attack_probability'] * 100).round(1)
    num_exported = AlertExporter.export_alerts(full_df, explanations, min_score=50)
    
    # 7. Comparison Evaluation (Research Question)
    print("\n[7/8] Running Comparison Evaluation (Research Question)...")
    comparison_results = run_comparison(
        df, rule_alerts, supervised_results['supervised_pred']
    )
    
    # 8. Evaluation
    print("\n[8/8] Evaluation Results...")
    
    # Get top predictions
    top_risks = full_df.nlargest(20, 'attack_probability')
    
    print("\n" + "=" * 60)
    print("   TOP HIGH-RISK EVENTS")
    print("=" * 60)
    print(top_risks[['timestamp', 'upn', 'eventType', 'final_risk_score', 'risk_level', 'is_attack']].to_string(index=False))
    
    # Show sample explanations
    print("\n" + "=" * 60)
    print("   SAMPLE ALERT EXPLANATIONS")
    print("=" * 60)
    sample_alerts = full_df[full_df['predicted_attack'] == 1].nlargest(5, 'final_risk_score')
    for _, row in sample_alerts.iterrows():
        print(f"\n  [{row['risk_level']}] Score: {row['final_risk_score']} | {row['upn']}")
        print(f"  Signals: {row['top_signals']}")
        print(f"  Rationale: {row['rationale']}")
    
    # Metrics using supervised predictions
    y_true = full_df['is_attack'].fillna(False).astype(int)
    y_pred = full_df['predicted_attack'].astype(int)
    
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    acc = (tp + tn) / len(full_df) if len(full_df) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "=" * 60)
    print("   DETECTION METRICS (Supervised Model)")
    print("=" * 60)
    print(f"   Total Attacks in Dataset: {y_true.sum()}")
    print(f"   Attacks Detected (TP):    {tp}")
    print(f"   Attacks Missed (FN):      {fn}")
    print(f"   False Positives (FP):     {fp}")
    print(f"\n   >>> ACCURACY:   {acc:.1%} <<<")
    print(f"   Precision: {precision:.1%}")
    print(f"   Recall:    {recall:.1%}")
    print(f"   F1 Score:  {f1:.1%}")
    print(f"\n   Alerts Exported: {num_exported}")
    print("=" * 60)

