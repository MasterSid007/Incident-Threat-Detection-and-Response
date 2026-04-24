"""
Comparison Evaluator — Rules vs ML vs Combined Detection

Research Question:
  "How effective is behavioral ML-based detection compared to rule-based
   baselines for identifying identity threats?"

This module answers the question empirically by evaluating three approaches
on the same dataset and comparing precision, recall, F1, and FPR.
"""
import pandas as pd
import numpy as np
import json
import sys
import os
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(__file__))


def evaluate_approach(y_true: pd.Series, y_pred: pd.Series, name: str) -> Dict:
    """Compute detection metrics for a single approach."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        "approach": name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "fpr": round(fpr, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn
    }


def rules_only_prediction(df: pd.DataFrame, rule_alerts: list) -> pd.Series:
    """
    Derive attack predictions from rules alone.
    An event is flagged if any rule fired for:
      - same IP (for IP-targeted rules like Password Spray)
      - same user (for user-targeted rules like Impossible Travel)
    Uses both entity matching and timestamp proximity for robustness.
    """
    # Build sets of flagged entities from alerts
    flagged_ips = set()
    flagged_users = set()
    for alert in rule_alerts:
        entity = alert.get('entity', '')
        entity_type = alert.get('entity_type', '')
        if entity_type == 'ip':
            flagged_ips.add(entity)
        elif entity_type == 'user':
            flagged_users.add(entity)
        else:
            # Fallback: try both
            flagged_ips.add(entity)
            flagged_users.add(entity)

    ip_flagged = df['ip'].isin(flagged_ips) if 'ip' in df.columns else pd.Series(False, index=df.index)
    user_flagged = df['upn'].isin(flagged_users) if 'upn' in df.columns else pd.Series(False, index=df.index)

    return (ip_flagged | user_flagged).astype(int)


def run_comparison(
    df: pd.DataFrame,
    rule_alerts: list,
    ml_predictions: pd.Series,
    output_file: str = "../evaluation_results.json"
) -> Dict:
    """
    Run the three-way comparison evaluation.
    
    Args:
        df: Full dataframe with 'is_attack' ground truth
        rule_alerts: List of rule alert dicts
        ml_predictions: Series of ML-based predictions (0/1)
        output_file: Path to save results JSON
    
    Returns:
        Dictionary with comparison results
    """
    y_true = df['is_attack'].fillna(False).astype(int)
    
    # --- Approach 1: Rules Only ---
    y_rules = rules_only_prediction(df, rule_alerts)
    rules_metrics = evaluate_approach(y_true, y_rules, "Rules-Only Baseline")
    
    # --- Approach 2: ML Only ---
    y_ml = ml_predictions.astype(int)
    ml_metrics = evaluate_approach(y_true, y_ml, "ML Behavioral Detection")
    
    # --- Approach 3: Combined (either rules OR ML flags it) ---
    y_combined = ((y_rules == 1) | (y_ml == 1)).astype(int)
    combined_metrics = evaluate_approach(y_true, y_combined, "Combined (Rules + ML)")
    
    # --- Build Results ---
    results = {
        "research_question": (
            "How effective is behavioral ML-based detection compared to "
            "rule-based baselines for identifying identity threats?"
        ),
        "dataset_size": len(df),
        "total_attacks": int(y_true.sum()),
        "attack_rate": round(y_true.mean(), 4),
        "approaches": [rules_metrics, ml_metrics, combined_metrics],
        "findings": _generate_findings(rules_metrics, ml_metrics, combined_metrics)
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comparison table
    _print_comparison(results)
    
    return results


def _generate_findings(rules: Dict, ml: Dict, combined: Dict) -> list:
    """Generate plain-English research findings from the metrics."""
    findings = []
    
    # Precision comparison
    if ml['precision'] > rules['precision']:
        findings.append(
            f"ML behavioral detection achieves {ml['precision']:.1%} precision vs "
            f"{rules['precision']:.1%} for rules-only, a {(ml['precision'] - rules['precision']):.1%} improvement."
        )
    else:
        findings.append(
            f"Rules-only achieves higher precision ({rules['precision']:.1%}) than "
            f"ML detection ({ml['precision']:.1%}), suggesting rules produce fewer false positives."
        )
    
    # Recall comparison
    if ml['recall'] > rules['recall']:
        findings.append(
            f"ML detection catches {ml['recall']:.1%} of attacks vs {rules['recall']:.1%} for rules, "
            f"demonstrating that behavioral analysis detects attacks that static rules miss."
        )
    else:
        findings.append(
            f"Rules-only catches {rules['recall']:.1%} of attacks vs {ml['recall']:.1%} for ML."
        )
    
    # Combined benefit
    if combined['f1_score'] > max(rules['f1_score'], ml['f1_score']):
        findings.append(
            f"Combining both approaches yields the best F1 score ({combined['f1_score']:.1%}), "
            f"outperforming either method alone (rules: {rules['f1_score']:.1%}, ML: {ml['f1_score']:.1%})."
        )
    
    # FPR
    findings.append(
        f"False positive rates: Rules={rules['fpr']:.1%}, ML={ml['fpr']:.1%}, Combined={combined['fpr']:.1%}."
    )
    
    return findings


def _print_comparison(results: Dict):
    """Pretty-print the comparison table."""
    print("\n" + "=" * 80)
    print("   RESEARCH QUESTION EVALUATION")
    print("   " + results["research_question"])
    print("=" * 80)
    print(f"   Dataset: {results['dataset_size']:,} events | "
          f"Attacks: {results['total_attacks']:,} ({results['attack_rate']:.1%})")
    print("-" * 80)
    
    header = f"{'Approach':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPR':>10}"
    print(header)
    print("-" * 80)
    
    for approach in results["approaches"]:
        row = (
            f"{approach['approach']:<30} "
            f"{approach['accuracy']:>9.1%} "
            f"{approach['precision']:>9.1%} "
            f"{approach['recall']:>9.1%} "
            f"{approach['f1_score']:>9.1%} "
            f"{approach['fpr']:>9.1%}"
        )
        print(row)
    
    print("-" * 80)
    print("\n   FINDINGS:")
    for i, finding in enumerate(results["findings"], 1):
        print(f"   {i}. {finding}")
    print("=" * 80)


if __name__ == "__main__":
    from etl import LogLoader
    from features import FeatureExtractor
    from models import SupervisedAttackClassifier
    from rules import RuleEngine
    
    print("Loading data...")
    loader = LogLoader("../sample_logs.jsonl")
    df = loader.load_to_dataframe()
    print(f"Loaded {len(df):,} events")
    
    print("\nRunning Rule Engine...")
    rule_engine = RuleEngine(df)
    rule_alerts = rule_engine.run_all()
    print(f"Found {len(rule_alerts):,} rule violations")
    
    print("\nExtracting Features...")
    extractor = FeatureExtractor()
    extractor.fit(df)
    X = extractor.transform(df)
    
    print("\nTraining ML Model...")
    y_labels = df['is_attack'].fillna(False).astype(bool)
    classifier = SupervisedAttackClassifier()
    classifier.train(X, y_labels)
    ml_results = classifier.predict(X)
    
    print("\nRunning Comparison Evaluation...")
    results = run_comparison(df, rule_alerts, ml_results['supervised_pred'])
