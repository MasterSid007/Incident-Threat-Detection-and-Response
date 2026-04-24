"""
MITRE ATT&CK Mapping for ITDR Detection Rules.

Maps each detection rule and attack type to MITRE ATT&CK technique IDs,
providing industry-standard threat taxonomy for alerts and dashboards.

Reference: https://attack.mitre.org/
"""
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────
# MITRE ATT&CK Technique Registry
# ─────────────────────────────────────────────────────────

MITRE_TECHNIQUES = {
    "T1110.003": {
        "id": "T1110.003",
        "name": "Brute Force: Password Spraying",
        "tactic": "Credential Access",
        "description": "Adversaries use a single or small list of commonly used passwords against many accounts to attempt to acquire valid credentials.",
        "url": "https://attack.mitre.org/techniques/T1110/003/",
        "severity": "Critical"
    },
    "T1078": {
        "id": "T1078",
        "name": "Valid Accounts",
        "tactic": "Defense Evasion, Persistence, Initial Access",
        "description": "Adversaries obtain and abuse credentials of existing accounts as a means of gaining initial access, persistence, or privilege escalation.",
        "url": "https://attack.mitre.org/techniques/T1078/",
        "severity": "High"
    },
    "T1078.004": {
        "id": "T1078.004",
        "name": "Valid Accounts: Cloud Accounts",
        "tactic": "Defense Evasion, Persistence, Initial Access",
        "description": "Adversaries obtain and abuse credentials of cloud accounts to gain initial access or elevate privileges in cloud environments.",
        "url": "https://attack.mitre.org/techniques/T1078/004/",
        "severity": "Critical"
    },
    "T1528": {
        "id": "T1528",
        "name": "Steal Application Access Token",
        "tactic": "Credential Access",
        "description": "Adversaries steal application access tokens to acquire credentials and bypass MFA or session controls.",
        "url": "https://attack.mitre.org/techniques/T1528/",
        "severity": "Critical"
    },
    "T1098": {
        "id": "T1098",
        "name": "Account Manipulation",
        "tactic": "Persistence, Privilege Escalation",
        "description": "Adversaries manipulate accounts to maintain and/or elevate access to victim systems.",
        "url": "https://attack.mitre.org/techniques/T1098/",
        "severity": "Critical"
    },
    "T1059": {
        "id": "T1059",
        "name": "Command and Scripting Interpreter",
        "tactic": "Execution",
        "description": "Adversaries abuse command and script interpreters to execute commands or scripts.",
        "url": "https://attack.mitre.org/techniques/T1059/",
        "severity": "Medium"
    },
    "T1530": {
        "id": "T1530",
        "name": "Data from Cloud Storage",
        "tactic": "Collection",
        "description": "Adversaries access data from improperly secured cloud storage, often as part of data exfiltration.",
        "url": "https://attack.mitre.org/techniques/T1530/",
        "severity": "High"
    },
    "T1090": {
        "id": "T1090",
        "name": "Proxy",
        "tactic": "Command and Control",
        "description": "Adversaries use proxies or anonymization services to direct network traffic through intermediary systems.",
        "url": "https://attack.mitre.org/techniques/T1090/",
        "severity": "Medium"
    },
}


# ─────────────────────────────────────────────────────────
# Rule → MITRE Mapping
# ─────────────────────────────────────────────────────────

RULE_TO_MITRE = {
    "Password Spray":          "T1110.003",
    "Impossible Travel":       "T1078",
    "Token Theft":             "T1528",
    "Privilege Escalation":    "T1098",
    "Suspicious User Agent":   "T1059",
    "Bulk Download":           "T1530",
    "Suspicious IP":           "T1090",
    "High-Risk ASN":           "T1090",
}

# Attack type (from ML/simulation labels) → MITRE mapping
ATTACK_TYPE_TO_MITRE = {
    "PasswordSpray":           "T1110.003",
    "ImpossibleTravel":        "T1078",
    "TokenTheft":              "T1528",
    "PrivilegeEscalation":     "T1098",
    "Account Takeover":        "T1078.004",
    "Attacker IP":             "T1090",
}


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

def get_mitre_for_rule(rule_name: str) -> Optional[Dict]:
    """Look up MITRE technique details for a given rule name."""
    technique_id = RULE_TO_MITRE.get(rule_name)
    if technique_id:
        return MITRE_TECHNIQUES.get(technique_id)
    return None


def get_mitre_for_attack_type(attack_type: str) -> Optional[Dict]:
    """Look up MITRE technique details for a given attack type label."""
    technique_id = ATTACK_TYPE_TO_MITRE.get(attack_type)
    if technique_id:
        return MITRE_TECHNIQUES.get(technique_id)
    return None


def enrich_alerts(alerts: List[Dict]) -> List[Dict]:
    """
    Enrich a list of alert dicts with MITRE ATT&CK fields.

    Adds to each alert:
      - mitre_id: e.g. "T1110.003"
      - mitre_name: e.g. "Brute Force: Password Spraying"
      - mitre_tactic: e.g. "Credential Access"
      - mitre_url: link to ATT&CK page
    """
    for alert in alerts:
        rule_name = alert.get("rule", "")
        mitre = get_mitre_for_rule(rule_name)
        if mitre:
            alert["mitre_id"] = mitre["id"]
            alert["mitre_name"] = mitre["name"]
            alert["mitre_tactic"] = mitre["tactic"]
            alert["mitre_url"] = mitre["url"]
        else:
            alert["mitre_id"] = ""
            alert["mitre_name"] = ""
            alert["mitre_tactic"] = ""
            alert["mitre_url"] = ""
    return alerts


def get_coverage_matrix() -> List[Dict]:
    """
    Return the full MITRE ATT&CK coverage matrix for the dashboard.
    Shows which techniques are covered, by which rules, and the tactic.
    """
    matrix = []
    seen = set()
    for rule_name, technique_id in RULE_TO_MITRE.items():
        technique = MITRE_TECHNIQUES.get(technique_id, {})
        key = (technique_id, rule_name)
        if key not in seen:
            seen.add(key)
            matrix.append({
                "Technique ID": technique_id,
                "Technique Name": technique.get("name", "Unknown"),
                "Tactic": technique.get("tactic", "Unknown"),
                "Detection Rule": rule_name,
                "Severity": technique.get("severity", "Medium"),
                "Reference": technique.get("url", ""),
            })
    return matrix
