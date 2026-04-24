"""
Automated Response Actions for ITDR Prototype.

Recommends and simulates remediation actions based on alert severity and type.
Completes the "Response" aspect of Identity Threat Detection & Response.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional
from enum import Enum


class ActionType(Enum):
    BLOCK_IP = "Block IP"
    REVOKE_SESSIONS = "Revoke Sessions"
    FORCE_MFA = "Force MFA Re-challenge"
    QUARANTINE_ACCOUNT = "Quarantine Account"
    NOTIFY_SOC = "Notify SOC Team"
    LOG_REVIEW = "Log for Review"
    DISABLE_ACCOUNT = "Disable Account"
    RESET_PASSWORD = "Force Password Reset"


class ActionStatus(Enum):
    RECOMMENDED = "Recommended"
    SIMULATED = "Simulated"
    PENDING = "Pending"


@dataclass
class ResponseAction:
    """Represents a single automated response action."""
    action_type: ActionType
    target: str                          # IP, user UPN, or session ID
    target_type: str                     # "ip", "user", "session"
    severity: str                        # Critical, High, Medium, Low
    description: str
    status: ActionStatus = ActionStatus.RECOMMENDED
    triggered_by: str = ""               # Rule or ML model that triggered
    mitre_id: str = ""
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "action": self.action_type.value,
            "target": self.target,
            "target_type": self.target_type,
            "severity": self.severity,
            "description": self.description,
            "status": self.status.value,
            "triggered_by": self.triggered_by,
            "mitre_id": self.mitre_id,
            "timestamp": self.timestamp or "",
        }


# ── Action Templates by Rule + Severity ──────────────────

RESPONSE_PLAYBOOKS = {
    "Password Spray": {
        "actions": [ActionType.BLOCK_IP, ActionType.RESET_PASSWORD, ActionType.NOTIFY_SOC],
        "description_template": "Block attacker IP {target} and force password reset for targeted accounts",
    },
    "Impossible Travel": {
        "actions": [ActionType.REVOKE_SESSIONS, ActionType.FORCE_MFA, ActionType.NOTIFY_SOC],
        "description_template": "Revoke active sessions for {target} and require MFA re-authentication",
    },
    "Token Theft": {
        "actions": [ActionType.REVOKE_SESSIONS, ActionType.DISABLE_ACCOUNT, ActionType.NOTIFY_SOC],
        "description_template": "Immediately revoke all tokens for {target} and disable account pending review",
    },
    "Privilege Escalation": {
        "actions": [ActionType.QUARANTINE_ACCOUNT, ActionType.REVOKE_SESSIONS, ActionType.NOTIFY_SOC],
        "description_template": "Quarantine account {target} and revoke all active sessions",
    },
    "Suspicious User Agent": {
        "actions": [ActionType.LOG_REVIEW, ActionType.NOTIFY_SOC],
        "description_template": "Flag automation from {target} for SOC review",
    },
    "Bulk Download": {
        "actions": [ActionType.QUARANTINE_ACCOUNT, ActionType.NOTIFY_SOC],
        "description_template": "Quarantine account {target} and investigate potential data exfiltration",
    },
    "Suspicious IP": {
        "actions": [ActionType.BLOCK_IP, ActionType.NOTIFY_SOC],
        "description_template": "Block suspicious IP {target} and investigate related activity",
    },
    "High-Risk ASN": {
        "actions": [ActionType.BLOCK_IP, ActionType.LOG_REVIEW],
        "description_template": "Block traffic from high-risk ASN {target}",
    },
}

# Severity-based escalation
SEVERITY_ESCALATION = {
    "Critical": [ActionType.BLOCK_IP, ActionType.REVOKE_SESSIONS, ActionType.QUARANTINE_ACCOUNT, ActionType.NOTIFY_SOC],
    "High":     [ActionType.FORCE_MFA, ActionType.REVOKE_SESSIONS, ActionType.NOTIFY_SOC],
    "Medium":   [ActionType.NOTIFY_SOC, ActionType.LOG_REVIEW],
    "Low":      [ActionType.LOG_REVIEW],
}


class ResponseEngine:
    """
    Recommends and simulates automated response actions based on alerts.
    """

    def __init__(self):
        self.action_log: List[ResponseAction] = []

    def recommend_actions(self, alert: Dict) -> List[ResponseAction]:
        """
        Generate recommended response actions for a given alert.
        
        Args:
            alert: Alert dict with keys: rule, severity, entity, entity_type, mitre_id, etc.
            
        Returns:
            List of ResponseAction objects
        """
        rule_name = alert.get("rule", "")
        severity = alert.get("severity", "Medium")
        entity = str(alert.get("entity", "Unknown"))
        entity_type = alert.get("entity_type", "user")
        mitre_id = alert.get("mitre_id", "")

        actions = []
        playbook = RESPONSE_PLAYBOOKS.get(rule_name)

        if playbook:
            for action_type in playbook["actions"]:
                target_type = "ip" if action_type in (ActionType.BLOCK_IP,) else entity_type
                action = ResponseAction(
                    action_type=action_type,
                    target=entity,
                    target_type=target_type,
                    severity=severity,
                    description=playbook["description_template"].format(target=entity),
                    triggered_by=rule_name,
                    mitre_id=mitre_id,
                )
                actions.append(action)
        else:
            # Fallback: use severity-based escalation
            for action_type in SEVERITY_ESCALATION.get(severity, [ActionType.LOG_REVIEW]):
                action = ResponseAction(
                    action_type=action_type,
                    target=entity,
                    target_type=entity_type,
                    severity=severity,
                    description=f"{action_type.value} for {entity} (severity: {severity})",
                    triggered_by="Severity Escalation",
                    mitre_id=mitre_id,
                )
                actions.append(action)

        return actions

    def simulate_action(self, action: ResponseAction) -> ResponseAction:
        """
        Mark an action as simulated (for demo purposes).
        In a production system, this would execute the actual remediation.
        """
        action.status = ActionStatus.SIMULATED
        action.timestamp = datetime.now(timezone.utc).isoformat()
        self.action_log.append(action)
        return action

    def process_alerts(self, alerts: List[Dict]) -> List[ResponseAction]:
        """
        Process a batch of alerts, recommend and simulate actions.
        Deduplicates actions to avoid duplicate blocks/revokes.
        """
        all_actions = []
        seen = set()  # (action_type, target) dedup

        for alert in alerts:
            actions = self.recommend_actions(alert)
            for action in actions:
                key = (action.action_type, action.target)
                if key not in seen:
                    seen.add(key)
                    self.simulate_action(action)
                    all_actions.append(action)

        return all_actions

    def get_response_summary(self) -> Dict:
        """Get aggregate statistics for the dashboard."""
        if not self.action_log:
            return {
                "total_actions": 0,
                "by_type": {},
                "by_severity": {},
                "by_status": {},
            }

        by_type = {}
        by_severity = {}
        by_status = {}

        for action in self.action_log:
            t = action.action_type.value
            by_type[t] = by_type.get(t, 0) + 1
            s = action.severity
            by_severity[s] = by_severity.get(s, 0) + 1
            st = action.status.value
            by_status[st] = by_status.get(st, 0) + 1

        return {
            "total_actions": len(self.action_log),
            "by_type": by_type,
            "by_severity": by_severity,
            "by_status": by_status,
        }
