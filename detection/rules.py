"""
Rule-based detection engine for ITDR.
Implements deterministic rules for known attack patterns.
"""
import pandas as pd
from datetime import timedelta
from typing import List, Dict, Optional
import logging
import yaml
import os
from .mitre_mapping import enrich_alerts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RuleEngine:
    """
    Rule-based detection engine that identifies known attack patterns.
    
    Supported Rules:
    - Password Spray: Multiple failed logins from same IP across users
    - Impossible Travel: Geographic impossibilities
    - Token Theft: Same session used from different IP/device
    - Privilege Escalation: Role changes followed by suspicious activity
    - Suspicious User Agent: Known attack tools
    """
    
    # High-risk countries (for impossible travel severity)
    HIGH_RISK_COUNTRIES = {"RU", "CN", "KP", "IR", "BY", "VE", "CU", "SY"}
    
    # Suspicious user agents
    SUSPICIOUS_AGENTS = ["curl", "wget", "python-requests", "httpie", "Headless", 
                         "Go-http-client", "Java/", "Scrapy", "Bot"]
    
    def __init__(self, df: pd.DataFrame, config_path: Optional[str] = None):
        """
        Initialize the rule engine with a DataFrame of log events.
        
        Args:
            df: DataFrame containing authentication events
            config_path: Optional path to config.yaml. If None, searches
                         common locations automatically.
        """
        self.df = df.sort_values("timestamp").copy()
        self.alerts: List[Dict] = []
        self.config = self._load_config(config_path)
        
    def detect_password_spray(
        self, 
        time_window_min: int = 5, 
        fail_threshold: int = 5,
        min_unique_users: int = 3
    ) -> List[Dict]:
        """
        Detects password spray attacks: Single IP targeting multiple accounts.
        
        Args:
            time_window_min: Time window in minutes to aggregate failures
            fail_threshold: Minimum failures to trigger alert
            min_unique_users: Minimum unique users targeted
            
        Returns:
            List of alert dictionaries
        """
        logger.info("Running Password Spray Detection...")
        alerts = []
        
        failed_logins = self.df[self.df['status'] == 'Failure'].copy()
        if failed_logins.empty:
            return alerts
            
        failed_logins.set_index('timestamp', inplace=True)
        
        for ip, group in failed_logins.groupby('ip'):
            resampled = group.resample(f'{time_window_min}min').agg({
                'upn': 'nunique',
                'status': 'count'
            })
            
            detected = resampled[
                (resampled['upn'] >= min_unique_users) & 
                (resampled['status'] >= fail_threshold)
            ]
            
            for ts, row in detected.iterrows():
                alerts.append({
                    "timestamp": ts,
                    "rule": "Password Spray",
                    "severity": "Critical",
                    "entity": ip,
                    "entity_type": "ip",
                    "details": f"IP {ip} targeted {row['upn']} accounts with {row['status']} failures in {time_window_min}m",
                    "recommendation": "Block IP and force password reset for targeted accounts"
                })
                
        return alerts

    def detect_impossible_travel(self, max_travel_hours: float = 2.0) -> List[Dict]:
        """
        Detects logins from geographically impossible locations.
        
        Args:
            max_travel_hours: Maximum hours between logins to flag as suspicious
            
        Returns:
            List of alert dictionaries
        """
        logger.info("Running Impossible Travel Detection...")
        alerts = []
        
        for user, group in self.df.groupby('upn'):
            group = group.sort_values('timestamp').copy()
            
            group['prev_timestamp'] = group['timestamp'].shift(1)
            group['prev_country'] = group['country'].shift(1)
            group['prev_ip'] = group['ip'].shift(1)
            
            suspicious = group[
                (group['prev_country'].notna()) & 
                (group['country'] != group['prev_country'])
            ].copy()
            
            if suspicious.empty:
                continue
                
            suspicious['time_diff'] = (
                suspicious['timestamp'] - suspicious['prev_timestamp']
            ).dt.total_seconds() / 3600.0
            
            for idx, row in suspicious.iterrows():
                # Check if involves high-risk countries
                is_high_risk = (
                    row['country'] in self.HIGH_RISK_COUNTRIES or 
                    row['prev_country'] in self.HIGH_RISK_COUNTRIES
                )
                
                # Only flag if:
                # 1. Truly impossible (< 30 min between countries), OR
                # 2. High-risk country AND < 2 hours
                should_alert = False
                severity = "Medium"
                
                if row['time_diff'] < 0.5:  # < 30 minutes = truly impossible
                    should_alert = True
                    severity = "Critical"
                elif is_high_risk and row['time_diff'] < max_travel_hours:
                    should_alert = True
                    severity = "High"
                
                if should_alert:
                    alerts.append({
                        "timestamp": row['timestamp'],
                        "rule": "Impossible Travel",
                        "severity": severity,
                        "entity": user,
                        "entity_type": "user",
                        "details": f"Jump from {row['prev_country']} to {row['country']} in {row['time_diff']:.2f} hours",
                        "recommendation": "Verify user identity and revoke sessions if unauthorized"
                    })

                    
        return alerts
    
    def detect_token_theft(self, session_window_min: int = 60) -> List[Dict]:
        """
        Detects potential token theft: Same session/correlation ID from different IP.
        
        Args:
            session_window_min: Time window to look for session reuse
            
        Returns:
            List of alert dictionaries
        """
        logger.info("Running Token Theft Detection...")
        alerts = []
        
        # Group by correlation_id (session token)
        for corr_id, group in self.df.groupby('correlationId'):
            if len(group) < 2:
                continue
                
            group = group.sort_values('timestamp').copy()
            unique_ips = group['ip'].nunique()
            unique_devices = group['deviceId'].nunique() if 'deviceId' in group.columns else 1
            
            if unique_ips > 1 or unique_devices > 1:
                # Same session, different IP/device = suspicious
                first_event = group.iloc[0]
                last_event = group.iloc[-1]
                
                time_diff = (
                    last_event['timestamp'] - first_event['timestamp']
                ).total_seconds() / 60.0
                
                if time_diff <= session_window_min:
                    alerts.append({
                        "timestamp": last_event['timestamp'],
                        "rule": "Token Theft",
                        "severity": "Critical",
                        "entity": first_event['upn'],
                        "entity_type": "user",
                        "details": f"Session token used from {unique_ips} IPs in {time_diff:.1f} minutes",
                        "recommendation": "Revoke all sessions for user and rotate tokens"
                    })
                    
        return alerts
    
    def detect_privilege_escalation(self, follow_up_window_min: int = 30) -> List[Dict]:
        """
        Detects privilege escalation: Role assignment followed by suspicious activity.
        
        Args:
            follow_up_window_min: Time window after role change to look for suspicious actions
            
        Returns:
            List of alert dictionaries
        """
        logger.info("Running Privilege Escalation Detection...")
        alerts = []
        
        # Find role assignment events
        role_events = self.df[self.df['eventType'] == 'RoleAssigned'].copy()
        
        for idx, role_event in role_events.iterrows():
            user = role_event['upn']
            role_time = role_event['timestamp']
            
            # Look for suspicious follow-up activity
            window_end = role_time + timedelta(minutes=follow_up_window_min)
            
            follow_up = self.df[
                (self.df['upn'] == user) &
                (self.df['timestamp'] > role_time) &
                (self.df['timestamp'] <= window_end) &
                (self.df['eventType'].isin(['AdminAction', 'BulkDownload']))
            ]
            
            if not follow_up.empty:
                for _, action in follow_up.iterrows():
                    alerts.append({
                        "timestamp": action['timestamp'],
                        "rule": "Privilege Escalation",
                        "severity": "Critical",
                        "entity": user,
                        "entity_type": "user",
                        "details": f"{action['eventType']} within {follow_up_window_min}m of role assignment",
                        "recommendation": "Review role assignment legitimacy and revoke if unauthorized"
                    })
                    
        return alerts
    
    def detect_suspicious_user_agent(self) -> List[Dict]:
        """
        Detects logins from suspicious user agents (attack tools, headless browsers).
        
        Returns:
            List of alert dictionaries
        """
        logger.info("Running Suspicious User Agent Detection...")
        alerts = []
        
        for idx, row in self.df.iterrows():
            user_agent = str(row.get('userAgent', ''))
            
            for suspicious in self.SUSPICIOUS_AGENTS:
                if suspicious.lower() in user_agent.lower():
                    alerts.append({
                        "timestamp": row['timestamp'],
                        "rule": "Suspicious User Agent",
                        "severity": "Medium",
                        "entity": row['upn'],
                        "entity_type": "user",
                        "details": f"Login with suspicious agent: {user_agent[:50]}",
                        "recommendation": "Verify automation is authorized"
                    })
                    break
                    
        return alerts
    
    def detect_bulk_operations(self, threshold: int = 5, window_min: int = 10) -> List[Dict]:
        """
        Detects unusual bulk operations (downloads, deletions, etc.).
        
        Args:
            threshold: Minimum operations to flag
            window_min: Time window for aggregation
            
        Returns:
            List of alert dictionaries
        """
        logger.info("Running Bulk Operations Detection...")
        alerts = []
        
        bulk_events = self.df[self.df['eventType'] == 'BulkDownload'].copy()
        
        if bulk_events.empty:
            return alerts
            
        for user, group in bulk_events.groupby('upn'):
            alerts.append({
                "timestamp": group.iloc[0]['timestamp'],
                "rule": "Bulk Download",
                "severity": "High",
                "entity": user,
                "entity_type": "user",
                "details": f"User performed {len(group)} bulk download operations",
                "recommendation": "Investigate data exfiltration and check DLP policies"
            })
            
        return alerts
    
    def detect_suspicious_ip(
        self, 
        min_events: int = 5,
        failure_rate_threshold: float = 0.5,
        min_users_threshold: int = 3
    ) -> List[Dict]:
        """
        Detects suspicious IPs based on their overall behavior patterns.
        Designed for datasets like RBA where attacks are IP-reputation based.
        
        Args:
            min_events: Minimum events to analyze
            failure_rate_threshold: Failure rate to trigger alert
            min_users_threshold: Users targeted to trigger alert
        """
        logger.info("Running Suspicious IP Detection...")
        alerts = []
        
        # Group by IP and calculate stats
        ip_stats = self.df.groupby('ip').agg({
            'upn': 'nunique',
            'status': lambda x: (x == 'Failure').sum(),
            'timestamp': 'count'
        }).rename(columns={'status': 'failures', 'timestamp': 'total', 'upn': 'unique_users'})
        
        ip_stats['failure_rate'] = ip_stats['failures'] / ip_stats['total']
        
        # Flag suspicious IPs
        suspicious = ip_stats[
            (ip_stats['total'] >= min_events) & 
            (
                (ip_stats['failure_rate'] >= failure_rate_threshold) |
                (ip_stats['unique_users'] >= min_users_threshold)
            )
        ]
        
        suspicious_ips = set(suspicious.index)
        
        for ip in suspicious_ips:
            ip_data = ip_stats.loc[ip]
            ip_events = self.df[self.df['ip'] == ip]
            
            for _, event in ip_events.iterrows():
                severity = "High" if ip_data['failure_rate'] > 0.5 else "Medium"
                alerts.append({
                    "timestamp": event['timestamp'],
                    "rule": "Suspicious IP",
                    "severity": severity,
                    "entity": ip,
                    "entity_type": "ip",
                    "details": f"IP {str(ip)[:20]}: {ip_data['failure_rate']*100:.0f}% fail rate, {ip_data['unique_users']:.0f} users",
                    "recommendation": "Investigate IP reputation and consider blocking"
                })
                
        return alerts
    
    def detect_high_risk_asn(
        self, 
        min_events: int = 10,
        failure_rate_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Detects events from high-risk ASNs based on failure patterns.
        ASNs with high failure rates are flagged as suspicious.
        """
        logger.info("Running High-Risk ASN Detection...")
        alerts = []
        
        if 'asn' not in self.df.columns or self.df['asn'].isna().all():
            return alerts
        
        # Group by ASN and calculate failure stats
        asn_stats = self.df.groupby('asn').agg({
            'status': lambda x: (x == 'Failure').sum(),
            'timestamp': 'count'
        }).rename(columns={'status': 'failures', 'timestamp': 'total'})
        
        asn_stats['failure_rate'] = asn_stats['failures'] / asn_stats['total']
        
        # Flag high-risk ASNs
        high_risk = asn_stats[
            (asn_stats['total'] >= min_events) & 
            (asn_stats['failure_rate'] >= failure_rate_threshold)
        ]
        
        high_risk_asns = set(high_risk.index)
        
        for asn in high_risk_asns:
            asn_data = asn_stats.loc[asn]
            asn_events = self.df[self.df['asn'] == asn]
            
            for _, event in asn_events.iterrows():
                severity = "High" if asn_data['failure_rate'] > 0.6 else "Medium"
                alerts.append({
                    "timestamp": event['timestamp'],
                    "rule": "High-Risk ASN",
                    "severity": severity,
                    "entity": str(asn),
                    "entity_type": "asn",
                    "details": f"ASN {asn}: {asn_data['failure_rate']*100:.0f}% failure rate from {asn_data['total']:.0f} events",
                    "recommendation": "Investigate network origin and consider geo-blocking"
                })
                
        return alerts
    
    @staticmethod
    def _load_config(config_path: Optional[str] = None) -> dict:
        """Load config.yaml from given path or common locations."""
        search_paths = [
            config_path,
            os.path.join(os.path.dirname(__file__), 'rules_config.yaml'),
            os.path.join(os.path.dirname(__file__), '..', 'config.yaml'),
            'config.yaml',
        ]
        for path in search_paths:
            if path and os.path.isfile(path):
                try:
                    with open(path) as f:
                        return yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"Failed to load config {path}: {e}")
        return {}

    def run_all(self) -> List[Dict]:
        """
        Run all enabled detection rules dynamically from rules_config.yaml.
        """
        self.alerts = []
        rules = self.config.get('rules', [])
        
        # Fallback to hardcoded sequence if config is missing or empty
        if not rules:
            logger.warning("No rules found in config, falling back to all defaults.")
            self.alerts.extend(self.detect_password_spray())
            self.alerts.extend(self.detect_impossible_travel())
            self.alerts.extend(self.detect_token_theft())
            self.alerts.extend(self.detect_privilege_escalation())
            self.alerts.extend(self.detect_bulk_operations())
            self.alerts.extend(self.detect_suspicious_ip())
            self.alerts.extend(self.detect_high_risk_asn())
            self.alerts.extend(self.detect_suspicious_user_agent())
        else:
            # Execute dynamically configured rules
            for rule_def in rules:
                if not rule_def.get('enabled', True):
                    continue
                    
                handler_name = rule_def.get('handler')
                if not handler_name or not hasattr(self, handler_name):
                    logger.error(f"Rule '{rule_def.get('name')}' specifies missing handler '{handler_name}'")
                    continue
                    
                # Extract params from YAML and execute handler
                params = rule_def.get('params', {})
                handler_func = getattr(self, handler_name)
                
                try:
                    rule_alerts = handler_func(**params)
                    self.alerts.extend(rule_alerts)
                except TypeError as e:
                    logger.error(f"Parameter mismatch for {handler_name}: {e}")
                except Exception as e:
                    logger.error(f"Error executing rule handler {handler_name}: {e}")

        # Enrich all alerts with MITRE ATT&CK technique IDs
        self.alerts = enrich_alerts(self.alerts)

        logger.info(f"Total rules fired: {len(self.alerts)}")
        return self.alerts


if __name__ == "__main__":
    import argparse
    from etl import LogLoader
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="../rba-dataset.csv", help="Path to logs")
    args = parser.parse_args()
    
    try:
        loader = LogLoader(args.file)
        df = loader.load_to_dataframe(nrows=100000)
        engine = RuleEngine(df)
        alerts = engine.run_all()
        
        print(f"\nFound {len(alerts)} alerts:")
        for a in alerts[:10]:
            print(f"  [{a['severity']}] {a['rule']}: {a['details']}")
    except FileNotFoundError:
        print(f"Log file {args.file} not found.")
