import pandas as pd
import json
from typing import List, Dict, Any

class LogLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_to_dataframe(self, nrows=None, skiprows=None) -> pd.DataFrame:
        """
        Loads JSONL or CSV file into a Pandas DataFrame.
        """
        if self.filepath.endswith('.csv'):
            df = pd.read_csv(self.filepath, nrows=nrows, skiprows=skiprows)
            
            # Map RBA columns to expected schema
            rename_map = {
                'Login Timestamp': 'timestamp',
                'User ID': 'upn',
                'IP Address': 'ip',
                'Country': 'country',
                'City': 'city',
                'ASN': 'asn',
                'User Agent String': 'userAgent',
                'Browser Name and Version': 'browser',
                'OS Name and Version': 'os',
                'Device Type': 'device_name',
                'Login Successful': 'status',
                'Is Attack IP': 'is_attack',
                'Is Account Takeover': 'attack_type'
            }
            df = df.rename(columns=rename_map)
            
            # Additional processing
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['status'] = df['status'].apply(lambda x: 'Success' if x else 'Failure')
            df['is_attack'] = df['is_attack'].fillna(False).astype(bool)
            
            # Map attack types to standard strings
            df['attack_type'] = df.apply(
                lambda row: 'Account Takeover' if row['is_attack'] and row['attack_type'] else ('Attacker IP' if row['is_attack'] else 'None'), 
                axis=1
            )
            
            # Fill missing required columns
            for col in ['correlationId', 'eventType', 'appName', 'is_managed']:
                if col not in df.columns:
                    # Provide defaults
                    if col == 'eventType': df[col] = 'Authentication'
                    elif col == 'is_managed': df[col] = False
                    else: df[col] = 'Unknown'
            
            return df.sort_values("timestamp")
            
        else:
            data = []
            with open(self.filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            
            # Flatten basic structure
            # Note: We need to handle nested dicts like identity, device, location
            processed_data = []
            for entry in data:
                # Get nested objects with defaults
                identity = entry.get("identity", {}) or {}
                device = entry.get("device", {}) or {}
                location = entry.get("location", {}) or {}
                
                flat_entry = {
                    "id": entry.get("id"),
                    "timestamp": pd.to_datetime(entry.get("timestamp")),
                    "correlationId": entry.get("correlation_id") or entry.get("correlationId"),
                    "userAgent": entry.get("user_agent") or entry.get("userAgent"),
                    "eventType": entry.get("event_type") or entry.get("eventType"),
                    "appName": entry.get("app_name") or entry.get("appName"),
                    "status": entry.get("status"),
                    "failureReason": entry.get("failureReason"),
                    
                    # Identity - support both field naming conventions
                    "user_id": identity.get("id"),
                    "upn": identity.get("user_principal_name") or identity.get("upn"),
                    "display_name": identity.get("display_name"),
                    "role": identity.get("role"),
                    "department": identity.get("department"),
                    
                    # Device
                    "device_id": device.get("id"),
                    "device_name": device.get("display_name"),
                    "os": device.get("os"),
                    "browser": device.get("browser"),
                    "is_managed": device.get("is_managed") or device.get("isManaged"),
                    "is_compliant": device.get("is_compliant") or device.get("isCompliant"),
                    
                    # Location
                    "ip": location.get("ip_address") or location.get("ip"),
                    "country": location.get("country"),
                    "state": location.get("state"),
                    "city": location.get("city"),
                    "asn": location.get("asn"),
                    
                    # Label (Ground Truth) - support root level or nested
                    "is_attack": entry.get("is_attack") or entry.get("label", {}).get("isAttack", False),
                    "attack_type": entry.get("attack_type") or entry.get("label", {}).get("attackType")
                }
                processed_data.append(flat_entry)
                
            df = pd.DataFrame(processed_data)
            return df.sort_values("timestamp")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="../rba-dataset.csv", help="Path to logs")
    args = parser.parse_args()
    
    loader = LogLoader(args.file)
    try:
        df = loader.load_to_dataframe()
        print(df.head())
        print(f"\nTotal: {len(df)}")
        print(f"Attacks: {df['is_attack'].sum() if 'is_attack' in df.columns else 'N/A'}")
        print(df.info())
    except FileNotFoundError:
        print(f"File {args.file} not found.")
