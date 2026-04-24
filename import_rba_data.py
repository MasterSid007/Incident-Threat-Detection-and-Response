#!/usr/bin/env python3
"""
Import script for RBA Dataset.
Converts the Kaggle RBA dataset to our JSONL format.
"""
import pandas as pd
import json
import uuid
from datetime import datetime
import argparse

from typing import Optional

def import_rba_dataset(
    input_file: str = "rba-dataset.csv",
    output_file: str = "sample_logs.jsonl",
    max_rows: Optional[int] = None  # None = use ALL data
):
    """
    Import RBA dataset and convert to our log format.
    
    Args:
        input_file: Path to rba-dataset.csv
        output_file: Output JSONL file
        max_rows: Maximum rows to import (None = all rows)
    """
    print("=" * 60)
    print("   RBA Dataset Importer")
    print("=" * 60)
    
    # Load CSV
    print(f"[+] Loading {input_file}...")
    if max_rows:
        print(f"    (Limited to {max_rows:,} rows)")
        df = pd.read_csv(input_file, nrows=max_rows)
    else:
        print(f"    (Loading ALL rows - this may take a while...)")
        df = pd.read_csv(input_file)
    print(f"    Loaded {len(df):,} rows")
    
    # Stats
    attack_ips = df['Is Attack IP'].sum()
    account_takeovers = df['Is Account Takeover'].sum()
    print(f"    Attack IPs: {attack_ips}")
    print(f"    Account Takeovers: {account_takeovers}")
    
    # Convert to our format
    print("\n[+] Converting to ITDR format...")
    
    events = []
    for _, row in df.iterrows():
        # Determine if this is an attack
        is_attack = row['Is Attack IP'] or row['Is Account Takeover']
        attack_type = None
        if row['Is Account Takeover']:
            attack_type = "AccountTakeover"
        elif row['Is Attack IP']:
            attack_type = "SuspiciousIP"
        
        # Map login success to event type
        event_type = "UserLoggedIn" if row['Login Successful'] else "UserLoginFailed"
        
        # Create event in our schema
        event = {
            "timestamp": row['Login Timestamp'],
            "identity": {
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, str(row['User ID']))),
                "user_principal_name": f"user{row['User ID']}@company.com",
                "display_name": f"User {row['User ID']}",
                "department": "Unknown",
                "role": "User"
            },
            "device": {
                "id": str(uuid.uuid4()),
                "display_name": row['Device Type'] if pd.notna(row['Device Type']) else "Unknown",
                "os": row['OS Name and Version'] if pd.notna(row['OS Name and Version']) else "Unknown",
                "browser": row['Browser Name and Version'] if pd.notna(row['Browser Name and Version']) else "Unknown",
                "is_managed": False,
                "is_compliant": True
            },
            "location": {
                "ip_address": row['IP Address'] if pd.notna(row['IP Address']) else "0.0.0.0",
                "country": row['Country'] if pd.notna(row['Country']) else "Unknown",
                "state": row['Region'] if pd.notna(row['Region']) else "",
                "city": row['City'] if pd.notna(row['City']) else "",
                "asn": str(row['ASN']) if pd.notna(row['ASN']) else ""
            },
            "user_agent": row['User Agent String'] if pd.notna(row['User Agent String']) else "",
            "event_type": event_type,
            "app_name": "WebApp",
            "status": "Success" if row['Login Successful'] else "Failure",
            "correlation_id": str(uuid.uuid4()),
            "is_attack": bool(is_attack),
            "attack_type": attack_type
        }
        events.append(event)
    
    # Save to JSONL
    print(f"\n[+] Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    
    # Summary
    attack_count = sum(1 for e in events if e['is_attack'])
    normal_count = len(events) - attack_count
    
    print("\n" + "=" * 60)
    print("   Import Summary")
    print("=" * 60)
    print(f"   Total Events:  {len(events)}")
    print(f"   Normal:        {normal_count}")
    print(f"   Attacks:       {attack_count}")
    print(f"   Attack Rate:   {attack_count/len(events)*100:.1f}%")
    print("=" * 60)
    print(f"[+] Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import RBA dataset")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Maximum rows to import (default: all rows)")
    parser.add_argument("--input", type=str, default="rba-dataset.csv",
                        help="Input CSV file")
    parser.add_argument("--output", type=str, default="sample_logs.jsonl",
                        help="Output JSONL file")
    args = parser.parse_args()
    
    import_rba_dataset(
        input_file=args.input,
        output_file=args.output,
        max_rows=args.max_rows
    )
