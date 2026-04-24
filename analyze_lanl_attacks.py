
import pandas as pd
import gzip

def check_attacks():
    print("Reading redteam.txt.gz...")
    with gzip.open('data/lanl/redteam.txt.gz', 'rt') as f:
        # Read header
        header = f.readline().strip().split(',')
        print(f"Header: {header}")
        
        # Read first few attacks
        attacks = []
        for i, line in enumerate(f):
            parts = line.strip().split(',')
            # LANL format: time,src_user,dst_user,src_comp,dst_comp,auth_type,logon_type,auth_orient,success
            attacks.append({'time': int(parts[0]), 'src_user': parts[1], 'src_comp': parts[3]})
            if i >= 5: break
            
    print("\nFirst 5 attacks:")
    for a in attacks:
        print(a)
        
    print("\nReading auth.txt.gz (first 5 lines)...")
    with gzip.open('data/lanl/auth.txt.gz', 'rt') as f:
        for i, line in enumerate(f):
            if i >= 5: break
            print(f"Line {i}: {line.strip()}")

if __name__ == "__main__":
    check_attacks()
