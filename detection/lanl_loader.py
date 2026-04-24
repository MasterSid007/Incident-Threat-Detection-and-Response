import gzip
import pandas as pd
import hashlib

class LANLLoader:
    """
    Streaming loader for the LANL 'auth.txt.gz' dataset.
    
    The dataset is too large (58GB uncompressed) to load into RAM.
    This loader reads it line-by-line and yields chunks for processing.
    
    Schema mapping:
    - time (int) -> timestamp (datetime)
    - src_user@domain -> user (upn)
    - src_computer -> ip (simulated via hash for consistency)
    - dst_computer -> server
    - auth_type -> method
    - logon_type -> logon_type
    """
    
    def __init__(self, auth_file: str, redteam_file: str = None, chunk_size=100000):
        self.auth_file = auth_file
        self.redteam_file = redteam_file
        self.chunk_size = chunk_size
        self.redteam_events = set()
        
        if redteam_file:
            self._load_redteam_labels()
            
    def _load_redteam_labels(self):
        """Pre-load all red team events into a set for fast lookup."""
        print(f"Loading red team labels from {self.redteam_file}...")
        try:
            with gzip.open(self.redteam_file, 'rt') as f:
                for line in f:
                    parts = line.strip().split(',')
                    # Key: (time, src_user, src_comp, dst_comp)
                    # Note: Red team file format is slightly different, check LANL docs
                    # LANL Redteam: time,user@domain,src_comp,dst_comp
                    if len(parts) >= 4:
                        key = tuple(parts[:4])
                        self.redteam_events.add(key)
            print(f"Loaded {len(self.redteam_events)} red team events.")
        except Exception as e:
            print(f"Warning: Could not load red team labels: {e}")

    def _map_computer_to_ip(self, computer_name):
        """Simulate an IP address from a computer name for compatibility."""
        if not computer_name or computer_name == '?':
            return '192.168.0.1'
        
        # Consistent mapping: C123 -> 10.0.0.123 (simplified)
        # Use hash to get 4 octets
        h = int(hashlib.md5(computer_name.encode()).hexdigest(), 16)
        return f"10.{(h >> 16) & 0xFF}.{(h >> 8) & 0xFF}.{h & 0xFF}"

    def stream_chunks(self):
        """Yield pandas DataFrames in chunks."""
        chunk = []
        
        with gzip.open(self.auth_file, 'rt') as f:
            for line in f:
                parts = line.strip().split(',')
                # auth.txt columns:
                # 0: time
                # 1: src_user@domain
                # 2: dst_user@domain
                # 3: src_comp
                # 4: dst_comp
                # 5: auth_type
                # 6: logon_type
                # 7: auth_orient
                # 8: success/failure
                
                if len(parts) < 9:
                    continue
                    
                timestamp = int(parts[0])
                src_user = parts[1]
                src_comp = parts[3]
                dst_comp = parts[4]
                
                # Check label
                # Match redteam criteria: time, user, src, dst
                is_attack = (parts[0], parts[1], parts[3], parts[4]) in self.redteam_events
                
                # Map to our ITDR schema
                event = {
                    'timestamp': pd.to_datetime(timestamp, unit='s', origin='unix'),
                    'upn': src_user,
                    'ip': self._map_computer_to_ip(src_comp),
                    'country': 'Internal', # LANL is internal network
                    'city': 'Los Alamos',
                    'asn': 'Internal',
                    'user_agent': 'Windows Auth',
                    'os': 'Windows',
                    'browser': 'N/A',
                    'status': 'Success' if parts[8] == 'Success' else 'Failure',
                    'is_attack': is_attack,
                    # Mapped fields for FeatureExtractor
                    'eventType': 'UserLoggedIn' if parts[7] in ['LogOn', 'TGS', 'TGT'] else 'UserLoginFailed' if parts[8] == 'Fail' else parts[7],
                    'appName': parts[5],   # Kerberos, NTLM, etc.
                    'device_name': src_comp, # For new device detection
                    'is_managed': True,      # Assume internal devices are managed
                    'is_compliant': True,    # Assume internal devices are compliant
                    # Additional LANL fields preserved
                    'auth_type': parts[5],
                    'logon_type': parts[6],
                    'auth_orient': parts[7],
                    'source_computer': src_comp,
                    'dest_computer': dst_comp
                }
                
                chunk.append(event)
                
                if len(chunk) >= self.chunk_size:
                    yield pd.DataFrame(chunk)
                    chunk = []
            
            # Yield remaining
            if chunk:
                yield pd.DataFrame(chunk)
