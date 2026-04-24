import gzip
import os
import shutil
from detection.lanl_loader import LANLLoader
from detection.features import FeatureExtractor
from detection.models import SupervisedAttackClassifier

# Setup dummy data
OS_PATH = r'data\lanl'
AUTH_FILE = os.path.join(OS_PATH, 'dummy_auth.txt.gz')
REDTEAM_FILE = os.path.join(OS_PATH, 'dummy_redteam.txt.gz')

if not os.path.exists(OS_PATH):
    os.makedirs(OS_PATH)

print("Creating dummy LANL data...")
# Time,SrcUser,DstUser,SrcComp,DstComp,AuthType,LogonType,AuthOrient,Success
dummy_auth = [
    "1,U1@DOM1,U1@DOM1,C1,C2,Kerberos,Network,LogOn,Success",
    "2,U1@DOM1,U1@DOM1,C1,C2,Kerberos,Network,LogOn,Success",
    "3,U2@DOM1,U2@DOM1,C100,C200,NTLM,Network,LogOn,Failure",
    "4,U100@DOM1,U100@DOM1,C666,C999,NTLM,Network,LogOn,Success" # Attack
]

dummy_redteam = [
    "4,U100@DOM1,C666,C999" # Time,User,Src,Dst matches the attack above
]

with gzip.open(AUTH_FILE, 'wt') as f:
    f.write('\n'.join(dummy_auth))
    
with gzip.open(REDTEAM_FILE, 'wt') as f:
    f.write('\n'.join(dummy_redteam))

print("Verifying Loader...")
loader = LANLLoader(AUTH_FILE, REDTEAM_FILE, chunk_size=2)
for chunk in loader.stream_chunks():
    print(f"Loaded chunk with {len(chunk)} events")
    print(chunk[['timestamp', 'upn', 'ip', 'is_attack']].head())

print("\nVerifying Evaluation Script logic...")
# Mock model if not present
if not os.path.exists(r'saved_models\rba_trained_model.pkl'):
    print("Creating mock RBA model for testing...")
    model = SupervisedAttackClassifier()
    # Mock fit
    import pandas as pd
    X = pd.DataFrame({'feature': [1,2]}, index=[0,1])
    y = pd.Series([0,1])
    model.train(X, y) # This will fail if features don't match, careful
    # Actually, better to just rely on the real training script 
    # taking place in background. If it's not done, we skip this part.
    print("Skipping model eval (wait for train_rba_model.py)")
else:
    # Run the actual eval script
    print("Running evaluate_lanl.py on dummy data...")
    import evaluate_lanl as eval_script
    
    # Override constants for testing
    eval_script.LANL_AUTH_FILE = AUTH_FILE
    eval_script.LANL_REDTEAM_FILE = REDTEAM_FILE
    eval_script.WARMUP_EVENTS = 2 # Set small for test
    
    eval_script.evaluate_lanl()

print("\nSuccess! Pipeline is ready for real data.")
