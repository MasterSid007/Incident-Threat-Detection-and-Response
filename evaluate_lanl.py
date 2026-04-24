import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from detection.lanl_loader import LANLLoader
from detection.features import FeatureExtractor
from detection.models import SupervisedAttackClassifier

# Configuration
LANL_AUTH_FILE = r'data\lanl\auth.txt.gz'
LANL_REDTEAM_FILE = r'data\lanl\redteam.txt.gz'
MODEL_PATH = r'saved_models\rba_trained_model.pkl'
WARMUP_EVENTS = 50000  # Number of events to learn user behavior
CHUNK_SIZE = 10000

def evaluate_lanl():
    print("=== LANL Dataset Evaluation (Transfer Learning) ===")
    
    if not os.path.exists(LANL_AUTH_FILE):
        print(f"Error: LANL dataset not found at {LANL_AUTH_FILE}")
        print("Please download auth.txt.gz and redteam.txt.gz to the data/lanl/ folder.")
        return

    # 1. Load Pre-trained RBA Model
    print("Loading pre-trained RBA model...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Run train_rba_model.py first.")
        return
        
    model = SupervisedAttackClassifier()
    model.load_model(MODEL_PATH)
    print("Model loaded successfully.")

    # 2. Initialize Loader
    loader = LANLLoader(LANL_AUTH_FILE, LANL_REDTEAM_FILE, chunk_size=CHUNK_SIZE)
    
    # 3. Warmup Phase (Learn behavior)
    print(f"Starting warmup phase ({WARMUP_EVENTS} events)...")
    warmup_data = []
    total_processed = 0
    
    stream = loader.stream_chunks()
    
    try:
        for chunk in stream:
            warmup_data.append(chunk)
            total_processed += len(chunk)
            print(f"  Warmup loaded: {total_processed} events", end='\r')
            if total_processed >= WARMUP_EVENTS:
                break
                
        warmup_df = pd.concat(warmup_data)
        print(f"\nWarmup complete. Fitting FeatureExtractor on {len(warmup_df)} events...")
        
        # Fit a FRESH extractor on LANL users
        # This learns normal behavior (time, device) for User@LANL
        extractor = FeatureExtractor()
        extractor.fit(warmup_df)
        print("FeatureExtractor fitted on LANL baselines.")
        
    except StopIteration:
        print("Dataset too small for warmup.")
        return

    # 4. evaluation Phase
    print("\nStarting evaluation phase...")
    print("Fast-forwarding to attack window (T=150,000s)...")
    
    y_true_all = []
    y_pred_all = []
    
    # Attack start target
    TARGET_START_TIME = pd.to_datetime(150000, unit='s', origin='unix')
    
    try:
        for i, chunk in enumerate(stream):
            # Fast-forward check
            if chunk['timestamp'].max() < TARGET_START_TIME:
                print(f"  Skipping chunk {i+1} (Max Time: {chunk['timestamp'].max()})...", end='\r')
                continue
                
            # Transform features using LANL-fitted extractor
            # This generates behavioral deviations relative to LANL norms
            X_chunk = extractor.transform(chunk)
            
            # --- TRANSFER LEARNING ALIGNMENT ---
            # The RBA model expects specific columns (e.g. eventType_UserLoggedIn).
            # The LANL data might produce different columns or miss some.
            # We align to the RBA model's schema, filling missing features with 0.
            if model.feature_cols:
                X_chunk = X_chunk.reindex(columns=model.feature_cols, fill_value=0)
            
            # Predict using RBA-trained model weights
            # The model sees "high deviation" and predicts attack
            preds = model.predict(X_chunk)
            
            y_chunk = chunk['is_attack'].fillna(False).astype(int)
            y_pred = preds['supervised_pred'].values
            
            y_true_all.extend(y_chunk)
            y_pred_all.extend(y_pred)
            
            # Check if we caught the attack
            current_attacks = np.sum(y_chunk)
            if current_attacks > 0:
                print(f"\n[!!!] ATTACK FOUND in chunk {i+1}!")
                print(chunk[chunk['is_attack'] == True][['timestamp', 'upn', 'ip', 'eventType', 'appName']])
            
            if i % 10 == 0:
                print(f"  Processed chunk {i+1} ({len(y_pred_all)} evaluated)...", end='\r')
                
            # Stop after finding attacks or limit
            if np.sum(y_true_all) > 5: # Stop after finding 5 attacks
                print("\nFound sufficient attacks. Stopping.")
                break
                
            if len(y_pred_all) > 2000000: # Safety limit 2M evaluated
                print("\nReached 2M event limit for quick eval.")
                break
                
    except Exception as e:
        print(f"\nError during evaluation: {e}")

    # 5. Final Metrics
    print("\n\n=== Final Results ===")
    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    
    attacks = np.sum(y_true)
    detected = np.sum(y_pred[y_true == 1])
    fps = np.sum(y_pred[y_true == 0])
    
    print(f"Total Events Evaluated: {len(y_true)}")
    print(f"Total Red Team Attacks: {attacks}")
    print(f"Attacks Detected: {detected}")
    print(f"False Positives: {fps}")
    
    if attacks > 0:
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"\nPrecision: {prec:.1%}")
        print(f"Recall: {rec:.1%}")
        print(f"F1 Score: {f1:.1%}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Attack']))
    else:
        print("\nNo attacks found in evaluation set.")

if __name__ == "__main__":
    evaluate_lanl()
