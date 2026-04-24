from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score
import pandas as pd
import joblib
import numpy as np
import os


class SupervisedAttackClassifier:
    """
    Precision-targeted ensemble classifier for attack detection.
    Uses RF + HistGBT ensemble with precision-targeted threshold.
    Optimized for scalability on large (1M+) datasets.
    """
    def __init__(self, n_estimators=200):
        # Random Forest - highly parallelizable
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight='balanced_subsample',
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # HistGradientBoosting - very fast even on large datasets
        hgb = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_depth=8,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42
        )
        
        self.model = VotingClassifier(
            estimators=[('rf', rf), ('hgb', hgb)],
            voting='soft',
            weights=[1, 1]
        )
        self.is_fitted = False
        self.feature_cols = None
        self.optimal_threshold = 0.5
        self.target_precision = 0.88  # Set above 80% to account for cal/test variance
        
    def _find_precision_target_threshold(self, y_true, y_probs, target_precision=0.80):
        """
        Find the threshold that MAXIMIZES RECALL while keeping precision >= target.
        Searches the full precision-recall curve for the optimal operating point.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        
        # Find ALL thresholds where precision >= target
        valid_mask = precision[:-1] >= target_precision
        
        if valid_mask.any():
            valid_indices = np.where(valid_mask)[0]
            # Among valid thresholds, pick the one with HIGHEST recall
            best_valid_idx = valid_indices[np.argmax(recall[valid_indices])]
            best_threshold = thresholds[best_valid_idx]
            achieved_precision = precision[best_valid_idx]
            achieved_recall = recall[valid_indices[np.argmax(recall[valid_indices])]] # Corrected to use best_valid_idx for recall
            print(f"  [OK] Found threshold achieving >={target_precision:.0%} precision with max recall")
        else:
            # Fall back to F1-optimal
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
            best_valid_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_valid_idx]
            achieved_precision = precision[best_valid_idx]
            achieved_recall = recall[best_valid_idx]
            print(f"  [WARN] Target precision {target_precision:.0%} not achievable, using F1-optimal")
        
        print(f"  Threshold: {best_threshold:.4f} -> Precision: {achieved_precision:.1%}, Recall: {achieved_recall:.1%}")
        return best_threshold, achieved_precision, achieved_recall
        
    def train(self, X: pd.DataFrame, y: pd.Series, test_size=0.2):
        """
        Train with 3-way split: train / calibrate / test.
        Skips expensive cross-validation on large datasets for speed.
        """
        print("Training Precision-Targeted Ensemble Classifier...")
        self.feature_cols = X.columns.tolist()
        n_samples = len(X)
        
        # 3-way split: 60% train, 20% calibrate, 20% test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
        )
        
        # Cross-validation only on smaller datasets (< 50K rows)
        if n_samples < 50000:
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
            print(f"Cross-Validation Accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*2*100:.1f}%)")
        else:
            print(f"Skipping CV for large dataset ({n_samples} rows) - using cal set for validation")
        
        # Train on training set
        print(f"Training on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Find optimal threshold on calibration set (prevents overfitting the threshold)
        y_cal_probs = self.model.predict_proba(X_cal)[:, 1]
        self.optimal_threshold, prec_cal, rec_cal = self._find_precision_target_threshold(
            y_cal, y_cal_probs, self.target_precision
        )
        
        # Final evaluation on held-out test set
        y_test_probs = self.model.predict_proba(X_test)[:, 1]
        y_pred_optimized = (y_test_probs >= self.optimal_threshold).astype(int)
        acc = accuracy_score(y_test, y_pred_optimized)
        prec = precision_score(y_test, y_pred_optimized, zero_division=0)
        rec = recall_score(y_test, y_pred_optimized, zero_division=0)
        f1 = f1_score(y_test, y_pred_optimized)
        
        print(f"\n=== HELD-OUT TEST SET Evaluation ===")
        print(f"Threshold: {self.optimal_threshold:.4f}")
        print(f"Accuracy: {acc*100:.1f}%")
        print(f"Precision: {prec:.1%}  (target: >=80%)")
        print(f"Recall: {rec:.1%}")
        print(f"F1 Score: {f1:.1%}")
        print(classification_report(y_test, y_pred_optimized, target_names=['Normal', 'Attack']))
        
        return acc
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict attack probability using the optimal threshold."""
        if not self.is_fitted:
            raise ValueError("Model not trained!")
            
        # Ensure same columns
        X_aligned = X[self.feature_cols] if self.feature_cols else X
        
        probas = self.model.predict_proba(X_aligned)[:, 1]
        preds = (probas >= self.optimal_threshold).astype(int)
        
        return pd.DataFrame({
            'attack_probability': probas,
            'supervised_pred': preds
        })
        
    def save_model(self, filepath='saved_models/supervised_model.pkl'):
        """Save the trained model and metadata."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'model': self.model,
            'feature_cols': self.feature_cols,
            'optimal_threshold': self.optimal_threshold,
            'target_precision': self.target_precision
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath='saved_models/supervised_model.pkl'):
        """Load a trained model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
            
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_cols = model_data['feature_cols']
        self.optimal_threshold = model_data.get('optimal_threshold', 0.5)
        self.target_precision = model_data.get('target_precision', 0.88)
        self.is_fitted = True
        print(f"Model loaded from {filepath}")

class AnomalyDetector:
    def __init__(self, contamination=0.01):
        """
        contamination: unexpected proportion of outliers in the dataset.
        For security, we assume attacks are rare (<1-5%).
        """
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination, 
            random_state=42,
            n_jobs=-1
        )
        self.is_fitted = False

    def train(self, X_train: pd.DataFrame):
        """
        Train the Isolation Forest model. 
         Ideally, X_train should be mostly "normal" behavior.
        """
        print("Training Isolation Forest...")
        self.model.fit(X_train)
        self.is_fitted = True

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
        - anomaly_score: float (lower is more anomalous)
        - is_anomaly: bool (True if outlier)
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet!")
            
        # Decision function: average anomaly score of X of the base classifiers.
        # The anomaly score of an input sample is computed as the mean anomaly score of the trees in the forest.
        # The measure of normality of an observation given a tree is the depth of the leaf containing this observation.
        # Outliers tend to have shorter path lengths.
        
        scores = self.model.decision_function(X_test)
        preds = self.model.predict(X_test) # -1 for outlier, 1 for inlier
        
        results = pd.DataFrame(index=X_test.index)
        results['anomaly_score'] = scores # negative scores are anomalies
        results['is_anomaly'] = preds == -1
        
        return results

    def save(self, filepath="model.pkl"):
        joblib.dump(self.model, filepath)

    def load(self, filepath="model.pkl"):
        self.model = joblib.load(filepath)
        self.is_fitted = True


class Autoencoder:
    """
    Simple Autoencoder for anomaly detection using reconstruction error.
    Higher reconstruction error = more anomalous.
    """
    def __init__(self, input_dim=None, encoding_dim=8, learning_rate=0.001):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.model = None
        self.threshold = None
        self.is_fitted = False
        
    def _build_model(self, input_dim):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            print("PyTorch not installed. Autoencoder unavailable.")
            return None
            
        class AutoencoderNet(nn.Module):
            def __init__(self, input_dim, encoding_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, encoding_dim),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, input_dim),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
                
        return AutoencoderNet(input_dim, self.encoding_dim)
    
    def train(self, X_train: pd.DataFrame, epochs=50, batch_size=32):
        """Train the autoencoder on normal data."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            print("PyTorch not installed. Skipping Autoencoder training.")
            return
            
        print("Training Autoencoder...")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_train.values)
        dataset = TensorDataset(X_tensor, X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Build model
        self.input_dim = X_train.shape[1]
        self.model = self._build_model(self.input_dim)
        if self.model is None:
            return
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, _ in loader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
        
        # Calculate threshold (95th percentile of reconstruction error on training data)
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
            self.threshold = np.percentile(errors, 95)
            
        self.is_fitted = True
        print(f"Autoencoder trained. Anomaly threshold: {self.threshold:.4f}")
    
    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Predict anomalies based on reconstruction error."""
        if not self.is_fitted or self.model is None:
            return pd.DataFrame(index=X_test.index)
            
        import torch
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X_test.values)
        
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
        
        results = pd.DataFrame(index=X_test.index)
        results['reconstruction_error'] = errors
        results['ae_is_anomaly'] = errors > self.threshold
        
        return results


class UserProfiler:
    """
    Maintains per-user behavioral baselines.
    Flags deviations from a user's own typical patterns.
    """
    def __init__(self):
        self.user_profiles = {}  # {user_id: {'mean': array, 'std': array}}
        self.feature_cols = None
        
    def fit(self, df: pd.DataFrame, X_features: pd.DataFrame, user_col='upn'):
        """
        Build per-user profiles from historical data.
        df: Original dataframe with user identifiers
        X_features: Transformed feature matrix
        """
        print("Building per-user behavioral profiles...")
        
        self.feature_cols = X_features.columns.tolist()
        
        # Merge user info with features
        combined = X_features.copy()
        combined['_user'] = df[user_col].values
        
        # Calculate per-user statistics
        for user, group in combined.groupby('_user'):
            feature_data = group.drop('_user', axis=1)
            self.user_profiles[user] = {
                'mean': feature_data.mean().values,
                'std': feature_data.std().values + 1e-6,  # Avoid division by zero
                'count': len(group)
            }
            
        print(f"Built profiles for {len(self.user_profiles)} users.")
        
    def score(self, df: pd.DataFrame, X_features: pd.DataFrame, user_col='upn') -> pd.DataFrame:
        """
        Score each event based on deviation from user's baseline.
        Returns z-score based deviation.
        """
        results = pd.DataFrame(index=X_features.index)
        results['user_deviation_score'] = 0.0
        
        combined = X_features.copy()
        combined['_user'] = df[user_col].values
        
        for idx, row in combined.iterrows():
            user = row['_user']
            if user in self.user_profiles:
                profile = self.user_profiles[user]
                features = row.drop('_user').values
                
                # Calculate average z-score across all features
                z_scores = np.abs((features - profile['mean']) / profile['std'])
                avg_z = np.mean(z_scores)
                results.loc[idx, 'user_deviation_score'] = avg_z
            else:
                # Unknown user - flag as suspicious
                results.loc[idx, 'user_deviation_score'] = 5.0
                
        return results


if __name__ == "__main__":
    # Test Run
    from etl import LogLoader
    from features import FeatureExtractor
    
    # 1. Load Data
    loader = LogLoader("../sample_logs.jsonl")
    df = loader.load_to_dataframe()
    
    # 2. Extract Features
    extractor = FeatureExtractor()
    extractor.fit(df)
    X = extractor.transform(df)
    
    # 3. Train Model
    detector = AnomalyDetector(contamination=0.05)
    detector.train(X)
    
    # 4. Predict
    results = detector.predict(X)
    
    # 5. Inspect Results
    df_results = pd.concat([df, results], axis=1)
    
    print("Detected Anomalies:")
    anomalies = df_results[df_results['is_anomaly']]
    print(anomalies[['timestamp', 'event_type', 'is_attack', 'attack_type', 'anomaly_score']])
    
    # Evaluation on known attacks
    true_positives = anomalies[anomalies['is_attack'] == True]
    print(f"\nCaught {len(true_positives)} actual attacks out of {len(df[df['is_attack']==True])} total attacks.")
