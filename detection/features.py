import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()
        self.categorical_cols = ['eventType', 'status', 'appName']
        
        # Frequency maps for high-cardinality categoricals
        self.country_freqs = {}
        self.browser_freqs = {}
        self.asn_freqs = {}
        self.os_freqs = {}
        
        # Attack correlation features (fit from training split only)
        self.asn_attack_rate = {}
        self.country_attack_rate = {}
        self.ip_attack_rate = {}  # Per-IP attack history
        
        # Behavioral baselines (per-user and per-IP)
        self.user_ip_counts = {}       # How many unique IPs each user normally uses
        self.user_country_counts = {}  # How many unique countries per user
        self.ip_user_counts = {}       # How many users share each IP
        self.user_device_set = {}      # Typical devices per user
        self.user_hour_mean = {}       # Typical login hour per user
        self.user_hour_std = {}        # Variance in login hour per user

    def fit(self, X, y=None):
        """
        Learn statistics from the training set.
        X: Pandas DataFrame containing raw logs
        """
        # 1. Fit OneHotEncoder for low-cardinality columns
        self.encoder.fit(X[self.categorical_cols])
        
        # 2. Learn frequencies for high-cardinality columns
        self.country_freqs = X['country'].value_counts(normalize=True).to_dict()
        self.browser_freqs = X['browser'].value_counts(normalize=True).to_dict()
        self.asn_freqs = X['asn'].value_counts(normalize=True).to_dict()
        self.os_freqs = X['os'].value_counts(normalize=True).to_dict()
        
        # 3. Learn attack correlation (if labels available)
        if 'is_attack' in X.columns:
            for ip, group in X.groupby('ip'):
                self.ip_attack_rate[ip] = group['is_attack'].fillna(False).mean()
            for asn, group in X.groupby('asn'):
                attack_rate = group['is_attack'].fillna(False).mean()
                self.asn_attack_rate[asn] = attack_rate
            for country, group in X.groupby('country'):
                attack_rate = group['is_attack'].fillna(False).mean()
                self.country_attack_rate[country] = attack_rate
        
        # 4. Learn per-user behavioral baselines
        for user, group in X.groupby('upn'):
            self.user_ip_counts[user] = group['ip'].nunique()
            self.user_country_counts[user] = group['country'].nunique()
            if 'device_name' in group.columns:
                self.user_device_set[user] = set(group['device_name'].dropna().unique())
            else:
                self.user_device_set[user] = set()
            hours = group['timestamp'].dt.hour
            self.user_hour_mean[user] = hours.mean()
            self.user_hour_std[user] = hours.std() if len(hours) > 1 else 4.0
        
        # 5. Learn per-IP user sharing
        for ip, group in X.groupby('ip'):
            self.ip_user_counts[ip] = group['upn'].nunique()
        
        return self

    def transform(self, X):
        """
        Transform raw logs into a numerical feature matrix.
        Enhanced with per-user behavioral deviation features for high precision.
        """
        X = X.copy()
        
        # --- Time Features ---
        X['hour'] = X['timestamp'].dt.hour
        X['hour_sin'] = np.sin(2 * np.pi * X['hour']/24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour']/24)
        X['day_of_week'] = X['timestamp'].dt.dayofweek
        X['is_weekend'] = (X['day_of_week'] >= 5).astype(int)
        X['is_business_hours'] = ((X['hour'] >= 9) & (X['hour'] <= 17)).astype(int)
        
        # --- Categorical Features (One-Hot) ---
        encoded_cats = self.encoder.transform(X[self.categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_cats, 
            columns=self.encoder.get_feature_names_out(self.categorical_cols),
            index=X.index
        )
        
        # --- Categorical Features (Frequency Encoding) ---
        X['country_freq'] = X['country'].map(self.country_freqs).fillna(0)
        X['browser_freq'] = X['browser'].map(self.browser_freqs).fillna(0)
        X['asn_freq'] = X['asn'].map(self.asn_freqs).fillna(0)
        X['os_freq'] = X['os'].map(self.os_freqs).fillna(0)
        
        # --- Attack correlation features ---
        X['ip_attack_rate'] = X['ip'].map(self.ip_attack_rate).fillna(0)
        X['asn_attack_rate'] = X['asn'].map(self.asn_attack_rate).fillna(0)
        X['country_attack_rate'] = X['country'].map(self.country_attack_rate).fillna(0)
        
        # --- Status Feature ---
        X['is_failure'] = (X['status'] == 'Failure').astype(int)
        
        # --- Device Features ---
        X['is_managed'] = X['is_managed'].fillna(False).astype(int)
        X['is_compliant'] = X['is_compliant'].fillna(False).astype(int) if 'is_compliant' in X.columns else 0
        
        # =============================================
        # BEHAVIORAL DEVIATION FEATURES (precision boosters)
        # =============================================
        
        # --- Per-IP user sharing score ---
        # High values = IP shared by many users → credential abuse / spray
        X['ip_user_count'] = X['ip'].map(self.ip_user_counts).fillna(1)
        X['ip_sharing_score'] = np.log1p(X['ip_user_count'])
        
        # --- Per-user IP diversity (baseline) ---
        # If user normally uses 2 IPs but this login is from a new IP range → suspicious
        X['user_typical_ips'] = X['upn'].map(self.user_ip_counts).fillna(1)
        
        # --- Per-user country diversity ---
        X['user_typical_countries'] = X['upn'].map(self.user_country_counts).fillna(1)
        # If user typically accesses from 1 country, multi-country is suspicious
        X['country_anomaly'] = (X['user_typical_countries'] > 2).astype(int)
        
        # --- Login hour deviation from user baseline ---
        X['user_mean_hour'] = X['upn'].map(self.user_hour_mean).fillna(12)
        X['user_std_hour'] = X['upn'].map(self.user_hour_std).fillna(4)
        X['hour_deviation'] = np.abs(X['hour'] - X['user_mean_hour']) / (X['user_std_hour'] + 1e-6)
        X['hour_deviation'] = X['hour_deviation'].clip(0, 5)  # Cap at 5 std devs
        
        # --- Device familiarity ---
        if 'device_name' in X.columns:
            def check_new_device(row):
                user = row['upn']
                device = row.get('device_name', '')
                if user in self.user_device_set and device:
                    return 0 if device in self.user_device_set[user] else 1
                return 0
            X['is_new_device'] = X.apply(check_new_device, axis=1)
        else:
            X['is_new_device'] = 0
        
        # --- Aggregated behavioral features (rolling window) ---
        # Per-user failure rate in the dataset
        user_fail_rates = X.groupby('upn')['is_failure'].transform('mean')
        X['user_fail_rate'] = user_fail_rates
        
        # Per-user event count (activity volume) 
        user_event_counts = X.groupby('upn')['is_failure'].transform('count')
        X['user_event_count'] = np.log1p(user_event_counts)
        
        # --- Interaction features ---
        # High attack rate ASN + failure = strong indicator
        X['asn_attack_x_failure'] = X['asn_attack_rate'] * X['is_failure']
        # High attack rate country + unusual hour = strong indicator
        X['country_attack_x_hour_dev'] = X['country_attack_rate'] * X['hour_deviation']
        
        # Combine all features
        numeric_cols = [
            'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 'is_business_hours',
            'country_freq', 'browser_freq', 'asn_freq', 'os_freq',
            'asn_attack_rate', 'country_attack_rate', 'ip_attack_rate',
            'is_failure', 'is_managed',
            # NEW behavioral features
            'ip_sharing_score', 'user_typical_ips', 'user_typical_countries',
            'country_anomaly', 'hour_deviation', 'is_new_device',
            'user_fail_rate', 'user_event_count',
            'asn_attack_x_failure', 'country_attack_x_hour_dev'
        ]
        
        # Final Feature Matrix
        X_final = pd.concat([X[numeric_cols], encoded_df], axis=1)
        
        # Fill any remaining NaNs
        X_final = X_final.fillna(0)
        
        return X_final

if __name__ == "__main__":
    from etl import LogLoader
    
    loader = LogLoader("../sample_logs.jsonl")
    df = loader.load_to_dataframe()
    
    # Split for demo
    train_df = df.iloc[:200]
    test_df = df.iloc[200:]
    
    extractor = FeatureExtractor()
    extractor.fit(train_df)
    
    X_train = extractor.transform(train_df)
    
    print("Feature Matrix Shape:", X_train.shape)
    print("Features:", X_train.columns.tolist())
    print(X_train.head())
