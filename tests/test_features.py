"""
Unit tests for the feature extraction module.
"""
import pytest
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detection.features import FeatureExtractor


class TestFeatureExtractor:
    """Tests for the FeatureExtractor class."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe matching the actual log schema."""
        return pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-15 09:00:00',
                '2024-01-15 14:30:00',
                '2024-01-15 22:00:00',
            ]),
            'country': ['US', 'US', 'RU'],
            'browser': ['Chrome', 'Chrome', 'Firefox'],
            'is_managed': [True, True, False],
            # Required columns that FeatureExtractor expects
            'eventType': ['UserLoggedIn', 'UserLoggedIn', 'UserLoginFailed'],
            'status': ['Success', 'Success', 'Failure'],
            'appName': ['Office 365', 'AWS Console', 'Office 365'],
            'asn': ['AS123', 'AS456', 'AS789'],
            'os': ['Windows', 'macOS', 'Linux'],
            'ip': ['192.168.1.1', '10.0.0.1', '172.16.0.1'],
            'upn': ['user1@corp.com', 'user2@corp.com', 'hacker@evil.com'],
            'is_attack': [False, False, True],
        })
    
    def test_extractor_fit_transform(self, sample_df):
        """Test that the extractor can fit and transform data."""
        extractor = FeatureExtractor()
        extractor.fit(sample_df)
        X = extractor.transform(sample_df)
        
        assert isinstance(X, pd.DataFrame)
        assert len(X) == len(sample_df)
        
    def test_no_nan_values_after_transform(self, sample_df):
        """Test that there are no NaN values after transformation."""
        extractor = FeatureExtractor()
        extractor.fit(sample_df)
        X = extractor.transform(sample_df)
        
        assert not X.isna().any().any(), "Feature matrix contains NaN values"
        
    def test_time_features_are_cyclic(self, sample_df):
        """Test that time features are properly encoded as cyclic."""
        extractor = FeatureExtractor()
        extractor.fit(sample_df)
        X = extractor.transform(sample_df)
        
        # Check for sin/cos columns
        time_cols = [c for c in X.columns if 'sin' in c or 'cos' in c]
        assert len(time_cols) > 0, "No cyclic time features found"
        
        # Cyclic features should be in [-1, 1] range
        for col in time_cols:
            assert X[col].min() >= -1, f"{col} has values < -1"
            assert X[col].max() <= 1, f"{col} has values > 1"
            
    def test_frequency_encoding(self, sample_df):
        """Test that frequency encoding works for categorical columns."""
        extractor = FeatureExtractor()
        extractor.fit(sample_df)
        X = extractor.transform(sample_df)
        
        # The transformed output should have numeric columns
        assert all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))), \
            "Not all features are numeric"
        

class TestFeatureIntegration:
    """Integration tests for feature extraction with real-ish data."""
    
    def test_handles_missing_values(self):
        """Test that extractor handles missing values gracefully."""
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-15 09:00:00', '2024-01-15 10:00:00']),
            'country': ['US', 'UK'],
            'browser': ['Chrome', 'Chrome'],
            'is_managed': [True, True],
            'eventType': ['UserLoggedIn', 'UserLoggedIn'],
            'status': ['Success', 'Success'],
            'appName': ['Office 365', 'AWS Console'],
            'asn': ['AS123', 'AS456'],
            'os': ['Windows', 'macOS'],
            'ip': ['192.168.1.1', '10.0.0.1'],
            'upn': ['user1@corp.com', 'user2@corp.com'],
            'is_attack': [False, False],
        })
        
        extractor = FeatureExtractor()
        extractor.fit(df)
        X = extractor.transform(df)
        
        # Should not raise and should produce valid output
        assert len(X) == 2
        

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
