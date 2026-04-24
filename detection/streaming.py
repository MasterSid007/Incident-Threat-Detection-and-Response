"""
Real-Time Streaming Pipeline for ITDR Prototype.

Simulates a live event stream by reading from the CSV in small batches,
processing each through the full detection pipeline, and maintaining
a sliding window of recent results for the dashboard.
"""
import os
import time
import pandas as pd
import joblib
from collections import deque
from typing import Optional, Dict, List, Tuple


class StreamingPipeline:
    """
    Event-by-event processing pipeline that simulates real-time ingestion.
    
    Reads events in small batches from the dataset, passes them through
    feature extraction → ML prediction → risk scoring, and stores results
    in a sliding-window buffer.
    """

    def __init__(
        self,
        csv_path: str,
        model_path: str,
        extractor_path: str,
        batch_size: int = 100,
        buffer_size: int = 2000,
    ):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.stats = {
            "total_processed": 0,
            "total_alerts": 0,
            "current_offset": 0,
            "events_per_second": 0.0,
            "is_running": False,
        }
        self._model = None
        self._extractor = None
        self._model_path = model_path
        self._extractor_path = extractor_path

    def _load_models(self):
        """Lazy-load the ML model and feature extractor."""
        if self._model is None:
            from detection.models import SupervisedAttackClassifier
            self._extractor = joblib.load(self._extractor_path)
            self._model = SupervisedAttackClassifier()
            self._model.load_model(self._model_path)

    def process_next_batch(self) -> Optional[pd.DataFrame]:
        """
        Read and process the next batch of events from the CSV.
        
        Returns:
            DataFrame of scored events, or None if end of file
        """
        from detection.etl import LogLoader

        self._load_models()
        self.stats["is_running"] = True

        try:
            # Read a small batch with skiprows to simulate streaming
            loader = LogLoader(self.csv_path)
            skip = list(range(1, self.stats["current_offset"] + 1))
            df = loader.load_to_dataframe(nrows=self.batch_size, skiprows=skip if skip else None)

            if df.empty:
                self.stats["is_running"] = False
                return None

            start_time = time.time()

            # Feature extraction & ML prediction
            X = self._extractor.transform(df)
            preds = self._model.predict(X)

            # Attach predictions
            df = df.copy()
            df["attack_probability"] = preds["attack_probability"].values
            df["predicted_attack"] = preds["supervised_pred"].values
            df["final_risk_score"] = (df["attack_probability"] * 100).round(1)
            df["risk_level"] = df["final_risk_score"].apply(
                lambda x: "Critical" if x >= 70 else ("High" if x >= 50 else ("Medium" if x >= 30 else "Low"))
            )

            elapsed = time.time() - start_time

            # Update stats
            self.stats["current_offset"] += len(df)
            self.stats["total_processed"] += len(df)
            num_alerts = int((df["final_risk_score"] >= 50).sum())
            self.stats["total_alerts"] += num_alerts
            self.stats["events_per_second"] = round(len(df) / max(elapsed, 0.001), 1)

            # Add to buffer
            for _, row in df.iterrows():
                self.buffer.append(row.to_dict())

            return df

        except Exception as e:
            self.stats["is_running"] = False
            raise e

    def get_latest_events(self, n: int = 50) -> pd.DataFrame:
        """Return the N most recent events from the buffer."""
        items = list(self.buffer)[-n:]
        if not items:
            return pd.DataFrame()
        return pd.DataFrame(items)

    def get_stats(self) -> Dict:
        """Return current streaming statistics."""
        return self.stats.copy()

    def reset(self):
        """Reset the pipeline to start from the beginning."""
        self.buffer.clear()
        self.stats = {
            "total_processed": 0,
            "total_alerts": 0,
            "current_offset": 0,
            "events_per_second": 0.0,
            "is_running": False,
        }
