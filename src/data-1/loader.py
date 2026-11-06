import os
from pathlib import Path
import pandas as pd
import numpy as np

# Dictionary mapping our study event sessions to stress risk tiers
# 0 = Normal / Baseline / Rest
# 1 = Moderate Cognitive Load / Prep
# 2 = High Stress (TSST Speech / SSST Singing)
SESSION_STRESS_MAP = {
    "adaptation":      0,
    "baseline":        0,
    "rest-1":          0,
    "rest-2":          0,
    "rest-3":          0,
    "meditation-1":    0,
    "meditation-2":    0,
    "tsst-prep":       1,
    "tsst-speech":     2,   # Major acute stress trigger
    "screen-reading":  1,
    "ssst-prep":       1,
    "ssst-sing":       2,
    "keyboard-typing": 1,
    "mobile-typing":   1,
    "standing":        0,
    "walking":         1,
    "recovery":        0,
}

# Custom data loader
class AstronautDataLoader:
    def __init__(self, base_data_dir: str):
        self.base_dir = Path(base_data_dir)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Whoops! Can't find data directory at: {self.base_dir}")
        
    # Scans dir and return list of participant folders
    def get_available_participants(self) -> list:
        return sorted([d.name for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith("P")])

    # Loads individual sensor CSV file + Cleans header whitespaces, normalises timestamps to Unix sec
    def load_sensor_csv(self, participant_id: str, device_name: str, signal_name: str) -> pd.DataFrame: 
        file_path = self.base_dir / participant_id / device_name / f"{signal_name}.csv"

        if not file_path.exists():
            return None

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        timestamp_col = None
        for col in df.columns:
            col_lower = col.lower()
            if "timestamp" in col_lower or "timereceived" in col_lower or "datareceived" in col_lower:
                timestamp_col = col
                break

        if timestamp_col and timestamp_col != "timestamp":
            df = df.rename(columns={timestamp_col: "timestamp"})

        if "timestamp" in df.columns:
            first_val = float(df["timestamp"].iloc[0])
            if first_val > 1e15:
                # Empatica E4, microsecond timestamps -> seconds
                df["timestamp"] = df["timestamp"] / 1e6
            elif first_val > 1e12:
                # Millisecond timestamps -> seconds
                df["timestamp"] = df["timestamp"] / 1e3
        df["device"] = device_name
        df["signal"] = signal_name
        df["participant_id"] = participant_id

        return df

    # Loads Event.csv
    def load_participant_events(self, participant_id: str) -> pd.DataFrame:
        event_path = self.base_dir / participant_id / "Event.csv"
        if not event_path.exists():
            return None

        df = pd.read_csv(event_path)
        df.columns = df.columns.str.strip()

        if "timestamp" in df.columns:
            ts = float(df["timestamp"].iloc[0])
            if ts > 1e12:
                df["timestamp"] = df["timestamp"] / 1e3

        # Maps session names to numeric stress risk levels
        if "session" in df.columns:
            df["stress_label"] = df["session"].map(SESSION_STRESS_MAP).fillna(1).astype(int)

        return df

    # Loads all available sensors and event logs for a single participant into memory
    def load_complete_participant_profile(self, participant_id: str) -> dict:
        profile = {
            "participant": participant_id,
            "events": self.load_participant_events(participant_id),
            "signals": {}
        }

        devices_to_check = {
            "E4": ["BVP", "HR", "ACC", "IBI", "TEMP"],
            "PolarH10": ["ECG", "HR", "ACC", "IBI"],
            "GalaxyWatch": ["PPG", "HR", "ACC", "SkinTemp"]
        }

        for dev, signals in devices_to_check.items():
            for sig in signals:
                df = self.load_sensor_csv(participant_id, dev, sig)
                if df is not None:
                    profile["signals"][f"{dev}_{sig}"] = df

        return profile


if __name__ == "__main__":
    # Quick test run
    print("[DEBUG] Testing AstronautDataLoader locally...")
    loader = AstronautDataLoader("Ligo-HRV-Astronaut-Health/Dataset")
    pids = loader.get_available_participants()
    print(f"Found participants: {pids}")

    if pids:
        sample_data = loader.load_complete_participant_profile(pids[0])
        print(f"Successfully loaded profile for {pids[0]}: {list(sample_data['signals'].keys())}")
