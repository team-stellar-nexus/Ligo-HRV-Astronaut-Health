import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MultiWindowAnomalyDetector:
    """
    Stage 3: Multi-window anomaly fusion engine.
    Maintains sliding buffers for different window durations and fuses decisions.
    """

    def __init__(self, thresholds: dict[str, float], required_active_windows: int = 2):
        """
        Args:
            thresholds: dict mapping window sizes (e.g. '30s', '60s') to their error thresholds.
            required_active_windows: Min number of windows that must exceed threshold to trigger alert.
        """
        self.thresholds = thresholds
        self.required_active = required_active_windows

    def evaluate_timestep(self, errors_per_window: dict[str, float]) -> dict:
        """
        Evaluate anomalies across multiple time windows.

        Args:
            errors_per_window: {'30s': 0.15, '60s': 0.22, '120s': 0.08, '240s': 0.05}

        Returns:
            Dict containing alert status, active window count, and normalized risk score (0-100).
        """
        active_count = 0
        normalized_scores = []

        for w_name, err in errors_per_window.items():
            thresh = self.thresholds.get(w_name, 0.1)
            is_active = err > thresh
            if is_active:
                active_count += 1
            # Ratio of error to threshold as normalized score component
            normalized_scores.append(err / (thresh + 1e-8))

        risk_score = float(np.mean(normalized_scores) * 25.0)  # Scale to ~0-100
        risk_score = min(max(risk_score, 0.0), 100.0)

        # Determine alert level
        if active_count >= self.required_active + 1:
            status = "CONFIRMED_STRESS"
        elif active_count >= self.required_active:
            status = "SUSPECTED_STRESS"
        else:
            status = "NORMAL"

        return {
            "status": status,
            "active_windows": active_count,
            "risk_score": round(risk_score, 1),
            "window_details": {w: {"error": round(err, 4), "threshold": self.thresholds.get(w, 0), "exceeded": err > self.thresholds.get(w, 0)}
                               for w, err in errors_per_window.items()}
        }


class WitnessChannelValidator:
    """
    Stage 4: Witness Channel Validation (OmegaNeuron / GravitySpy style).
    Cross-validates primary sensor anomaly with auxiliary channels.
    """

    def __init__(self, channel_embeddings: dict[str, np.ndarray]):
        """
        Args:
            channel_embeddings: Baseline embedding vectors per channel for cosine similarity reference.
        """
        self.baseline_embeddings = channel_embeddings

    def validate_anomaly(self, primary_channel: str, current_embeddings: dict[str, np.ndarray]) -> dict:
        """
        Check if auxiliary channels co-vary with the primary channel during an anomaly.

        Args:
            primary_channel: e.g. 'E4_BVP' or 'PolarH10_ECG'
            current_embeddings: dict mapping channel name -> embedding vector

        Returns:
            Dict with witness channels and confirmation score.
        """
        if primary_channel not in current_embeddings:
            return {"witness_channels": [], "correlation_score": 0.0, "confirmed": False}

        primary_emb = current_embeddings[primary_channel].reshape(1, -1)
        witnesses = []
        similarities = {}

        for chan, emb in current_embeddings.items():
            if chan == primary_channel:
                continue
            sim = float(cosine_similarity(primary_emb, emb.reshape(1, -1))[0, 0])
            similarities[chan] = round(sim, 3)
            # If high cosine similarity, it's a witness channel
            if sim > 0.65:
                witnesses.append(chan)

        confirmed = len(witnesses) >= 1

        return {
            "witness_channels": witnesses,
            "similarities": similarities,
            "confirmed": confirmed
        }


if __name__ == "__main__":
    # Test multi-window fusion
    detector = MultiWindowAnomalyDetector({"30s": 0.1, "60s": 0.12, "120s": 0.15, "240s": 0.18})
    result = detector.evaluate_timestep({"30s": 0.15, "60s": 0.18, "120s": 0.10, "240s": 0.09})
    print("Fusion test result:", result)
