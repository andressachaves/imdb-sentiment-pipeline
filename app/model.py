import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "model" / "linear_svc_tfidf_pipeline.joblib"

_pipeline = None

def load_model():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}. Baixe do Google Drive.")
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline

def predict(text: str) -> dict:
    pipeline = load_model()
    label = int(pipeline.predict([text])[0])
    score = float(pipeline.decision_function([text])[0])
    confidence = round(float(1 / (1 + np.exp(-abs(score)))), 4)
    return {
        "sentiment": "positive" if label == 1 else "negative",
        "confidence": confidence,
        "label": label,
    }