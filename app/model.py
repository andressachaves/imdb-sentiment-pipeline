import joblib
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download

MODEL_PATH = Path(__file__).parent.parent / "model" / "linear_svc_tfidf_pipeline.joblib"

_pipeline = None

def load_model():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            MODEL_PATH.parent.mkdir(exist_ok=True)
            hf_hub_download(
                repo_id="anddz/imdb-sentiment-linearsvc",
                filename="linear_svc_tfidf_pipeline.joblib",
                local_dir=MODEL_PATH.parent
            )
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