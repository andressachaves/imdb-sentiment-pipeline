from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import predict, load_model

app = FastAPI(
    title="IMDB Sentiment API",
    description="Classifica sentimento de reviews com LinearSVC + TF-IDF (92,33% de acurácia)",
    version="1.0.0",
)

@app.on_event("startup")
def startup():
    load_model()

class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    label: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(body: ReviewRequest):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Texto não pode ser vazio.")
    return predict(body.text)