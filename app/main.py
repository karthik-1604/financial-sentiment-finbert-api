import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from predictor import SentimentPredictor
from database import init_db, log_prediction, get_recent_predictions, get_prediction_stats
from models import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    HistoryResponse, StatsResponse, HealthResponse,
    Probabilities, PredictionRecord
)

MODEL_DIR = r'C:\Users\Karthik\Desktop\BNP_PARIBAS\Financial-Sentiment-FinBERT\finbert_finetuned'

predictor: SentimentPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and initialize database on startup."""
    global predictor
    init_db()
    predictor = SentimentPredictor(model_dir=MODEL_DIR)
    print(f'Model loaded from {MODEL_DIR}')
    print(f'Running on device: {predictor.device}')
    yield


app = FastAPI(
    title='Financial Sentiment Analysis API',
    description='Production-ready REST API for real-time sentiment analysis on financial news and trading signals using fine-tuned FinBERT.',
    version='1.0.0',
    lifespan=lifespan
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')


@app.get('/', response_class=FileResponse)
def serve_ui():
    return os.path.join(STATIC_DIR, 'index.html')


@app.get('/health', response_model=HealthResponse, tags=['Health'])
def health_check():
    """Check if the API and model are running correctly."""
    return HealthResponse(
        status='healthy',
        model='ProsusAI/finbert (fine-tuned)',
        device=str(predictor.device)
    )


@app.post('/predict', response_model=PredictResponse, tags=['Prediction'])
def predict(request: PredictRequest):
    """
    Predict sentiment for a single financial text.
    Returns sentiment label (positive/neutral/negative), confidence score,
    and probability distribution across all three classes.
    """
    try:
        result = predictor.predict(request.text)
        log_prediction(
            text=result['text'],
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            probabilities=result['probabilities']
        )
        return PredictResponse(
            text=result['text'],
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            probabilities=Probabilities(**result['probabilities'])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predict/batch', response_model=BatchPredictResponse, tags=['Prediction'])
def predict_batch(request: BatchPredictRequest):
    """
    Predict sentiment for a batch of financial texts.
    Maximum 32 texts per request.
    """
    try:
        results = predictor.predict_batch(request.texts)
        responses = []
        for r in results:
            log_prediction(
                text=r['text'],
                sentiment=r['sentiment'],
                confidence=r['confidence'],
                probabilities=r['probabilities']
            )
            responses.append(PredictResponse(
                text=r['text'],
                sentiment=r['sentiment'],
                confidence=r['confidence'],
                probabilities=Probabilities(**r['probabilities'])
            ))
        return BatchPredictResponse(results=responses, total=len(responses))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/history', response_model=HistoryResponse, tags=['Monitoring'])
def prediction_history(limit: int = 10):
    """
    Retrieve the most recent predictions from the database.
    Default limit is 10, maximum is 100.
    """
    if limit > 100:
        raise HTTPException(status_code=400, detail='Limit cannot exceed 100')
    rows = get_recent_predictions(limit=limit)
    records = [
        PredictionRecord(
            id=r['id'],
            text=r['text'],
            sentiment=r['sentiment'],
            confidence=r['confidence'],
            probabilities=Probabilities(**r['probabilities']),
            created_at=r['created_at']
        )
        for r in rows
    ]
    return HistoryResponse(predictions=records, total=len(records))


@app.get('/stats', response_model=StatsResponse, tags=['Monitoring'])
def prediction_stats():
    """
    Return aggregate statistics over all logged predictions.
    Shows total prediction count and sentiment distribution.
    """
    stats = get_prediction_stats()
    return StatsResponse(
        total_predictions=stats['total_predictions'],
        distribution=stats['distribution']
    )