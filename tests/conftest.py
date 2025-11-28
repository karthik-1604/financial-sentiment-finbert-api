import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from fastapi.testclient import TestClient
from main import app
import main
import database
from predictor import SentimentPredictor

MODEL_DIR = r'C:\Users\Karthik\Desktop\BNP_PARIBAS\Financial-Sentiment-FinBERT\finbert_finetuned'

@pytest.fixture(scope='session', autouse=True)
def setup():
    database.init_db()
    main.predictor = SentimentPredictor(model_dir=MODEL_DIR)