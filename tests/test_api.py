import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from main import app

client = TestClient(app)


def test_health_check():
    """Health endpoint should return healthy status."""
    res = client.get('/health')
    assert res.status_code == 200
    data = res.json()
    assert data['status'] == 'healthy'
    assert 'model' in data
    assert 'device' in data


def test_predict_positive():
    """Clearly positive financial text should return positive sentiment."""
    res = client.post('/predict', json={
        'text': 'The company reported record profits and raised its annual dividend by 25%.'
    })
    assert res.status_code == 200
    data = res.json()
    assert data['sentiment'] == 'positive'
    assert data['confidence'] > 0.8
    assert 'probabilities' in data
    assert set(data['probabilities'].keys()) == {'positive', 'neutral', 'negative'}


def test_predict_negative():
    """Clearly negative financial text should return negative sentiment."""
    res = client.post('/predict', json={
        'text': 'The company defaulted on its debt obligations and filed for bankruptcy.'
    })
    assert res.status_code == 200
    data = res.json()
    assert data['sentiment'] == 'negative'
    assert data['confidence'] > 0.8


def test_predict_neutral():
    """Neutral financial text should return neutral sentiment."""
    res = client.post('/predict', json={
        'text': 'The company will release its quarterly earnings report next Monday.'
    })
    assert res.status_code == 200
    data = res.json()
    assert data['sentiment'] in ['neutral', 'positive', 'negative']
    assert 0 < data['confidence'] <= 1.0


def test_predict_empty_text():
    """Empty text should return 422 validation error."""
    res = client.post('/predict', json={'text': ''})
    assert res.status_code == 422


def test_predict_missing_field():
    """Missing text field should return 422 validation error."""
    res = client.post('/predict', json={})
    assert res.status_code == 422


def test_predict_batch():
    """Batch prediction should return results for all input texts."""
    texts = [
        'The company reported record profits this quarter.',
        'The firm faces serious liquidity concerns.',
        'Revenue remained stable compared to last year.'
    ]
    res = client.post('/predict/batch', json={'texts': texts})
    assert res.status_code == 200
    data = res.json()
    assert data['total'] == 3
    assert len(data['results']) == 3
    for result in data['results']:
        assert result['sentiment'] in ['positive', 'neutral', 'negative']
        assert 0 < result['confidence'] <= 1.0


def test_predict_batch_empty_list():
    """Empty batch should return 422 validation error."""
    res = client.post('/predict/batch', json={'texts': []})
    assert res.status_code == 422


def test_history_endpoint():
    """History endpoint should return list of recent predictions."""
    res = client.get('/history')
    assert res.status_code == 200
    data = res.json()
    assert 'predictions' in data
    assert 'total' in data


def test_history_limit():
    """History endpoint should respect limit parameter."""
    res = client.get('/history?limit=2')
    assert res.status_code == 200
    data = res.json()
    assert data['total'] <= 2


def test_history_limit_exceeded():
    """History limit above 100 should return 400 error."""
    res = client.get('/history?limit=101')
    assert res.status_code == 400


def test_stats_endpoint():
    """Stats endpoint should return total predictions and distribution."""
    res = client.get('/stats')
    assert res.status_code == 200
    data = res.json()
    assert 'total_predictions' in data
    assert 'distribution' in data