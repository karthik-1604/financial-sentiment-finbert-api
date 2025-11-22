import sqlite3
import json
from datetime import datetime, timezone


DB_PATH = 'predictions.db'


def init_db():
    """
    Initialize SQLite database and create predictions table if it does not exist.
    Called once on application startup.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            text          TEXT NOT NULL,
            sentiment     TEXT NOT NULL,
            confidence    REAL NOT NULL,
            probabilities TEXT NOT NULL,
            created_at    TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def log_prediction(text: str, sentiment: str, confidence: float, probabilities: dict):
    """
    Insert a single prediction record into the database.
    Probabilities dict is stored as a JSON string.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (text, sentiment, confidence, probabilities, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        text,
        sentiment,
        confidence,
        json.dumps(probabilities),
        datetime.now(timezone.utc).isoformat()
    ))
    conn.commit()
    conn.close()


def get_recent_predictions(limit: int = 10) -> list[dict]:
    """
    Retrieve the most recent predictions from the database.
    Returns a list of prediction records ordered by newest first.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, text, sentiment, confidence, probabilities, created_at
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            'id'           : row[0],
            'text'         : row[1],
            'sentiment'    : row[2],
            'confidence'   : row[3],
            'probabilities': json.loads(row[4]),
            'created_at'   : row[5]
        }
        for row in rows
    ]


def get_prediction_stats() -> dict:
    """
    Return aggregate statistics over all logged predictions.
    Useful for monitoring sentiment distribution over time.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM predictions')
    total = cursor.fetchone()[0]

    cursor.execute('''
        SELECT sentiment, COUNT(*) as cnt
        FROM predictions
        GROUP BY sentiment
    ''')
    rows = cursor.fetchall()
    conn.close()

    distribution = {row[0]: row[1] for row in rows}
    return {
        'total_predictions': total,
        'distribution'     : distribution
    }