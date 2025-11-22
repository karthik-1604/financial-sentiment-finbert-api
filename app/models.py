from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request body for single text sentiment prediction."""
    text: str = Field(
        min_length=1,
        max_length=512,
        description="Financial text to analyze for sentiment",
        examples=["The company reported record profits this quarter."]
    )


class BatchPredictRequest(BaseModel):
    """Request body for batch sentiment prediction."""
    texts: list[str] = Field(
        min_length=1,
        max_length=32,
        description="List of financial texts to analyze, maximum 32 at a time",
        examples=[["Revenue grew 20% year over year.", "The firm faces bankruptcy risk."]]
    )


class Probabilities(BaseModel):
    """Class probability scores for all three sentiment labels."""
    negative: float
    neutral:  float
    positive: float


class PredictResponse(BaseModel):
    """Response body for a single sentiment prediction."""
    text:          str
    sentiment:     str
    confidence:    float
    probabilities: Probabilities


class BatchPredictResponse(BaseModel):
    """Response body for batch sentiment prediction."""
    results: list[PredictResponse]
    total:   int


class PredictionRecord(BaseModel):
    """Single prediction record retrieved from the database."""
    id:            int
    text:          str
    sentiment:     str
    confidence:    float
    probabilities: Probabilities
    created_at:    str


class HistoryResponse(BaseModel):
    """Response body for prediction history endpoint."""
    predictions: list[PredictionRecord]
    total:       int


class StatsResponse(BaseModel):
    """Response body for prediction statistics endpoint."""
    total_predictions: int
    distribution:      dict[str, int]


class HealthResponse(BaseModel):
    """Response body for health check endpoint."""
    status:  str
    model:   str
    device:  str