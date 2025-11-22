import numpy as np
import torch
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification


class SentimentPredictor:
    """
    Loads fine-tuned FinBERT model and runs inference on financial text.
    Encapsulates all model loading and prediction logic in a single class.
    """

    LABEL2ID = {'negative': 0, 'neutral': 1, 'positive': 2}
    ID2LABEL = {0: 'negative', 1: 'neutral', 2: 'positive'}

    def __init__(self, model_dir: str, max_len: int = 64):
        self.model_dir = Path(model_dir).resolve().as_posix()
        self.max_len   = max_len
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.model     = BertForSequenceClassification.from_pretrained(self.model_dir, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """
        Run inference on a single financial text string.
        Returns sentiment label, confidence score, and all class probabilities.
        """
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.model(
                input_ids=enc['input_ids'].to(self.device),
                attention_mask=enc['attention_mask'].to(self.device),
                token_type_ids=enc['token_type_ids'].to(self.device)
            )
        probs   = torch.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()
        pred_id = int(np.argmax(probs))

        return {
            'text'         : text,
            'sentiment'    : self.ID2LABEL[pred_id],
            'confidence'   : round(float(probs[pred_id]), 4),
            'probabilities': {
                'negative': round(float(probs[0]), 4),
                'neutral'  : round(float(probs[1]), 4),
                'positive' : round(float(probs[2]), 4)
            }
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Run inference on a list of financial text strings.
        Returns a list of prediction dictionaries.
        """
        return [self.predict(t) for t in texts]