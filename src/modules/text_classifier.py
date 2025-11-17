from typing import List
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TextRiskClassifier:
    def __init__(self, model_dir: str = "models/bert-suicide-model"):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"No se encontró el directorio del modelo BERT en {self.model_dir}. "
                "Descomprime allí el ZIP exportado desde Colab."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)

    def predict_proba(self, texts: List[str]) -> List[float]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        return [float(p[1]) for p in probs]
