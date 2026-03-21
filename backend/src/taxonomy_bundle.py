from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from sklearn.pipeline import Pipeline

@dataclass
class TaxonomyL1Bundle:
    model: Pipeline
    vertical_model: Optional[Pipeline] = None
    b2b_label: str = "B2B Audience"
    vertical_min_confidence: float = 0.55
    vertical_margin: float = 0.08

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("Underlying L1 model does not support predict_proba()")

    def decision_function(self, X):
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
        raise AttributeError("Underlying L1 model does not support decision_function()")

    @property
    def classes_(self):
        return getattr(self.model, "classes_", None)