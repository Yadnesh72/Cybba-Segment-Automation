from __future__ import annotations

from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _tfidf_params_from_config(cfg: dict) -> Dict[str, Any]:
    tfidf_cfg = cfg.get("model", {}).get("tfidf", {}) or {}
    ngram_min = int(tfidf_cfg.get("ngram_min", 1))
    ngram_max = int(tfidf_cfg.get("ngram_max", 2))
    min_df = int(tfidf_cfg.get("min_df", 2))
    stop_words = tfidf_cfg.get("stop_words", "english")

    max_features = tfidf_cfg.get("max_features", 50000)
    try:
        max_features = int(max_features) if max_features is not None else None
    except Exception:
        max_features = 50000

    return dict(
        lowercase=True,
        stop_words=stop_words,
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        max_features=max_features,
        dtype=np.float32,
    )


class L2Retriever:
    def __init__(self, vectorizer: TfidfVectorizer, X_matrix, l2_labels: np.ndarray):
        self.vectorizer = vectorizer
        self.X_matrix = X_matrix
        self.l2_labels = l2_labels

    @classmethod
    def fit_from_text(cls, cfg: dict, texts: List[str], l2: List[str]) -> "L2Retriever":
        tfidf_params = _tfidf_params_from_config(cfg)
        vec = TfidfVectorizer(**tfidf_params)
        X = vec.fit_transform(texts)
        return cls(vec, X, np.array(l2, dtype=object))

    def predict(self, texts: List[str]) -> List[str]:
        Q = self.vectorizer.transform(texts)
        sims = Q @ self.X_matrix.T
        best_idx = np.asarray(sims.argmax(axis=1)).ravel()
        return [str(self.l2_labels[i]) for i in best_idx]