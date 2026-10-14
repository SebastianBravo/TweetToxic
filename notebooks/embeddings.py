"""Embeddings de palabras entrenados aquí (Word2Vec y FastText con gensim) para representar tuits.

Nada es preentrenado: los vectores se aprenden en fit() con los textos de entrenamiento (dentro de
cada fold en la CV). El tuit se representa como la media (simple o ponderada por IDF) de los vectores
de sus palabras.

Especificación en modeling.make_features: 'emb-<algoritmo>[-idf][-d<N>][-e<N>][-w<N>]'
    algoritmo: w2v (Word2Vec skip-gram), cbow (Word2Vec CBOW), ft (FastText skip-gram con subpalabras)
    -idf: media ponderada por IDF (en lugar de media simple)
    -d<N>, -e<N>, -w<N>: dimensión, épocas y ventana (por defecto 100, 30 y 5)

Los embeddings se entrenan SOLO con los textos de entrenamiento (del fold): ni test ni corpus externos.

Nota: con gensim compilado desde su rama de desarrollo y Python 3.14 aparece, de forma esporádica, el
aviso "Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'" sin ninguna excepción real
asociada. El entrenamiento es determinista y los vecinos más cercanos son coherentes.
"""

from __future__ import annotations

import re
import zlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import preprocessing as pp
from common import SEED

TOKEN_RX = re.compile(r"<\w+>|\b\w+\b")  # mismo patrón que el TF-IDF de palabras (conserva <USER>, <URL>, 't')
ALGORITHMS = {"w2v": ("word2vec", 1), "cbow": ("word2vec", 0), "ft": ("fasttext", 1)}


def stable_hash(text: str) -> int:
    """Hash determinista para inicializar los vectores (el hash de Python cambia entre ejecuciones)."""
    return zlib.crc32(text.encode("utf-8"))


def tokenize(text: str) -> list[str]:
    return TOKEN_RX.findall(text)


def parse_spec(spec: str) -> dict:
    """'emb-ft-idf-d50' -> {'algorithm': 'fasttext', 'sg': 1, 'pooling': 'idf', 'vector_size': 50}."""
    parts = spec.split("-")
    if parts[0] != "emb" or len(parts) < 2 or parts[1] not in ALGORITHMS:
        raise ValueError(f"Especificación de embedding desconocida: {spec}")
    algorithm, sg = ALGORITHMS[parts[1]]
    cfg = {"algorithm": algorithm, "sg": sg, "pooling": "mean"}
    numeric = {"d": "vector_size", "e": "epochs", "w": "window"}
    for part in parts[2:]:
        if part == "idf":
            cfg["pooling"] = "idf"
        elif part[:1] in numeric and part[1:].isdigit():
            cfg[numeric[part[0]]] = int(part[1:])
        else:
            raise ValueError(f"Especificación de embedding desconocida: {spec}")
    return cfg


class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """Entrena Word2Vec/FastText en fit() y representa cada texto como la media de sus vectores.

    - Entrada: DataFrame con la columna `source` (texto procesado o lemas).
    - algorithm: 'word2vec' o 'fasttext'; sg=1 skip-gram, sg=0 CBOW.
    - pooling: 'mean' (media simple) o 'idf' (media ponderada por el IDF de cada palabra en el train).
    - FastText usa n-gramas de caracteres de min_n a max_n y una tabla de `bucket` filas para ellos.
    """

    def __init__(self, algorithm="word2vec", sg=1, vector_size=100, window=5, min_count=2, epochs=30,
                 negative=5, pooling="mean", min_n=3, max_n=5, bucket=20000, variant="V1", source="text"):
        self.algorithm = algorithm
        self.sg = sg
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.negative = negative
        self.pooling = pooling
        self.min_n = min_n
        self.max_n = max_n
        self.bucket = bucket
        self.variant = variant
        self.source = source

    def _column(self, X) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            return X[self.source].astype(str).reset_index(drop=True)
        return pp._as_text_series(X)

    def fit(self, X, y=None):
        from gensim.models import FastText, Word2Vec

        sentences = [tokenize(t) for t in self._column(X)]
        n_train = len(sentences)
        common = dict(vector_size=self.vector_size, window=self.window, min_count=self.min_count, sg=self.sg,
                      negative=self.negative, epochs=self.epochs, workers=1, seed=SEED, hashfxn=stable_hash)
        if self.algorithm == "fasttext":
            model = FastText(sentences, min_n=self.min_n, max_n=self.max_n, bucket=self.bucket, **common)
        else:
            model = Word2Vec(sentences, **common)
        self.wv_ = model.wv
        # IDF calculado con los mismos textos de entrenamiento
        df = pd.Series([w for s in sentences[:n_train] for w in set(s)]).value_counts()
        self.idf_ = (np.log((1 + n_train) / (1 + df)) + 1).to_dict()
        self.default_idf_ = float(np.log(1 + n_train) + 1)
        self.n_features_out_ = self.vector_size
        return self

    def _vector(self, tokens: list[str]) -> np.ndarray:
        fasttext = self.algorithm == "fasttext"
        words = [w for w in tokens if fasttext or w in self.wv_.key_to_index]
        if not words:
            return np.zeros(self.vector_size, dtype=np.float32)
        vectors = np.vstack([self.wv_[w] for w in words])
        if self.pooling == "idf":
            weights = np.array([self.idf_.get(w, self.default_idf_) for w in words])
            return (weights[:, None] * vectors).sum(axis=0) / weights.sum()
        return vectors.mean(axis=0)

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, "wv_")
        return np.vstack([self._vector(tokenize(t)) for t in self._column(X)]).astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"emb{i}" for i in range(self.vector_size)], dtype=object)

    def most_similar(self, word: str, topn: int = 8) -> list[tuple[str, float]]:
        """Vecinos más cercanos (para inspección)."""
        check_is_fitted(self, "wv_")
        if self.algorithm != "fasttext" and word not in self.wv_.key_to_index:
            return []
        return self.wv_.most_similar(word, topn=topn)
