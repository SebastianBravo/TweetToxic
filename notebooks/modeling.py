"""Modelos de TweetToxic: pipelines de scikit-learn reproducibles y guardables.

Un modelo completo es un único objeto `TweetToxicClassifier` que contiene:

    texto crudo
      -> TweetPreprocessor(variante)        (preprocessing.py; sin etiquetas)
      -> ColumnTransformer de TF-IDF         (palabras 1-2, caracteres char_wb 2-5 y, en V4, lemas)
      -> MultiOutputClassifier(clasificador) (un clasificador por etiqueta: HS, TR, AG)
      -> umbrales por etiqueta + regla de jerarquía (si HS=0 entonces TR=0)

Todo lo que se ajusta (diccionario de restauración, vocabularios TF-IDF, clasificadores) se ajusta en
fit(), así que dentro de una validación cruzada se ajusta solo con el fold de entrenamiento.

Uso:
    model = TweetToxicClassifier(variant="V1", classifier="logreg").fit(train_texts, Y)
    save_model(model, "V1__logreg", metadata)          # models/V1__logreg.joblib + .json
    model = load_model("V1__logreg")                    # requiere notebooks/ en sys.path
    model.predict(test_texts)                           # array (n, 3) con HS, TR, AG
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted

import preprocessing as pp
from common import LABELS, ROOT_DIR, SEED, evaluate_oof

MODELS_DIR = ROOT_DIR / "models"
TOKEN_PATTERN = r"(?u)<\w+>|\b\w+\b"  # conserva <USER>, <URL> y palabras de una letra (t = tú, q = que)
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


# ---------------------------------------------------------------------------
# Piezas del pipeline
# ---------------------------------------------------------------------------
def make_features(variant: pp.Variant | str) -> ColumnTransformer:
    """TF-IDF fijo (decisión D12): palabras 1-2 + caracteres char_wb 2-5 (+ lemas 1-2 si la variante lematiza).

    lowercase=False: pasar o no a minúsculas lo decide el preprocesamiento, no el vectorizador.
    """
    variant = pp.VARIANTS[variant] if isinstance(variant, str) else variant
    blocks = [
        ("word", TfidfVectorizer(ngram_range=(1, 2), lowercase=False, token_pattern=TOKEN_PATTERN,
                                 sublinear_tf=True), "text"),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=False, sublinear_tf=True,
                                 min_df=2), "text"),
    ]
    if variant.lemma:
        blocks.append(("lemma", TfidfVectorizer(ngram_range=(1, 2), lowercase=False, token_pattern=TOKEN_PATTERN,
                                                sublinear_tf=True), "lemmas"))
    return ColumnTransformer(blocks)


def _logreg():
    return LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=2000, random_state=SEED)


def _linsvm():
    # LinearSVC no da probabilidades: se calibra con una sigmoide (validación interna de 3 folds).
    return CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=SEED), method="sigmoid", cv=3)


def _cnb():
    return ComplementNB()


# Clasificadores comparados, todos con sus hiperparámetros por defecto (no se ajustan aquí).
CLASSIFIERS = {"logreg": _logreg, "linsvm": _linsvm, "cnb": _cnb}
CLASSIFIER_NAMES = {"logreg": "Regresión logística (balanceada)", "linsvm": "SVM lineal calibrada (balanceada)",
                    "cnb": "Complement Naive Bayes"}


def build_pipeline(variant: pp.Variant | str, classifier: str) -> Pipeline:
    return Pipeline([
        ("prep", pp.TweetPreprocessor(variant)),
        ("features", make_features(variant)),
        ("clf", MultiOutputClassifier(CLASSIFIERS[classifier]())),
    ])


# ---------------------------------------------------------------------------
# Umbrales y jerarquía
# ---------------------------------------------------------------------------
def apply_decision(proba: np.ndarray, thresholds=0.5, hierarchy: bool = True) -> np.ndarray:
    """Probabilidades (n, 3) -> etiquetas 0/1 con umbral por etiqueta y, opcionalmente, la regla
    de jerarquía (decisión D7): si se predice HS=0, TR se fuerza a 0."""
    if isinstance(thresholds, dict):
        thresholds = [thresholds[lab] for lab in LABELS]
    thr = np.broadcast_to(np.asarray(thresholds, dtype=float), (len(LABELS),))
    pred = (np.asarray(proba) >= thr).astype(int)
    if hierarchy:
        pred[:, LABELS.index("TR")] *= pred[:, LABELS.index("HS")]
    return pred


def best_thresholds(y_true: np.ndarray, proba: np.ndarray, grid=THRESHOLD_GRID) -> dict:
    """Umbral que maximiza el F1 de cada etiqueta por separado."""
    out = {}
    for j, lab in enumerate(LABELS):
        f1s = [f1_score(y_true[:, j], (proba[:, j] >= t).astype(int), zero_division=0) for t in grid]
        out[lab] = float(grid[int(np.argmax(f1s))])
    return out


# ---------------------------------------------------------------------------
# Modelo completo
# ---------------------------------------------------------------------------
class TweetToxicClassifier(BaseEstimator, ClassifierMixin):
    """Modelo multietiqueta completo: texto crudo -> HS, TR, AG.

    - variant: variante de preprocesamiento (ver preprocessing.VARIANTS).
    - classifier: clave de CLASSIFIERS ('logreg', 'linsvm', 'cnb').
    - thresholds: umbral por etiqueta (dict) o un número; no afecta a fit().
    - hierarchy: aplicar la regla HS=0 -> TR=0 en predict().
    - empty_as_negative: los tuits sin texto real (p. ej. la celda '#¿NOMBRE?' del test) se
      predicen 000 en predict() (decisión D2).
    """

    def __init__(self, variant: str = "V1", classifier: str = "logreg", thresholds=0.5, hierarchy: bool = True,
                 empty_as_negative: bool = True):
        self.variant = variant
        self.classifier = classifier
        self.thresholds = thresholds
        self.hierarchy = hierarchy
        self.empty_as_negative = empty_as_negative

    def fit(self, X, y):
        y = np.asarray(y)
        self.pipeline_ = build_pipeline(self.variant, self.classifier).fit(X, y)
        self.labels_ = list(LABELS)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Probabilidad de la clase positiva de cada etiqueta: array (n, 3) en el orden HS, TR, AG."""
        check_is_fitted(self, "pipeline_")
        return np.column_stack([p[:, 1] for p in self.pipeline_.predict_proba(X)])

    def predict(self, X) -> np.ndarray:
        """Etiquetas 0/1 (n, 3) con los umbrales y la regla de jerarquía del modelo."""
        pred = apply_decision(self.predict_proba(X), self.thresholds, self.hierarchy)
        if self.empty_as_negative:
            empty = pp._as_text_series(X).map(lambda t: pp.fix_csv_artifacts(pp.remove_non_ascii(t)) == "")
            pred[empty.to_numpy()] = 0
        return pred


# ---------------------------------------------------------------------------
# Validación cruzada con los folds compartidos
# ---------------------------------------------------------------------------
def cross_validate(model: TweetToxicClassifier, texts: pd.Series, Y: np.ndarray, folds: pd.Series) -> dict:
    """CV con los folds de common.make_folds. Todo (preprocesamiento, TF-IDF, clasificador) se ajusta
    solo con el fold de entrenamiento.

    Devuelve:
    - oof: probabilidades out-of-fold (n, 3);
    - at_05: métricas por fold con umbral 0.5 y sin jerarquía;
    - tuned: métricas por fold con umbral elegido en las OOF de los OTROS folds y con jerarquía
      (estimación casi honesta del modelo final);
    - fit_seconds: tiempo total.
    """
    texts = pd.Series(texts).reset_index(drop=True)
    folds = pd.Series(folds).reset_index(drop=True)
    oof = np.zeros(Y.shape)
    start = time.perf_counter()
    for k in sorted(folds.unique()):
        trn, val = (folds != k).to_numpy(), (folds == k).to_numpy()
        fitted = sklearn.base.clone(model).fit(texts[trn], Y[trn])
        oof[val] = fitted.predict_proba(texts[val])
    seconds = time.perf_counter() - start

    at_05, tuned = [], []
    for k in sorted(folds.unique()):
        val = (folds == k).to_numpy()
        at_05.append(evaluate_oof(Y[val], oof[val], 0.5))
        thr = best_thresholds(Y[~val], oof[~val])  # umbral elegido sin mirar el fold k
        tuned.append(evaluate_oof(Y[val], apply_decision(oof[val], thr, hierarchy=model.hierarchy), 0.5))
    return {"oof": oof, "at_05": pd.DataFrame(at_05), "tuned": pd.DataFrame(tuned), "fit_seconds": seconds}


# ---------------------------------------------------------------------------
# Guardado y carga
# ---------------------------------------------------------------------------
def model_name(variant: str, classifier: str) -> str:
    return f"{pp.variant_slug(variant)}__{classifier}"


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def environment_info() -> dict:
    import spacy

    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "spacy": spacy.__version__, "spacy_model": f"{pp.SPACY_MODEL}",
            "seed": SEED, "git_commit": _git_commit()}


def save_model(model: TweetToxicClassifier, name: str, metadata: dict | None = None,
               models_dir: Path = MODELS_DIR) -> Path:
    """Guarda el modelo (joblib, comprimido) y sus metadatos (JSON legible) en models/."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.joblib"
    joblib.dump(model, path, compress=3)
    meta = {"name": name, "created": dt.datetime.now().isoformat(timespec="seconds"),
            "variant": model.variant, "variant_steps": pp.VARIANTS[model.variant].steps(),
            "classifier": model.classifier, "classifier_params": repr(CLASSIFIERS[model.classifier]()),
            "thresholds": model.thresholds, "hierarchy": model.hierarchy,
            "empty_as_negative": model.empty_as_negative, "labels": LABELS,
            "environment": environment_info(), **(metadata or {})}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    return path


def load_model(name_or_path, models_dir: Path = MODELS_DIR) -> TweetToxicClassifier:
    """Carga un modelo guardado. Necesita que notebooks/ esté en sys.path (por preprocessing/modeling)."""
    path = Path(name_or_path)
    if not path.suffix:
        path = Path(models_dir) / f"{name_or_path}.joblib"
    return joblib.load(path)
