"""Modelos de TweetToxic: pipelines de scikit-learn reproducibles y guardables.

Un modelo completo es un único objeto `TweetToxicClassifier` que contiene:

    texto crudo
      -> TweetPreprocessor(variante)        (preprocessing.py; sin etiquetas)
      -> representación (FEATURES)           TF-IDF de palabras/caracteres/lemas, SVD, rasgos hechos a mano
      -> MultiOutputClassifier(clasificador) (un clasificador por etiqueta: HS, TR, AG; con selección chi2
                                              opcional, calculada por etiqueta)
      -> umbrales por etiqueta + regla de jerarquía (si HS=0 entonces TR=0)

Todo lo que se ajusta (diccionario de restauración, vocabularios TF-IDF, SVD, chi2, razones de NB-SVM,
clasificadores) se ajusta en fit(), así que dentro de una validación cruzada se ajusta solo con el fold
de entrenamiento.

Uso:
    model = TweetToxicClassifier(variant="V1", classifier="logreg").fit(train_texts, Y)
    save_model(model, "V1__logreg", metadata)          # models/V1__logreg.joblib + .json
    model = load_model("V1__logreg")                    # requiere notebooks/ en sys.path
    model.predict(test_texts)                           # array (n, 3) con HS, TR, AG
    make_submission("V3__linsvm")                       # submissions/V3__linsvm.csv
"""

from __future__ import annotations

import datetime as dt
import io
import json
import platform
import subprocess
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.validation import check_is_fitted

import embeddings as emb
import handcrafted as hf
import preprocessing as pp
from common import DATA_DIR, LABELS, ROOT_DIR, SEED, _near_duplicate_key, evaluate_oof, load_test

MODELS_DIR = ROOT_DIR / "models"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"
REGISTRY_PATH = MODELS_DIR / "registry.csv"
OOF_PATH = MODELS_DIR / "oof_predictions.parquet"
TEST_PRED_PATH = MODELS_DIR / "test_predictions.parquet"
TOKEN_PATTERN = r"(?u)<\w+>|\b\w+\b"  # conserva <USER>, <URL> y palabras de una letra (t = tú, q = que)
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
ESTIMATOR_N_JOBS = 2  # hilos por estimador de árboles/boosting (los folds se paralelizan aparte)
SVD_COMPONENTS = 300


# ---------------------------------------------------------------------------
# Representación (features)
# ---------------------------------------------------------------------------
def _resolve(variant: pp.Variant | str) -> pp.Variant:
    return pp.VARIANTS[variant] if isinstance(variant, str) else variant


def _text_blocks(variant: pp.Variant, use_char: bool) -> list:
    blocks = [("word", TfidfVectorizer(ngram_range=(1, 2), lowercase=False, token_pattern=TOKEN_PATTERN,
                                       sublinear_tf=True), "text")]
    if use_char:
        blocks.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=False,
                                               sublinear_tf=True, min_df=2), "text"))
    if variant.lemma:
        blocks.append(("lemma", TfidfVectorizer(ngram_range=(1, 2), lowercase=False, token_pattern=TOKEN_PATTERN,
                                                sublinear_tf=True), "lemmas"))
    return blocks


def _parse_features(features: str) -> tuple[str | None, bool, bool, list[str]]:
    parts = features.split("+")
    base = parts[0] if parts[0] in {"tfidf", "word"} else None
    embs = [p for p in parts if p.startswith("emb-")]
    extras = set(parts[1 if base else 0:]) - {"svd", "hand"} - set(embs)
    if (base is None and not embs) or extras:
        raise ValueError(f"Representación desconocida: {features}")
    for spec in embs:
        emb.parse_spec(spec)
    return base, "svd" in parts, "hand" in parts, embs


def make_features(variant: pp.Variant | str, features: str = "tfidf"):
    """Representación del texto. `features` es una especificación con '+':

    - base: 'tfidf' = palabras 1-2 + caracteres char_wb 2-5 (decisión D12); 'word' = solo palabras 1-2.
      En V4 se añaden siempre los lemas (1-2).
    - '+svd': reduce la base a SVD_COMPONENTS dimensiones densas (TruncatedSVD, sin etiquetas).
    - '+hand': concatena los rasgos hechos a mano (handcrafted.py) calculados sobre el texto crudo,
      escalados a [0, 1] con MaxAbsScaler.
    - 'emb-<alg>[-idf][-d<N>][-e<N>]': media de embeddings entrenados aquí (embeddings.py), estandarizada.
      Puede ser la única representación ('emb-ft') o sumarse a otra ('tfidf+emb-ft'). En V4 los
      embeddings se entrenan sobre los lemas.

    lowercase=False: pasar o no a minúsculas lo decide el preprocesamiento, no el vectorizador.
    """
    variant = _resolve(variant)
    base, use_svd, use_hand, embs = _parse_features(features)
    if base and not use_svd and not use_hand and not embs:
        return ColumnTransformer(_text_blocks(variant, use_char=base == "tfidf"))  # misma estructura que en el 03
    blocks = []
    if base:
        text_ct = ColumnTransformer(_text_blocks(variant, use_char=base == "tfidf"))
        text_cols = ["text"] + (["lemmas"] if variant.lemma else [])
        text_part = (Pipeline([("tfidf", text_ct), ("svd", TruncatedSVD(SVD_COMPONENTS, random_state=SEED))])
                     if use_svd else text_ct)
        blocks.append(("text", text_part, text_cols))
    source = "lemmas" if variant.lemma else "text"
    for spec in embs:
        cfg = emb.parse_spec(spec)
        vec = emb.EmbeddingVectorizer(variant=variant.name, source=source, **cfg)
        blocks.append((spec, Pipeline([("vectors", vec), ("scale", StandardScaler())]), [source]))
    if use_hand:
        blocks.append(("hand", Pipeline([("rules", hf.HandcraftedFeatures()), ("scale", MaxAbsScaler())]), "raw"))
    return ColumnTransformer(blocks)


# ---------------------------------------------------------------------------
# NB-SVM
# ---------------------------------------------------------------------------
class NBSVMClassifier(BaseEstimator, ClassifierMixin):
    """NB-SVM (Wang y Manning, 2012) en su versión con regresión logística.

    En fit() calcula, solo con los datos de entrenamiento, la razón log-count de Naive Bayes de cada
    rasgo, r = log((p/|p|) / (q/|q|)), con p y q las sumas (más alpha) del rasgo en la clase positiva y
    en la negativa. Después entrena una regresión logística sobre X escalada por r.
    """

    def __init__(self, C: float = 1.0, alpha: float = 1.0):
        self.C = C
        self.alpha = alpha

    def _scale(self, X):
        return sp.csr_matrix(sp.csr_matrix(X).multiply(self.r_))

    def fit(self, X, y):
        X = sp.csr_matrix(X)
        y = np.asarray(y)
        p = self.alpha + np.asarray(X[y == 1].sum(axis=0)).ravel()
        q = self.alpha + np.asarray(X[y == 0].sum(axis=0)).ravel()
        self.r_ = np.log((p / p.sum()) / (q / q.sum()))
        self.lr_ = LogisticRegression(C=self.C, class_weight="balanced", solver="liblinear", max_iter=2000,
                                      random_state=SEED).fit(self._scale(X), y)
        self.classes_ = self.lr_.classes_
        return self

    def predict_proba(self, X):
        return self.lr_.predict_proba(self._scale(X))

    def predict(self, X):
        return self.lr_.predict(self._scale(X))


# ---------------------------------------------------------------------------
# Catálogo de clasificadores
# ---------------------------------------------------------------------------
def _logreg(C=1.0, l1_ratio=None):
    """l1_ratio=None: L2 con liblinear (la del 03). l1_ratio en [0, 1]: saga (1 = L1, 0.5 = elasticnet)."""
    if l1_ratio is None:
        return LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=SEED)
    return LogisticRegression(C=C, l1_ratio=l1_ratio, class_weight="balanced", solver="saga", max_iter=3000,
                              tol=1e-3, random_state=SEED)


def _linsvm(C=1.0):
    # LinearSVC no da probabilidades: se calibra con una sigmoide (validación interna de 3 folds).
    return CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced", random_state=SEED), method="sigmoid", cv=3)


def _rbfsvm(C=1.0, gamma="scale"):
    return CalibratedClassifierCV(SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced"), method="sigmoid", cv=3)


def _sgd(loss="modified_huber", alpha=1e-4):
    return SGDClassifier(loss=loss, alpha=alpha, class_weight="balanced", max_iter=2000, tol=1e-4, random_state=SEED)


def _rf(n_estimators=300, max_features="sqrt", min_samples_leaf=1):
    return RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                  min_samples_leaf=min_samples_leaf, class_weight="balanced_subsample",
                                  n_jobs=ESTIMATOR_N_JOBS, random_state=SEED)


def _et(n_estimators=300, max_features="sqrt", min_samples_leaf=1):
    return ExtraTreesClassifier(n_estimators=n_estimators, max_features=max_features,
                                min_samples_leaf=min_samples_leaf, class_weight="balanced_subsample",
                                n_jobs=ESTIMATOR_N_JOBS, random_state=SEED)


def _xgb(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8):
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                         subsample=subsample, colsample_bytree=colsample_bytree, tree_method="hist",
                         eval_metric="logloss", n_jobs=ESTIMATOR_N_JOBS, random_state=SEED)


def _lgbm(n_estimators=300, num_leaves=15, learning_rate=0.05, min_child_samples=10, colsample_bytree=0.8):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate,
                          min_child_samples=min_child_samples, colsample_bytree=colsample_bytree,
                          subsample=0.8, subsample_freq=1, class_weight="balanced", n_jobs=ESTIMATOR_N_JOBS,
                          random_state=SEED, deterministic=True, force_row_wise=True, verbose=-1)


def _knn(n_neighbors=15):
    return KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine", algorithm="brute", weights="distance")


CLASSIFIERS = {
    "logreg": _logreg, "sgd": _sgd, "linsvm": _linsvm, "nbsvm": NBSVMClassifier,
    "mnb": MultinomialNB, "bnb": BernoulliNB, "cnb": ComplementNB, "knn": _knn,
    "rbfsvm": _rbfsvm, "rf": _rf, "et": _et, "xgb": _xgb, "lgbm": _lgbm,
}
CLASSIFIER_NAMES = {
    "logreg": "Regresión logística (balanceada)", "sgd": "SGDClassifier", "linsvm": "SVM lineal calibrada (balanceada)",
    "nbsvm": "NB-SVM", "mnb": "Multinomial NB", "bnb": "Bernoulli NB", "cnb": "Complement Naive Bayes",
    "knn": "kNN coseno", "rbfsvm": "SVM RBF calibrada", "rf": "Random Forest", "et": "ExtraTrees",
    "xgb": "XGBoost", "lgbm": "LightGBM",
}
FAMILIES = {"logreg": "lineal", "sgd": "lineal", "linsvm": "lineal", "nbsvm": "lineal", "mnb": "bayes",
            "bnb": "bayes", "cnb": "bayes", "knn": "vecinos", "rbfsvm": "kernel", "rf": "árboles", "et": "árboles",
            "xgb": "boosting", "lgbm": "boosting"}
# Probabilidades calibradas explícitamente (CalibratedClassifierCV con sigmoide). El resto son puntuaciones
# en [0, 1] no calibradas; la regresión logística y NB-SVM están razonablemente calibradas por construcción,
# pero class_weight="balanced" las desplaza hacia la clase positiva.
CALIBRATED = {"linsvm", "rbfsvm"}
# Redes neuronales (neural.py): misma API; familia "neuronal".
NEURAL_ARCHS = {"nn_mlp": "MLP sobre TF-IDF", "nn_fasttext": "Estilo fastText (palabras + n-gramas hasheados)",
                "nn_cnn_word": "CNN de palabras", "nn_cnn_char": "CNN de caracteres", "nn_bigru": "BiGRU con atención"}
FAMILIES.update({k: "neuronal" for k in NEURAL_ARCHS})
CLASSIFIER_NAMES.update(NEURAL_ARCHS)


def make_classifier(classifier: str, clf_params: dict | None = None):
    """Clasificador de una etiqueta. clf_params puede incluir 'select_k': selección chi2 de los k mejores
    rasgos para esa etiqueta, calculada en fit() (dentro del fold)."""
    params = dict(clf_params or {})
    select_k = params.pop("select_k", None)
    clf = CLASSIFIERS[classifier](**params)
    if select_k:
        return Pipeline([("chi2", SelectKBest(chi2, k=select_k)), ("model", clf)])
    return clf


def build_featurizer(variant: pp.Variant | str, features: str = "tfidf") -> Pipeline:
    _, _, use_hand, _ = _parse_features(features)
    keep_raw = use_hand
    return Pipeline([("prep", pp.TweetPreprocessor(variant, keep_raw=keep_raw)),
                     ("features", make_features(variant, features))])


def build_pipeline(variant: pp.Variant | str, classifier: str, features: str = "tfidf",
                   clf_params: dict | None = None) -> Pipeline:
    featurizer = build_featurizer(variant, features)
    return Pipeline(featurizer.steps + [("clf", MultiOutputClassifier(make_classifier(classifier, clf_params)))])


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


def tuned_macro(y_true: np.ndarray, proba: np.ndarray, hierarchy: bool = True) -> float:
    """F1 macro con el umbral óptimo de cada etiqueta sobre esas mismas predicciones (para comparar
    configuraciones dentro de la CV interna; es optimista, pero igual para todas)."""
    return evaluate_oof(y_true, apply_decision(proba, best_thresholds(y_true, proba), hierarchy), 0.5)["F1_macro"]


def empty_mask(X) -> np.ndarray:
    """True para los tuits sin texto real (p. ej. la celda '#¿NOMBRE?' del test)."""
    return pp._as_text_series(X).map(lambda t: pp.fix_csv_artifacts(pp.remove_non_ascii(t)) == "").to_numpy()


def decide(proba: np.ndarray, X, thresholds=0.5, hierarchy: bool = True, empty_as_negative: bool = True) -> np.ndarray:
    """Posprocesamiento común a todos los modelos (D13, D17, D2): umbral por etiqueta, jerarquía
    HS=0 -> TR=0 y tuits vacíos -> 000."""
    pred = apply_decision(proba, thresholds, hierarchy)
    if empty_as_negative:
        pred[empty_mask(X)] = 0
    return pred


class DecisionMixin:
    """Da a cualquier modelo con predict_proba(X) -> (n, 3) el mismo predict() que TweetToxicClassifier.
    Requiere los atributos thresholds, hierarchy y empty_as_negative."""

    def predict(self, X) -> np.ndarray:
        """Etiquetas 0/1 (n, 3) con los umbrales, la jerarquía y la regla de tuits vacíos del modelo."""
        return decide(self.predict_proba(X), X, self.thresholds, self.hierarchy, self.empty_as_negative)


# ---------------------------------------------------------------------------
# Modelo completo
# ---------------------------------------------------------------------------
class TweetToxicClassifier(DecisionMixin, BaseEstimator, ClassifierMixin):
    """Modelo multietiqueta completo: texto crudo -> HS, TR, AG.

    - variant: variante de preprocesamiento (ver preprocessing.VARIANTS).
    - classifier: clave de CLASSIFIERS.
    - features: representación (ver make_features): 'tfidf', 'word', '+svd', '+hand'.
    - clf_params: hiperparámetros del clasificador (incluido 'select_k' para chi2).
    - thresholds: umbral por etiqueta (dict) o un número; no afecta a fit().
    - hierarchy: aplicar la regla HS=0 -> TR=0 en predict().
    - empty_as_negative: los tuits sin texto real (p. ej. la celda '#¿NOMBRE?' del test) se
      predicen 000 en predict() (decisión D2).
    """

    def __init__(self, variant: str = "V1", classifier: str = "logreg", thresholds=0.5, hierarchy: bool = True,
                 empty_as_negative: bool = True, features: str = "tfidf", clf_params: dict | None = None):
        self.variant = variant
        self.classifier = classifier
        self.thresholds = thresholds
        self.hierarchy = hierarchy
        self.empty_as_negative = empty_as_negative
        self.features = features
        self.clf_params = clf_params

    def fit(self, X, y):
        y = np.asarray(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.pipeline_ = build_pipeline(self.variant, self.classifier, getattr(self, "features", "tfidf"),
                                            getattr(self, "clf_params", None)).fit(X, y)
        self.labels_ = list(LABELS)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Probabilidad (o puntuación en [0, 1]) de la clase positiva de cada etiqueta: (n, 3) en el orden HS, TR, AG."""
        check_is_fitted(self, "pipeline_")
        return np.column_stack([p[:, 1] for p in self.pipeline_.predict_proba(X)])


# ---------------------------------------------------------------------------
# Validación cruzada con los folds compartidos
# ---------------------------------------------------------------------------
def _fit_predict_fold(model, texts, Y, trn, val, return_train=False):
    fitted = clone(model).fit(texts[trn], Y[trn])
    if return_train:
        return fitted.predict_proba(texts[val]), fitted.predict_proba(texts[trn])
    return fitted.predict_proba(texts[val])


def fold_thresholds(oof: np.ndarray, Y: np.ndarray, folds: pd.Series, k) -> dict:
    """Umbral de evaluación del fold k: el óptimo sobre las OOF de los OTROS folds (D13)."""
    val = (pd.Series(folds).reset_index(drop=True) == k).to_numpy()
    return best_thresholds(Y[~val], oof[~val])


def decision_metrics(y_true: np.ndarray, proba: np.ndarray, thresholds, hierarchy: bool = True) -> dict:
    """F1 por etiqueta, macro y micro de unas probabilidades con un umbral dado y la jerarquía."""
    return evaluate_oof(y_true, apply_decision(proba, thresholds, hierarchy), 0.5)


def _summarize_folds(oof: np.ndarray, Y: np.ndarray, folds: pd.Series, hierarchy: bool = True) -> tuple:
    at_05, tuned = [], []
    for k in sorted(folds.unique()):
        val = (folds == k).to_numpy()
        at_05.append(evaluate_oof(Y[val], oof[val], 0.5))
        thr = best_thresholds(Y[~val], oof[~val])  # umbral elegido sin mirar el fold k
        tuned.append(evaluate_oof(Y[val], apply_decision(oof[val], thr, hierarchy=hierarchy), 0.5))
    return pd.DataFrame(at_05), pd.DataFrame(tuned)


def cross_validate(model: TweetToxicClassifier, texts: pd.Series, Y: np.ndarray, folds: pd.Series,
                   n_jobs: int = 1, return_train: bool = False) -> dict:
    """CV con los folds de common.make_folds. Todo (preprocesamiento, representación, clasificador) se
    ajusta solo con el fold de entrenamiento. n_jobs > 1 reparte los folds entre procesos.

    Devuelve:
    - oof: probabilidades out-of-fold (n, 3);
    - at_05: métricas por fold con umbral 0.5 y sin jerarquía;
    - tuned: métricas por fold con umbral elegido en las OOF de los OTROS folds y con jerarquía
      (estimación casi honesta del modelo final);
    - fit_seconds: tiempo total (de reloj);
    - train (si return_train): métricas por fold sobre los propios datos de entrenamiento del fold
      (dentro de la muestra), con el mismo umbral y jerarquía que el fold de validación. Sirve para
      medir el sobreajuste (brecha train − validación).
    """
    texts = pd.Series(texts).reset_index(drop=True)
    folds = pd.Series(folds).reset_index(drop=True)
    oof = np.zeros(Y.shape)
    ks = sorted(folds.unique())
    start = time.perf_counter()
    out = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict_fold)(model, texts, Y, (folds != k).to_numpy(), (folds == k).to_numpy(), return_train)
        for k in ks)
    for k, p in zip(ks, out):
        oof[(folds == k).to_numpy()] = p[0] if return_train else p
    seconds = time.perf_counter() - start
    at_05, tuned = _summarize_folds(oof, Y, folds, model.hierarchy)
    result = {"oof": oof, "at_05": at_05, "tuned": tuned, "fit_seconds": seconds}
    if return_train:
        result["train"] = pd.DataFrame([decision_metrics(Y[(folds != k).to_numpy()], p[1], fold_thresholds(oof, Y, folds, k),
                                                         model.hierarchy) for k, p in zip(ks, out)])
    return result


def _inner_splits(texts: pd.Series, Y: np.ndarray, n_splits: int = 3):
    combo = pd.Series(["".join(map(str, row)) for row in Y])
    groups = texts.map(_near_duplicate_key)
    return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(texts, combo, groups))


def select_params(variant, classifier, features, grid: list[dict], texts: pd.Series, Y: np.ndarray,
                  n_splits: int = 3) -> tuple[dict, pd.DataFrame]:
    """Elige hiperparámetros con una CV de n_splits folds sobre (texts, Y).

    La representación (preprocesamiento + TF-IDF/SVD/rasgos) se ajusta una vez por fold interno y se
    reutiliza para todas las configuraciones del clasificador. Criterio: F1 macro con umbral óptimo y
    jerarquía sobre las OOF internas (igual de optimista para todas las configuraciones).
    """
    texts = pd.Series(texts).reset_index(drop=True)
    inner_oof = {i: np.zeros(Y.shape) for i in range(len(grid))}
    for trn, val in _inner_splits(texts, Y, n_splits):
        featurizer = build_featurizer(variant, features).fit(texts[trn], Y[trn])
        X_trn, X_val = featurizer.transform(texts[trn]), featurizer.transform(texts[val])
        for i, params in enumerate(grid):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf = MultiOutputClassifier(make_classifier(classifier, params)).fit(X_trn, Y[trn])
            inner_oof[i][val] = np.column_stack([p[:, 1] for p in clf.predict_proba(X_val)])
    scores = pd.DataFrame({"params": [json.dumps(g, sort_keys=True) for g in grid],
                           "macro_interno": [tuned_macro(Y, inner_oof[i]) for i in range(len(grid))]})
    best = int(scores["macro_interno"].idxmax())
    return grid[best], scores


def _nested_fold(variant, classifier, features, grid, texts, Y, trn, val, inner_splits):
    best, _scores = select_params(variant, classifier, features, grid, texts[trn].reset_index(drop=True), Y[trn],
                                 inner_splits)
    model = TweetToxicClassifier(variant=variant, classifier=classifier, features=features, clf_params=best)
    return model.fit(texts[trn], Y[trn]).predict_proba(texts[val]), best


def nested_cross_validate(variant, classifier, features, grid: list[dict], texts: pd.Series, Y: np.ndarray,
                          folds: pd.Series, inner_splits: int = 3, n_jobs: int = 1) -> dict:
    """CV anidada: en cada fold externo, una CV interna de `inner_splits` folds (solo con los datos de
    entrenamiento de ese fold) elige los hiperparámetros; el fold externo solo mide.

    Devuelve lo mismo que cross_validate más `params_per_fold` (la configuración elegida en cada fold).
    """
    texts = pd.Series(texts).reset_index(drop=True)
    folds = pd.Series(folds).reset_index(drop=True)
    ks = sorted(folds.unique())
    start = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(
        delayed(_nested_fold)(variant, classifier, features, grid, texts, Y, (folds != k).to_numpy(),
                              (folds == k).to_numpy(), inner_splits) for k in ks)
    oof = np.zeros(Y.shape)
    for k, (p, _) in zip(ks, results):
        oof[(folds == k).to_numpy()] = p
    seconds = time.perf_counter() - start
    at_05, tuned = _summarize_folds(oof, Y, folds, hierarchy=True)
    return {"oof": oof, "at_05": at_05, "tuned": tuned, "fit_seconds": seconds,
            "params_per_fold": [best for _, best in results]}


# ---------------------------------------------------------------------------
# Guardado, registro y predicciones
# ---------------------------------------------------------------------------
def model_name(variant: str, classifier: str, features: str = "tfidf", tag: str | None = None) -> str:
    """'V3__linsvm' (como en el 03); con otra representación o etiqueta: 'V3__rf__word+svd', 'V3__linsvm__tfidf+hand'."""
    name = f"{pp.variant_slug(variant)}__{classifier}"
    if features != "tfidf":
        name += f"__{features}"
    if tag:
        name += f"__{tag}"
    return name


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def environment_info() -> dict:
    import spacy

    info = {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "spacy": spacy.__version__, "spacy_model": f"{pp.SPACY_MODEL}",
            "seed": SEED, "git_commit": _git_commit()}
    for lib in ("xgboost", "lightgbm"):
        try:
            info[lib] = __import__(lib).__version__
        except Exception:
            pass
    return info


def model_size_mb(model) -> float:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    return buffer.tell() / 1e6


# Cada notebook guarda sus modelos en su propia carpeta dentro de models/. El registro y los parquet de
# predicciones, compartidos por todos, quedan en la raíz de models/.
MODEL_FOLDERS = {"03": "03_modelos", "04": "04_modelos_clasicos", "05": "05_redes_neuronales"}


def model_folder(notebook: str | None, models_dir: Path = MODELS_DIR) -> Path:
    """models/<carpeta del notebook>/ (o models/ si no se indica notebook)."""
    folder = Path(models_dir) / MODEL_FOLDERS.get(str(notebook), str(notebook)) if notebook else Path(models_dir)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_model(model: TweetToxicClassifier, name: str, metadata: dict | None = None,
               models_dir: Path = MODELS_DIR, notebook: str | None = None) -> Path:
    """Guarda el modelo (joblib, comprimido) y sus metadatos (JSON legible) en models/<carpeta del notebook>/.
    El notebook se toma del argumento o, si falta, de metadata['notebook']."""
    notebook = notebook or (metadata or {}).get("notebook")
    path = model_folder(notebook, models_dir) / f"{name}.joblib"
    joblib.dump(model, path, compress=3)
    features = getattr(model, "features", "tfidf")
    meta = {"name": name, "created": dt.datetime.now().isoformat(timespec="seconds"),
            "variant": model.variant, "variant_steps": pp.VARIANTS[model.variant].steps(),
            "features": features, "classifier": model.classifier, "family": FAMILIES[model.classifier],
            "clf_params": getattr(model, "clf_params", None),
            "classifier_repr": repr(make_classifier(model.classifier, getattr(model, "clf_params", None))),
            "proba_calibrated": model.classifier in CALIBRATED,
            "thresholds": model.thresholds, "hierarchy": model.hierarchy,
            "empty_as_negative": model.empty_as_negative, "labels": LABELS,
            "environment": environment_info(), **(metadata or {})}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    return path


def find_model_path(model_id: str, models_dir: Path = MODELS_DIR) -> Path | None:
    """Busca models/**/<model_id>.joblib o .pt (en cualquier carpeta de notebook)."""
    for ext in (".joblib", ".pt"):
        found = sorted(Path(models_dir).rglob(f"{model_id}{ext}"))
        if found:
            return found[0]
    return None


def load_model(name_or_path, models_dir: Path = MODELS_DIR):
    """Carga un modelo guardado, según su extensión: .joblib (scikit-learn) o .pt (redes de neural.py).
    Con un model_id sin extensión lo busca en todas las carpetas de models/.
    Necesita que notebooks/ esté en sys.path (por preprocessing/modeling/handcrafted/neural)."""
    path = Path(name_or_path)
    if path.suffix not in {".joblib", ".pt"}:
        path = find_model_path(str(name_or_path), models_dir)
        if path is None:
            raise FileNotFoundError(f"No hay ningún models/**/{name_or_path}.joblib ni .pt")
    if path.suffix == ".pt":
        import neural

        return neural.load_neural(path)
    return joblib.load(path)


REGISTRY_COLUMNS = ["model_id", "family", "classifier", "variant", "features", "params", "notebook",
                    "F1_macro", "F1_macro_std", "F1_micro", "F1_HS", "F1_TR", "F1_AG",
                    "thr_HS", "thr_TR", "thr_AG", "proba_calibrated", "cv_seconds", "size_mb", "saved",
                    "architecture", "seeds", "epochs", "train_seconds", "path"]
REGISTRY_KEYS = ["model_id", "family", "F1_macro"]  # columnas mínimas para aceptar un registro existente


def registry_row(model_id: str, model, cv: dict, notebook: str, saved: bool,
                 size_mb: float | None = None, extra: dict | None = None) -> dict:
    t = cv["tuned"]
    thr = model.thresholds if isinstance(model.thresholds, dict) else {lab: model.thresholds for lab in LABELS}
    return {"model_id": model_id, "family": FAMILIES[model.classifier], "classifier": model.classifier,
            "variant": model.variant, "features": getattr(model, "features", "tfidf"),
            "params": json.dumps(getattr(model, "clf_params", None) or {}, sort_keys=True), "notebook": notebook,
            "F1_macro": round(t["F1_macro"].mean(), 4), "F1_macro_std": round(t["F1_macro"].std(), 4),
            "F1_micro": round(t["F1_micro"].mean(), 4),
            **{f"F1_{lab}": round(t[f"F1_{lab}"].mean(), 4) for lab in LABELS},
            **{f"thr_{lab}": thr[lab] for lab in LABELS},
            "proba_calibrated": model.classifier in CALIBRATED, "cv_seconds": round(cv["fit_seconds"], 1),
            "size_mb": None if size_mb is None else round(size_mb, 2), "saved": saved,
            "architecture": None, "seeds": None, "epochs": None, "train_seconds": None, "path": None,
            **(extra or {})}


def _upsert(path: Path, new: pd.DataFrame, key: str, required: list[str]) -> pd.DataFrame:
    if path.exists():
        old = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype={"notebook": str})
        if "notebook" in old.columns:  # '03' se guarda como texto; versiones antiguas lo leían como 3
            old["notebook"] = old["notebook"].astype(str).str.zfill(2)
        if set(required) <= set(old.columns):  # descarta archivos con el formato anterior
            new = pd.concat([old[~old[key].isin(new[key].unique())], new], ignore_index=True)
    return new


def update_registry(rows: list[dict], path: Path = REGISTRY_PATH) -> pd.DataFrame:
    """Añade o reemplaza filas (por model_id) en models/registry.csv."""
    reg = _upsert(path, pd.DataFrame(rows).reindex(columns=REGISTRY_COLUMNS), "model_id", REGISTRY_KEYS)
    reg = reg.reindex(columns=REGISTRY_COLUMNS)
    # ruta del archivo guardado (relativa a models/), buscada en disco para todas las filas
    reg["path"] = [(str(p.relative_to(path.parent)) if (p := find_model_path(m, path.parent)) else None)
                   for m in reg["model_id"]]
    reg = reg.sort_values(["F1_macro", "model_id"], ascending=[False, True]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    reg.to_csv(path, index=False)
    return reg


def prediction_frame(model_id: str, ids, proba: np.ndarray, fold=None, pred: np.ndarray | None = None,
                     calibrated: bool = False) -> pd.DataFrame:
    """Formato común de predicciones: model_id, id, fold, p_HS, p_TR, p_AG, calibrated (+ HS, TR, AG en test)."""
    df = pd.DataFrame({"model_id": model_id, "id": np.asarray(ids),
                       "fold": -1 if fold is None else np.asarray(fold),
                       **{f"p_{lab}": proba[:, j] for j, lab in enumerate(LABELS)}, "calibrated": calibrated})
    if pred is not None:
        for j, lab in enumerate(LABELS):
            df[lab] = pred[:, j].astype(int)
    return df


def update_predictions(frames: list[pd.DataFrame], path: Path) -> pd.DataFrame:
    """Añade o reemplaza (por model_id) predicciones en un parquet compartido."""
    required = ["model_id", "id", "fold", "p_HS", "p_TR", "p_AG", "calibrated"]
    out = _upsert(path, pd.concat(frames, ignore_index=True), "model_id", required)
    out = out.sort_values(["model_id", "id"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


# ---------------------------------------------------------------------------
# Envío a Kaggle
# ---------------------------------------------------------------------------
def make_submission(model_id: str, out_dir: Path = SUBMISSIONS_DIR) -> Path:
    """Carga models/<model_id>.joblib, predice sobre el texto crudo de data/toxic_test.csv y escribe
    submissions/<model_id>.csv con columnas id,HS,TR,AG (enteros 0/1). Verifica el archivo antes de escribirlo."""
    test = load_test()
    if test is None:
        raise FileNotFoundError("No existe data/toxic_test.csv")
    pred = load_model(model_id).predict(test["text"])
    sub = pd.DataFrame({"id": test["id"].to_numpy(), **{lab: pred[:, j].astype(int) for j, lab in enumerate(LABELS)}})

    assert len(sub) == len(test), "número de filas distinto al del test"
    assert (sub["id"].to_numpy() == test["id"].to_numpy()).all(), "ids distintos o en otro orden"
    assert not sub.isna().any().any(), "hay valores nulos"
    assert set(np.unique(sub[LABELS].to_numpy())) <= {0, 1}, "etiquetas fuera de {0, 1}"
    samples = sorted(p for p in DATA_DIR.glob("*.csv") if "sample" in p.name.lower() or "submission" in p.name.lower())
    if samples:
        sample_cols = list(pd.read_csv(samples[0], sep=None, engine="python", nrows=1).columns)
        assert list(sub.columns) == sample_cols, f"columnas {list(sub.columns)} != {sample_cols} ({samples[0].name})"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_id}.csv"
    sub.to_csv(path, index=False)
    return path
