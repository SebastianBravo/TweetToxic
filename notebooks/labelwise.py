"""Modelos especializados por etiqueta (09).

Los modelos de modeling.py ya son binarios por etiqueta (MultiOutputClassifier entrena un clasificador
para HS, otro para TR y otro para AG), pero los tres comparten variante, representación e
hiperparámetros. Aquí se especializan:

- LabelwiseClassifier: cada etiqueta con su propia configuración (variante, representación e
  hiperparámetros), elegida con una CV interna (select_label_configs).
- ConditionalClassifier: HS con todas las filas; TR|HS y AG|HS solo con las filas HS=1 (H2: TR=1 implica
  HS=1). P(TR) = P(HS)·P(TR|HS) y P(AG) = P(HS)·P(AG|HS).
- PowersetClassifier: un clasificador multiclase sobre las combinaciones de etiquetas (000, 100, 101, ...)
  y marginalización: P(HS) = suma de las probabilidades de las combinaciones con HS=1, etc.

Todos reciben texto crudo, devuelven predict_proba (n, 3) en el orden HS, TR, AG y tienen el mismo
predict() (umbrales, jerarquía y tuits vacíos) que el resto de modelos (modeling.DecisionMixin).
fit(X, y, sample_weight) acepta un peso por fila (11: filas de un corpus externo con peso < 1).
"""

from __future__ import annotations

import datetime as dt
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import modeling as md
from common import LABELS, SEED

AG = LABELS.index("AG")
BASE_CONFIG = {"variant": "V2", "features": "tfidf", "C": 10.0}  # la de V2__logreg__nested en todo el train
# Rejilla por etiqueta (10 configuraciones): variante × representación, con la regresión logística
# elasticnet (l1_ratio 0.5, C=10), la familia y el C que eligió la CV anidada del 04 en todo el train
LABEL_GRID = ([{"variant": v, "features": f, "C": 10.0} for v in ("V1", "V2", "V3")
               for f in ("tfidf", "char", "tfidf+hand")]
              + [{"variant": "V2", "features": "word", "C": 10.0}])


def _logreg(C):
    return md.make_classifier("logreg", {"C": C, "l1_ratio": 0.5})


def _fit_quiet(est, X, y, sample_weight=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if sample_weight is None:
            return est.fit(X, y)
        key = f"{est.steps[-1][0]}__sample_weight" if isinstance(est, Pipeline) else "sample_weight"
        return est.fit(X, y, **{key: np.asarray(sample_weight, float)})


def _subset(w, mask):
    return None if w is None else np.asarray(w)[mask]


class SingleLabelModel(BaseEstimator):
    """Texto crudo -> P(etiqueta) con una configuración {variant, features, C}."""

    def __init__(self, variant="V2", features="tfidf", C=10.0):
        self.variant = variant
        self.features = features
        self.C = C

    def fit(self, X, y, sample_weight=None):
        featurizer = md.build_featurizer(self.variant, self.features)
        self.pipeline_ = _fit_quiet(Pipeline(featurizer.steps + [("clf", _logreg(self.C))]), X, np.asarray(y), sample_weight)
        return self

    def predict_proba1(self, X) -> np.ndarray:
        return self.pipeline_.predict_proba(X)[:, 1]


class _Base(md.DecisionMixin, BaseEstimator, ClassifierMixin):
    thresholds = 0.5
    hierarchy = True
    empty_as_negative = True


class LabelwiseClassifier(_Base):
    """Una configuración por etiqueta: configs = {"HS": {...}, "TR": {...}, "AG": {...}}.
    exclude_quoted_ag: el modelo de AG se entrena sin las filas del artefacto de comillas (P2)."""

    def __init__(self, configs=None, exclude_quoted_ag=False, thresholds=0.5, hierarchy=True, empty_as_negative=True):
        self.configs = configs
        self.exclude_quoted_ag = exclude_quoted_ag
        self.thresholds = thresholds
        self.hierarchy = hierarchy
        self.empty_as_negative = empty_as_negative

    def fit(self, X, y, sample_weight=None):
        X = pd.Series(list(X)) if not isinstance(X, pd.Series) else X.reset_index(drop=True)
        Y = np.asarray(y)
        configs = self.configs or {lab: BASE_CONFIG for lab in LABELS}
        self.models_ = []
        for j, lab in enumerate(LABELS):
            keep = ~md.quoted_mask(X) if (lab == "AG" and self.exclude_quoted_ag) else np.ones(len(X), bool)
            self.models_.append(SingleLabelModel(**configs[lab]).fit(X[keep], Y[keep, j], _subset(sample_weight, keep)))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return np.column_stack([m.predict_proba1(X) for m in self.models_])


class ConditionalClassifier(_Base):
    """HS con todas las filas; TR|HS y AG|HS con las filas HS=1 (AG|HS además sin las filas del artefacto).
    P(TR) = P(HS)·P(TR|HS), P(AG) = P(HS)·P(AG|HS)."""

    def __init__(self, config=None, thresholds=0.5, hierarchy=True, empty_as_negative=True):
        self.config = config
        self.thresholds = thresholds
        self.hierarchy = hierarchy
        self.empty_as_negative = empty_as_negative

    def fit(self, X, y, sample_weight=None):
        X = pd.Series(list(X)) if not isinstance(X, pd.Series) else X.reset_index(drop=True)
        Y = np.asarray(y)
        cfg = self.config or BASE_CONFIG
        hs = Y[:, 0] == 1
        self.hs_ = SingleLabelModel(**cfg).fit(X, Y[:, 0], sample_weight)
        self.tr_hs_ = SingleLabelModel(**cfg).fit(X[hs], Y[hs, 1], _subset(sample_weight, hs))
        keep = hs & ~md.quoted_mask(X)
        self.ag_hs_ = SingleLabelModel(**cfg).fit(X[keep], Y[keep, AG], _subset(sample_weight, keep))
        return self

    def predict_proba(self, X) -> np.ndarray:
        p_hs = self.hs_.predict_proba1(X)
        return np.column_stack([p_hs, p_hs * self.tr_hs_.predict_proba1(X), p_hs * self.ag_hs_.predict_proba1(X)])


class PowersetClassifier(_Base):
    """Regresión logística multinomial (L2, lbfgs: la elasticnet multinomial con saga tarda ≈20 veces más)
    sobre las combinaciones de etiquetas presentes en el train; las de
    menos de min_count ejemplos se agrupan con la combinación frecuente más parecida (distancia de Hamming).
    Después se marginaliza: P(etiqueta) = suma de P(combinación) de las combinaciones con esa etiqueta a 1."""

    def __init__(self, config=None, min_count=20, thresholds=0.5, hierarchy=True, empty_as_negative=True):
        self.config = config
        self.min_count = min_count
        self.thresholds = thresholds
        self.hierarchy = hierarchy
        self.empty_as_negative = empty_as_negative

    def fit(self, X, y, sample_weight=None):
        Y = np.asarray(y).astype(int)
        cfg = self.config or BASE_CONFIG
        combos = np.array(["".join(map(str, r)) for r in Y])
        counts = pd.Series(combos).value_counts()
        frequent = [c for c in counts.index if counts[c] >= self.min_count]
        hamming = lambda a, b: sum(x != y for x, y in zip(a, b))
        self.mapping_ = {c: (c if c in frequent else min(frequent, key=lambda f: (hamming(c, f), -counts[f])))
                         for c in counts.index}
        target = np.array([self.mapping_[c] for c in combos])
        featurizer = md.build_featurizer(cfg["variant"], cfg["features"])
        clf = LogisticRegression(C=cfg["C"], solver="lbfgs", max_iter=3000, class_weight="balanced",
                                 random_state=SEED)
        self.pipeline_ = _fit_quiet(Pipeline(featurizer.steps + [("clf", clf)]), X, target, sample_weight)
        classes = self.pipeline_[-1].classes_
        self.marginal_ = np.array([[int(c[j]) for j in range(len(LABELS))] for c in classes], dtype=float)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.pipeline_.predict_proba(X) @ self.marginal_


# ---------------------------------------------------------------------------
# Selección de la configuración por etiqueta (CV interna)
# ---------------------------------------------------------------------------
def _best_f1(y, p) -> float:
    """F1 de una etiqueta con su umbral óptimo (para comparar configuraciones en la CV interna; optimista
    por igual para todas)."""
    y = np.asarray(y).astype(bool)
    pred = np.asarray(p)[:, None] >= md.THRESHOLD_GRID[None, :]
    tp = (pred & y[:, None]).sum(0)
    denom = 2 * tp + (pred & ~y[:, None]).sum(0) + (~pred & y[:, None]).sum(0)
    return float(np.max(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def _inner_fold_scores(texts, Y, trn, val, grid, quoted, exclude_quoted_ag):
    out = {}
    for vf in sorted({(g["variant"], g["features"]) for g in grid}):
        featurizer = md.build_featurizer(*vf).fit(texts[trn], Y[trn])
        Xtr, Xva = featurizer.transform(texts[trn]), featurizer.transform(texts[val])
        for i, g in enumerate(grid):
            if (g["variant"], g["features"]) != vf:
                continue
            cols = []
            for j, lab in enumerate(LABELS):
                keep = ~quoted[trn] if (lab == "AG" and exclude_quoted_ag) else np.ones(len(trn), bool)
                cols.append(_fit_quiet(_logreg(g["C"]), Xtr[keep], Y[trn][keep, j]).predict_proba(Xva)[:, 1])
            out[i] = np.column_stack(cols)
    return val, out


def select_label_configs(texts, Y, grid=LABEL_GRID, n_splits=3, exclude_quoted_ag=False, n_jobs=1):
    """Elige, para cada etiqueta por separado, la configuración de la rejilla con mayor F1 (umbral óptimo)
    en una CV interna de n_splits folds sobre (texts, Y). La representación se ajusta una vez por fold
    interno y combinación (variante, features). Devuelve ({etiqueta: config}, tabla de puntuaciones)."""
    from joblib import Parallel, delayed

    texts = pd.Series(texts).reset_index(drop=True)
    Y = np.asarray(Y)
    quoted = md.quoted_mask(texts)
    inner = {i: np.zeros(Y.shape) for i in range(len(grid))}
    res = Parallel(n_jobs=n_jobs)(delayed(_inner_fold_scores)(texts, Y, trn, val, grid, quoted, exclude_quoted_ag)
                                  for trn, val in md._inner_splits(texts, Y, n_splits))
    for val, out in res:
        for i, p in out.items():
            inner[i][val] = p
    rows = []
    for i, g in enumerate(grid):
        rows.append({**g, **{lab: _best_f1(Y[:, j], inner[i][:, j]) for j, lab in enumerate(LABELS)}})
    scores = pd.DataFrame(rows)
    best = {lab: {k: scores.loc[scores[lab].idxmax(), k] for k in ("variant", "features", "C")} for lab in LABELS}
    for lab in LABELS:
        best[lab]["C"] = float(best[lab]["C"])
    return best, scores


def _nested_labelwise_fold(texts, Y, trn, val, grid, exclude_quoted_ag):
    configs, _ = select_label_configs(texts[trn].reset_index(drop=True), Y[trn], grid,
                                      exclude_quoted_ag=exclude_quoted_ag)
    model = LabelwiseClassifier(configs, exclude_quoted_ag=exclude_quoted_ag).fit(texts[trn], Y[trn])
    return model.predict_proba(texts[val]), configs


def nested_labelwise_cv(texts, Y, folds, grid=LABEL_GRID, exclude_quoted_ag=False, n_jobs=5) -> dict:
    """CV anidada de LabelwiseClassifier: en cada fold externo, select_label_configs con sus filas de
    entrenamiento; el fold externo solo mide. Devuelve oof y la configuración elegida en cada fold."""
    from joblib import Parallel, delayed

    texts = pd.Series(texts).reset_index(drop=True)
    folds = np.asarray(folds)
    ks = sorted(np.unique(folds))
    res = Parallel(n_jobs=n_jobs)(delayed(_nested_labelwise_fold)(texts, Y, folds != k, folds == k, grid,
                                                                  exclude_quoted_ag) for k in ks)
    oof = np.zeros(np.asarray(Y).shape)
    for k, (p, _) in zip(ks, res):
        oof[folds == k] = p
    return {"oof": oof, "configs_per_fold": [c for _, c in res]}


def save_labelwise(model, name: str, metadata: dict | None = None, notebook: str = "09") -> Path:
    """models/<carpeta del notebook>/<name>.joblib + .json (modelo completo: texto crudo -> etiquetas)."""
    path = md.model_folder(notebook) / f"{name}.joblib"
    joblib.dump(model, path, compress=3)
    meta = {"name": name, "created": dt.datetime.now().isoformat(timespec="seconds"), "class": type(model).__name__,
            "params": {k: v for k, v in model.get_params().items()}, "labels": LABELS,
            "environment": md.environment_info(), **(metadata or {})}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    return path
