"""Ensambles de TweetToxic (07): combinan modelos propios ya entrenados a partir de sus probabilidades.

Entrada común de todos los métodos: un array P de forma (n, M, 3) con las probabilidades de M modelos base
para HS, TR y AG (OOF en la CV, o las del modelo final en el test). Cada método ("combinador") tiene

    fit(P, Y, groups)   ajusta lo que use etiquetas (calibradores, pesos, stacker) SOLO con esas filas;
                        groups = fold de cada fila, para las CV internas (p. ej. elegir C del stacker);
    predict(P)          devuelve probabilidades combinadas (n, 3).

Métodos, de menos a más complejo: Single (un modelo, referencia), Mean (promedio simple), CalMean
(promedio de probabilidades calibradas), Weighted (pesos >= 0 que suman 1, por log-loss), Stacking
(regresión logística L2 por etiqueta sobre los logits; con todas las etiquetas o solo la propia) y
Caruana (selección voraz con reemplazo y bagging).

Protocolo de evaluación (crossfit): para cada fold externo k, el combinador se ajusta con las OOF de los
otros 4 folds y se aplica al fold k. El umbral de cada etiqueta se elige también sin mirar el fold k: con
las predicciones del combinador sobre los 4 folds de entrenamiento obtenidas por una CV interna (ajustado
en 3, aplicado al 4.º), con la curva de F1 suavizada y el centro de la meseta (plateau_threshold).

EnsembleClassifier envuelve el resultado final: texto crudo -> modelos base (cargados por id con
modeling.load_model) -> combinador -> umbrales, jerarquía y regla de tuits vacíos.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import modeling as md
from common import LABELS, SEED, evaluate_oof

EPS = 1e-4                      # recorte de probabilidades antes del logit y de la log-loss
THRESHOLD_GRID = md.THRESHOLD_GRID
SMOOTH_HALF_WINDOW = 3          # promedio móvil de ±3 puntos de la rejilla (±0.03)
PLATEAU_TOL = 0.005             # la meseta: puntos con F1 suavizado >= máximo − 0.005
STACK_CS = (0.01, 0.1, 1.0)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def logit(p):
    p = clip(p)
    return np.log(p / (1 - p))


def log_loss(y, p) -> float:
    p = clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def log_loss_cols(y, Q) -> np.ndarray:
    """Log-loss de cada columna de Q (n, m) frente a y (n,)."""
    Q = clip(Q)
    y = np.asarray(y, dtype=float)[:, None]
    return -np.mean(y * np.log(Q) + (1 - y) * np.log(1 - Q), axis=0)


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - y) ** 2))


def stored_proba(model_ids: list[str], ids, path) -> np.ndarray:
    """(n, M, 3) con las probabilidades guardadas en un parquet de predicciones, en el orden de `ids`."""
    store = pd.read_parquet(path, columns=["model_id", "id", "p_HS", "p_TR", "p_AG"])
    out = []
    for m in model_ids:
        df = store[store["model_id"] == m].set_index("id")
        out.append(df.loc[np.asarray(ids), [f"p_{lab}" for lab in LABELS]].to_numpy())
    return np.stack(out, axis=1)


# ---------------------------------------------------------------------------
# Umbrales
# ---------------------------------------------------------------------------
def f1_curve(y, p, grid=THRESHOLD_GRID) -> np.ndarray:
    """F1 de una etiqueta para cada umbral de la rejilla (vectorizado)."""
    y = np.asarray(y).astype(bool)
    pred = np.asarray(p)[:, None] >= grid[None, :]
    tp = (pred & y[:, None]).sum(0)
    fp = (pred & ~y[:, None]).sum(0)
    fn = (~pred & y[:, None]).sum(0)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def smooth(curve, half_window=SMOOTH_HALF_WINDOW) -> np.ndarray:
    """Promedio móvil centrado (en los bordes, con los puntos disponibles)."""
    n = len(curve)
    return np.array([curve[max(0, i - half_window):min(n, i + half_window + 1)].mean() for i in range(n)])


def plateau_range(y, p, grid=THRESHOLD_GRID, half_window=SMOOTH_HALF_WINDOW, tol=PLATEAU_TOL) -> tuple[int, int]:
    """Índices (inicio, fin) de la meseta: el tramo continuo alrededor del máximo de la curva de F1
    suavizada donde el F1 suavizado está a menos de `tol` del máximo."""
    s = smooth(f1_curve(y, p, grid), half_window)
    best = int(np.argmax(s))
    ok = s >= s[best] - tol
    lo, hi = best, best
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(s) - 1 and ok[hi + 1]:
        hi += 1
    return lo, hi


def plateau_threshold(y, p, grid=THRESHOLD_GRID, half_window=SMOOTH_HALF_WINDOW, tol=PLATEAU_TOL) -> float:
    """Centro de la meseta (plateau_range). Menos sensible al ruido que el argmax."""
    lo, hi = plateau_range(y, p, grid, half_window, tol)
    return float(grid[int(round((lo + hi) / 2))])


def argmax_threshold(y, p, grid=THRESHOLD_GRID) -> float:
    return float(grid[int(np.argmax(f1_curve(y, p, grid)))])


THRESHOLD_RULES = {"plateau": plateau_threshold, "argmax": argmax_threshold}


def choose_thresholds(Y, proba, rule="plateau") -> dict:
    return {lab: THRESHOLD_RULES[rule](Y[:, j], proba[:, j]) for j, lab in enumerate(LABELS)}


def decide(proba, thresholds, hierarchy=True, empty=None) -> np.ndarray:
    """Umbral por etiqueta, jerarquía HS=0 -> TR=0 (D7) y tuits vacíos -> 000 (D2)."""
    pred = md.apply_decision(proba, thresholds, hierarchy)
    if empty is not None:
        pred[np.asarray(empty)] = 0
    return pred


def metrics(Y, pred) -> dict:
    return evaluate_oof(Y, pred, 0.5)


# ---------------------------------------------------------------------------
# Calibración (por modelo y etiqueta)
# ---------------------------------------------------------------------------
class Calibrator:
    """Transforma la puntuación de un modelo para una etiqueta.

    - none: sin cambios;
    - platt: regresión logística sobre el logit de la puntuación (2 parámetros: pendiente y sesgo);
    - isotonic: función escalonada creciente que minimiza el error cuadrático (sin forma fija);
    - quantile: función de distribución empírica de las puntuaciones de entrenamiento (posición relativa
      en [0, 1]). No es una probabilidad: sirve para poner todos los modelos en la misma escala.
    """

    def __init__(self, method="platt"):
        self.method = method

    def fit(self, p, y):
        p = np.asarray(p, dtype=float)
        if self.method == "platt":
            self.lr_ = LogisticRegression(C=100.0, max_iter=1000).fit(logit(p)[:, None], y)
        elif self.method == "isotonic":
            self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        elif self.method == "quantile":
            self.sorted_ = np.sort(p)
        elif self.method != "none":
            raise ValueError(f"Calibración desconocida: {self.method}")
        return self

    def transform(self, p):
        p = np.asarray(p, dtype=float)
        if self.method == "platt":
            out = self.lr_.predict_proba(logit(p)[:, None])[:, 1]
        elif self.method == "isotonic":
            out = self.iso_.predict(p)
        elif self.method == "quantile":
            n = len(self.sorted_)
            out = (np.searchsorted(self.sorted_, p, side="right") + 0.5) / (n + 1)
        else:
            out = p
        return clip(out)


class CalibratedSet:
    """Un calibrador por modelo y etiqueta: P (n, M, 3) -> P calibrada (n, M, 3)."""

    def __init__(self, method="platt"):
        self.method = method

    def fit(self, P, Y):
        _, M, L = P.shape
        self.cals_ = [[Calibrator(self.method).fit(P[:, m, j], Y[:, j]) for j in range(L)] for m in range(M)]
        return self

    def transform(self, P):
        out = np.empty_like(P, dtype=float)
        for m, row in enumerate(self.cals_):
            for j, cal in enumerate(row):
                out[:, m, j] = cal.transform(P[:, m, j])
        return out


# ---------------------------------------------------------------------------
# Combinadores
# ---------------------------------------------------------------------------
class Combiner:
    name = "base"

    def fit(self, P, Y, groups=None):
        return self

    def predict(self, P):
        raise NotImplementedError

    def describe(self) -> dict:
        return {"method": self.name}


class Single(Combiner):
    """Un solo modelo (la referencia E): devuelve sus probabilidades sin cambios."""
    name = "single"

    def __init__(self, index=0):
        self.index = index

    def predict(self, P):
        return np.asarray(P[:, self.index, :], dtype=float)


class Mean(Combiner):
    """(i) Promedio simple de las probabilidades, sin calibrar."""
    name = "mean"

    def predict(self, P):
        return np.asarray(P, dtype=float).mean(axis=1)


class CalMean(Combiner):
    """(ii) Promedio de las probabilidades calibradas."""
    name = "cal-mean"

    def __init__(self, calib="platt"):
        self.calib = calib

    def fit(self, P, Y, groups=None):
        self.cal_ = CalibratedSet(self.calib).fit(P, Y)
        return self

    def predict(self, P):
        return self.cal_.transform(P).mean(axis=1)

    def describe(self):
        return {"method": self.name, "calib": self.calib}


class Weighted(Combiner):
    """(iii) Media ponderada de probabilidades calibradas; por etiqueta, pesos >= 0 que suman 1 que
    minimizan la log-loss en las filas de entrenamiento."""
    name = "weights"

    def __init__(self, calib="platt"):
        self.calib = calib

    def fit(self, P, Y, groups=None):
        self.cal_ = CalibratedSet(self.calib).fit(P, Y)
        Q = self.cal_.transform(P)
        M = Q.shape[1]
        self.weights_ = np.zeros((M, Q.shape[2]))
        for j in range(Q.shape[2]):
            if M == 1:
                self.weights_[:, j] = 1.0
                continue
            res = minimize(lambda w, j=j: log_loss(Y[:, j], Q[:, :, j] @ w), np.full(M, 1 / M), method="SLSQP",
                           bounds=[(0.0, 1.0)] * M, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                           options={"maxiter": 500, "ftol": 1e-10})
            w = np.clip(res.x, 0, None)
            self.weights_[:, j] = w / w.sum()
        return self

    def predict(self, P):
        Q = self.cal_.transform(P)
        return np.einsum("nmj,mj->nj", Q, self.weights_)

    def describe(self):
        return {"method": self.name, "calib": self.calib, "weights": self.weights_.round(4).tolist()}


class Stacking(Combiner):
    """(iv)/(vi) Stacking: una regresión logística L2 por etiqueta cuya entrada son los logits de los
    modelos base. own_label_only=False usa los logits de las 3 etiquetas de cada modelo (3·M entradas);
    True, solo los de la etiqueta que se predice (M entradas). C se elige entre STACK_CS con una CV interna
    por grupos (los folds de las filas de entrenamiento), por log-loss."""

    def __init__(self, own_label_only=False, Cs=STACK_CS):
        self.own_label_only = own_label_only
        self.Cs = Cs

    @property
    def name(self):
        return "stack-own" if self.own_label_only else "stack-all"

    def _X(self, P, j):
        L = logit(P)
        return L[:, :, j] if self.own_label_only else L.reshape(len(L), -1)

    @staticmethod
    def _fit_predict(C, Xtr, ytr, Xte):
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), ytr)
        return lr.predict_proba(sc.transform(Xte))[:, 1]

    def fit(self, P, Y, groups=None):
        groups = np.zeros(len(P)) if groups is None else np.asarray(groups)
        ug = np.unique(groups)
        self.models_, self.C_ = [], []
        for j in range(Y.shape[1]):
            X = self._X(P, j)
            if len(ug) >= 2:
                losses = []
                for C in self.Cs:
                    oof = np.zeros(len(X))
                    for g in ug:
                        te = groups == g
                        oof[te] = self._fit_predict(C, X[~te], Y[~te, j], X[te])
                    losses.append(log_loss(Y[:, j], oof))
                C = self.Cs[int(np.argmin(losses))]
            else:
                C = self.Cs[-1]
            sc = StandardScaler().fit(X)
            lr = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X), Y[:, j])
            self.models_.append((sc, lr))
            self.C_.append(C)
        return self

    def predict(self, P):
        return np.column_stack([lr.predict_proba(sc.transform(self._X(P, j)))[:, 1]
                                for j, (sc, lr) in enumerate(self.models_)])

    def describe(self):
        return {"method": self.name, "C": self.C_,
                "coef": [lr.coef_.ravel().round(4).tolist() for _, lr in self.models_]}


class Caruana(Combiner):
    """(v) Selección voraz con reemplazo (Caruana et al., 2004) con bagging, por etiqueta.

    En cada una de n_bags bolsas se sortea un subconjunto de modelos (fracción bag_frac) y, partiendo de
    cero, se añade en cada paso (con reemplazo) el modelo que más baja la log-loss del promedio acumulado
    sobre las filas de entrenamiento; se guarda el número de pasos con la menor log-loss. Los pesos finales
    son la media, entre bolsas, de las veces que se eligió cada modelo. Entrada: probabilidades calibradas."""
    name = "caruana"

    def __init__(self, calib="platt", n_bags=20, bag_frac=0.5, max_iter=25, seed=SEED):
        self.calib = calib
        self.n_bags = n_bags
        self.bag_frac = bag_frac
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, P, Y, groups=None):
        self.cal_ = CalibratedSet(self.calib).fit(P, Y)
        Q = self.cal_.transform(P)
        n, M, L = Q.shape
        rng = np.random.default_rng(self.seed)
        self.weights_ = np.zeros((M, L))
        for j in range(L):
            y = Y[:, j]
            for _ in range(self.n_bags):
                members = np.sort(rng.choice(M, max(1, math.ceil(self.bag_frac * M)), replace=False))
                counts, best_counts, total, best = np.zeros(M), None, np.zeros(n), np.inf
                for it in range(1, self.max_iter + 1):
                    losses = log_loss_cols(y, (total[:, None] + Q[:, members, j]) / it)
                    pick = members[int(np.argmin(losses))]
                    counts[pick] += 1
                    total += Q[:, pick, j]
                    if losses.min() < best - 1e-12:
                        best, best_counts = losses.min(), counts.copy()
                self.weights_[:, j] += best_counts / best_counts.sum()
            self.weights_[:, j] /= self.n_bags
        return self

    def predict(self, P):
        return np.einsum("nmj,mj->nj", self.cal_.transform(P), self.weights_)

    def describe(self):
        return {"method": self.name, "calib": self.calib, "n_bags": self.n_bags, "bag_frac": self.bag_frac,
                "max_iter": self.max_iter, "weights": self.weights_.round(4).tolist()}


def make_combiner(method: str, calib: str = "platt") -> Combiner:
    return {"mean": lambda: Mean(), "cal-mean": lambda: CalMean(calib), "weights": lambda: Weighted(calib),
            "stack-all": lambda: Stacking(False), "caruana": lambda: Caruana(calib),
            "stack-own": lambda: Stacking(True)}[method]()


METHODS = ["mean", "cal-mean", "weights", "stack-all", "caruana", "stack-own"]
METHOD_NAMES = {"mean": "(i) promedio simple", "cal-mean": "(ii) promedio calibrado",
                "weights": "(iii) pesos por log-loss", "stack-all": "(iv) stacking (todas las etiquetas)",
                "caruana": "(v) selección voraz (Caruana)", "stack-own": "(vi) stacking (etiqueta propia)",
                "single": "modelo único"}


# ---------------------------------------------------------------------------
# Protocolo de evaluación
# ---------------------------------------------------------------------------
def crossfit(combiner: Combiner, P, Y, folds, empty=None) -> dict:
    """Evalúa un combinador con la CV externa (ver la docstring del módulo).

    Devuelve:
    - proba: probabilidades combinadas fuera de fold (n, 3);
    - thresholds: {regla: {fold: umbrales}} con las reglas 'plateau' y 'argmax';
    - folds: {(regla, jerarquía): DataFrame de métricas por fold};
    - pred: decisiones fuera de fold con la regla 'plateau' y jerarquía.
    """
    folds = np.asarray(folds)
    ks = sorted(np.unique(folds))
    proba = np.zeros(Y.shape)
    thresholds = {r: {} for r in THRESHOLD_RULES}
    rows = {(r, h): [] for r in THRESHOLD_RULES for h in (True, False)}
    pred = np.zeros_like(Y)
    for k in ks:
        val, trn = folds == k, folds != k
        inner = np.zeros((trn.sum(), Y.shape[1]))
        g_trn = folds[trn]
        for j in sorted(np.unique(g_trn)):
            te = g_trn == j
            c = copy.deepcopy(combiner).fit(P[trn][~te], Y[trn][~te], g_trn[~te])
            inner[te] = c.predict(P[trn][te])
        fitted = copy.deepcopy(combiner).fit(P[trn], Y[trn], g_trn)
        proba[val] = fitted.predict(P[val])
        emp = None if empty is None else np.asarray(empty)[val]
        for r in THRESHOLD_RULES:
            thr = choose_thresholds(Y[trn], inner, r)
            thresholds[r][k] = thr
            for h in (True, False):
                d = decide(proba[val], thr, h, emp)
                rows[(r, h)].append(metrics(Y[val], d))
                if r == "plateau" and h:
                    pred[val] = d
    return {"proba": proba, "thresholds": thresholds, "folds": {key: pd.DataFrame(v) for key, v in rows.items()},
            "pred": pred}


def f1_folds(res: dict, metric="macro", rule="plateau", hierarchy=True) -> pd.Series:
    return res["folds"][(rule, hierarchy)][f"F1_{metric}"]


def paired_delta(a: pd.Series, b: pd.Series) -> dict:
    d = np.asarray(a) - np.asarray(b)
    return {"Δ": float(d.mean()), "Δ desv.": float(d.std(ddof=1)), "folds a favor": int((d > 0).sum())}


def fit_final(combiner: Combiner, P, Y, folds, oof_proba, rule="plateau"):
    """Combinador ajustado con todas las OOF y umbrales elegidos sobre sus predicciones fuera de fold
    (las del crossfit), con la misma regla que en la evaluación."""
    fitted = copy.deepcopy(combiner).fit(P, Y, np.asarray(folds))
    return fitted, choose_thresholds(Y, oof_proba, rule)


# ---------------------------------------------------------------------------
# Modelo final
# ---------------------------------------------------------------------------
def _predict_one(model_id: str, texts) -> np.ndarray:
    """Probabilidades de un modelo base (se ejecuta en un proceso aparte)."""
    return md.load_model(model_id).predict_proba(texts)


class EnsembleClassifier(md.DecisionMixin):
    """Modelo final del 07: texto crudo -> [HS, TR, AG].

    Carga los modelos base por id (modeling.load_model, en cualquier carpeta de models/), obtiene sus
    probabilidades y aplica el combinador ya ajustado, los umbrales, la jerarquía y la regla de tuits
    vacíos (predict, heredado de DecisionMixin). Los modelos base no se guardan dentro: se cargan de
    disco cada vez.

    Los modelos base predicen en procesos aparte, en dos tandas con procesos nuevos: primero los de
    scikit-learn/xgboost/LightGBM y después las redes de torch. Cada grupo trae su propio runtime de OpenMP
    y, en un mismo proceso, el segundo se bloquea o falla (comprobado en macOS). Las probabilidades son
    idénticas a las de predecir en el proceso principal."""

    def __init__(self, base_ids, combiner: Combiner, thresholds, hierarchy=True, empty_as_negative=True,
                 name=None, n_jobs=4):
        self.base_ids = list(base_ids)
        self.combiner = combiner
        self.thresholds = thresholds
        self.hierarchy = hierarchy
        self.empty_as_negative = empty_as_negative
        self.name = name
        self.n_jobs = n_jobs

    def base_proba(self, X) -> np.ndarray:
        from joblib import Parallel, delayed
        from joblib.externals.loky import get_reusable_executor

        texts = X.reset_index(drop=True) if isinstance(X, pd.Series) else pd.Series(list(X))
        is_torch = {m: md.find_model_path(m).suffix == ".pt" for m in self.base_ids}
        out = {}
        for group in ([m for m in self.base_ids if not is_torch[m]], [m for m in self.base_ids if is_torch[m]]):
            if not group:
                continue
            get_reusable_executor().shutdown(wait=True)
            probas = Parallel(n_jobs=min(self.n_jobs, len(group)))(delayed(_predict_one)(m, texts) for m in group)
            out.update(zip(group, probas))
        get_reusable_executor().shutdown(wait=True)
        return np.stack([out[m] for m in self.base_ids], axis=1)

    def predict_proba_from_base(self, P) -> np.ndarray:
        return self.combiner.predict(np.asarray(P, dtype=float))

    def predict_proba(self, X) -> np.ndarray:
        return self.predict_proba_from_base(self.base_proba(X))


def save_ensemble(model: EnsembleClassifier, name: str, metadata: dict | None = None,
                  models_dir: Path = md.MODELS_DIR) -> Path:
    """models/07_ensamble/<name>.joblib (combinador, ids, umbrales) + .json legible."""
    import torch

    path = md.model_folder("07", models_dir) / f"{name}.joblib"
    joblib.dump(model, path, compress=3)
    meta = {"name": name, "created": dt.datetime.now().isoformat(timespec="seconds"), "family": "ensamble",
            "base_models": model.base_ids, "combiner": model.combiner.describe(),
            "thresholds": model.thresholds, "hierarchy": model.hierarchy,
            "empty_as_negative": model.empty_as_negative, "labels": LABELS,
            "environment": md.environment_info() | {"torch": str(torch.__version__)}, **(metadata or {})}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    return path
