"""Configuración compartida por los notebooks de TweetToxic.

Uso desde un notebook en notebooks/:
    from common import SEED, LABELS, DATA_DIR, load_train, load_test, make_folds, evaluate_oof
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

# Semilla única para todo lo aleatorio (splits, muestreos, modelos).
SEED = 42

# Etiquetas de la competencia, en el orden del CSV.
LABELS = ["HS", "TR", "AG"]

# Rutas resueltas a partir de la ubicación de este archivo (notebooks/),
# así funcionan igual desde VSCode, Jupyter o nbconvert.
NOTEBOOKS_DIR = Path(__file__).resolve().parent
ROOT_DIR = NOTEBOOKS_DIR.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"  # derivados (folds, textos procesados); los CSV originales no se tocan
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

TRAIN_PATH = DATA_DIR / "toxic_train.csv"
TEST_PATH = DATA_DIR / "toxic_test.csv"
FOLDS_PATH = PROCESSED_DIR / "folds.csv"

# Separador del CSV de la competencia.
CSV_SEP = ";"


def _read_csv(path: Path) -> pd.DataFrame:
    """Lee un CSV de la competencia probando UTF-8 y, si falla, Latin-1.

    El train es ASCII puro, pero el test trae algún byte Latin-1 suelto
    (p. ej. '¿' en '#¿NOMBRE?'), que no es UTF-8 válido.
    """
    try:
        df = pd.read_csv(path, sep=CSV_SEP, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=CSV_SEP, encoding="latin-1")
    # Una fila del train trae el id con comillas sobrantes ('"22621'):
    # se limpian para que la columna sea entera. El texto NO se toca aquí.
    df["id"] = df["id"].astype(str).str.strip('"').astype(int)
    return df


def load_train(path: Path = TRAIN_PATH) -> pd.DataFrame:
    """Carga el conjunto de entrenamiento (id, text, HS, TR, AG) sin limpiar el texto."""
    df = _read_csv(path)
    df[LABELS] = df[LABELS].astype(int)
    return df


def load_test(path: Path = TEST_PATH) -> pd.DataFrame | None:
    """Carga el conjunto de prueba (id, text). Devuelve None si el archivo no existe."""
    if not Path(path).exists():
        print(f"[load_test] No se encontró {path}; se omiten los análisis que dependen del test.")
        return None
    return _read_csv(path)


def _near_duplicate_key(text: str) -> str:
    """Texto normalizado para agrupar casi-duplicados: minúsculas, sin menciones, URLs, RT ni signos."""
    t = re.sub(r"https?://\S+|www\.\S+|(?<![\w.])@\w+", " ", text).lower()
    t = re.sub(r"\brt\b|[^\w\s]|_", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def make_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = SEED, path: Path = FOLDS_PATH,
               overwrite: bool = False) -> pd.Series:
    """Asigna cada fila a un fold y guarda la asignación en data/processed/folds.csv.

    - Estratifica por la combinación de etiquetas (p. ej. "101" = HS=1, TR=0, AG=1), para que
      cada fold tenga la misma proporción de cada combinación.
    - Mantiene los casi-duplicados (mismo texto normalizado) en el mismo fold, para que un
      tuit repetido no aparezca a la vez en entrenamiento y validación (decisión D8).
    - Si el archivo ya existe y cubre los mismos ids, lo reutiliza: todos los notebooks
      comparten así exactamente los mismos folds.

    Devuelve una Series `fold` (0..n_splits-1) alineada con el índice de `df`.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        saved = pd.read_csv(path)
        if set(saved["id"]) == set(df["id"]) and saved["fold"].nunique() == n_splits:
            return df["id"].map(saved.set_index("id")["fold"]).rename("fold")
        raise ValueError(f"{path} no coincide con los datos actuales; usa overwrite=True para regenerarlo.")

    combo = df[LABELS].astype(str).agg("".join, axis=1)
    groups = df["text"].map(_near_duplicate_key)
    fold = pd.Series(-1, index=df.index, name="fold")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(sgkf.split(df, combo, groups)):
        fold.iloc[val_idx] = k
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": df["id"], "fold": fold}).to_csv(path, index=False)
    return fold


def evaluate_oof(y_true, y_proba, thresholds=0.5) -> dict:
    """F1 por etiqueta, F1 macro y F1 micro a partir de probabilidades.

    - y_true, y_proba: arrays (n, 3) en el orden de LABELS.
    - thresholds: un número (mismo umbral para todas) o una secuencia/dict por etiqueta.
    - F1 macro = media simple de los F1 de las tres etiquetas.
    - F1 micro = F1 calculado juntando todas las decisiones (n x 3) en un único conteo de TP/FP/FN.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if isinstance(thresholds, dict):
        thresholds = [thresholds[lab] for lab in LABELS]
    thr = np.broadcast_to(np.asarray(thresholds, dtype=float), (len(LABELS),))
    y_pred = (y_proba >= thr).astype(int)
    out = {f"F1_{lab}": f1_score(y_true[:, j], y_pred[:, j], zero_division=0) for j, lab in enumerate(LABELS)}
    out["F1_macro"] = float(np.mean([out[f"F1_{lab}"] for lab in LABELS]))
    out["F1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    return out
