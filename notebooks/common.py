"""Configuración compartida por los notebooks de TweetToxic.

Uso desde un notebook en notebooks/:
    from common import SEED, LABELS, DATA_DIR, load_train, load_test
"""

from pathlib import Path

import pandas as pd

# Semilla única para todo lo aleatorio (splits, muestreos, modelos).
SEED = 42

# Etiquetas de la competencia, en el orden del CSV.
LABELS = ["HS", "TR", "AG"]

# Rutas resueltas a partir de la ubicación de este archivo (notebooks/),
# así funcionan igual desde VSCode, Jupyter o nbconvert.
NOTEBOOKS_DIR = Path(__file__).resolve().parent
ROOT_DIR = NOTEBOOKS_DIR.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

TRAIN_PATH = DATA_DIR / "toxic_train.csv"
TEST_PATH = DATA_DIR / "toxic_test.csv"

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
