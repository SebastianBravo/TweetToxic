"""Rasgos hechos a mano para TweetToxic (decisiones D6 y D10).

Se calculan sobre el texto CRUDO: las minúsculas de V1+ destruyen la señal de mayúsculas.
Ningún rasgo usa etiquetas; los léxicos se definieron a mano en 01_EDA (§6 y §5b) antes de
mirar los resultados y están citados en PROGRESO.md §9.

Uso dentro de un pipeline: HandcraftedFeatures() -> MaxAbsScaler() sobre la columna "raw".
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import preprocessing as pp

# ---------------------------------------------------------------------------
# Léxicos (copiados de 01_EDA §6 y §5b, sin cambios)
# ---------------------------------------------------------------------------
# Insultos generales. Las variantes con [..]? cubren la tilde borrada (imbcil) y la censura (p*ta, put@).
INSULTS = {
    "puta/puto": r"\bp[u\*@v]t[ao@\*x]s?\b|\bputi\w*|\bputer\w*",
    "hijo de puta / hdp": r"\bhij[oa]s? de (?:la gran |su )?p\S*t[ao]\b|\bhdp\b|\bhdlgp\b|\bhijueputa\b|\bhp\b",
    "ctm / concha tu madre": r"\bctm\b|\bcsm\b|\bconch[ae] ?(?:de )?(?:tu|su) ?madre\b|\bconchetumadre\b",
    "perra/o": r"\bp[e\*]rr[ao@]s?\b",
    "zorra": r"\bz[o\*]rr[ao@]s?\b",
    "marica/maricón": r"\bmari[ck]\w*",
    "pendejo": r"\bpendej\w*",
    "imbécil": r"\bimb[e]?cil\w*",
    "idiota": r"\bidiota\w*",
    "estúpido": r"\best[u]?pid\w*",
    "gilipollas": r"\bgilipoll\w*",
    "cabrón": r"\bcabr[o]?n\w*",
    "mierda": r"\bm[i\*]erd\w*",
    "coño": r"\bco[ñn]?o\b",
    "culiado/culero": r"\bculi[ao]d[ao]s?\b|\bculiao\b|\bculer[oa]s?\b",
    "malparido": r"\bmal ?parid[oa]s?\b",
    "verga": r"\bverg[ao]s?\b",
    "subnormal/retrasado": r"\bsubnormal\w*|\bretrasad[oa]s?\b|\bmong[o]?li\w*",
    "basura/escoria": r"\bbasura\b|\bescoria\b",
}
# Términos despectivos hacia grupos (origen, etnia, género)
GROUP_SLURS = {
    "sudaca": r"\bsudac\w*",
    "negrata": r"\bnegrat\w*",
    "panchito": r"\bpanchit\w*",
    "moro (despectivo)": r"\bmoracos?\b|\bmoros? de mierda\b",
    "feminazi": r"\bfeminazi\w*",
}
IMPERATIVE_FORMS = {
    "callate", "cllate", "callte", "calla", "callense", "callaos", "vete", "andate", "ndate", "largate", "lrgate",
    "muerete", "murete", "metete", "mtete", "jodete", "jdete", "matate", "mtate", "comemela", "chupamela",
    "chupala", "chpala", "fijate", "fjate", "sacate", "scate", "ponte", "dame",
}
SECOND_PERSON_FORMS = {"tu", "t", "te", "ti", "eres", "sos", "vos", "tus", "contigo"}

_INSULT_RX = {k: re.compile(p) for k, p in INSULTS.items()}
_SLUR_RX = {k: re.compile(p) for k, p in GROUP_SLURS.items()}
_WORD = re.compile(r"[^\W\d_]+")
_UPPER = re.compile(r"[A-ZÁÉÍÓÚÑÜ]")
_LETTER = re.compile(r"[^\W\d_]")

SURFACE_NAMES = ["log_chars", "log_words", "starts_mention", "log_mentions", "has_url", "has_hashtag", "has_laugh",
                 "upper_ratio", "log_excl", "log_quest"]
LEXICAL_NAMES = ([f"insult:{k}" for k in INSULTS] + [f"slur:{k}" for k in GROUP_SLURS]
                 + ["n_insult_groups", "any_slur", "imperative", "log_second_person", "person2_spacy"])
FEATURE_NAMES = SURFACE_NAMES + LEXICAL_NAMES

# Caché de Person=2 de spaCy (texto en minúsculas sin menciones/URLs -> 0/1). Determinista y sin etiquetas.
PERSON2_CACHE: dict[str, int] = {}


def _person2(texts: list[str]) -> np.ndarray:
    missing = [t for t in dict.fromkeys(texts) if t not in PERSON2_CACHE]
    if missing:
        for t, doc in zip(missing, pp.get_nlp().pipe(missing, batch_size=64)):
            PERSON2_CACHE[t] = int(any("2" in tok.morph.get("Person") for tok in doc))
    return np.array([PERSON2_CACHE[t] for t in texts])


def surface_features(raw: str) -> list[float]:
    clean = pp.RE_MENTION.sub(" ", pp.RE_URL.sub(" ", raw))  # estilo sin URLs ni menciones (como en 01 §3)
    letters = len(_LETTER.findall(clean))
    return [
        np.log1p(len(raw)),
        np.log1p(len(raw.split())),
        float(re.match(r"\s*@\w+", raw) is not None),
        np.log1p(len(pp.RE_MENTION.findall(raw))),
        float(pp.RE_URL.search(raw) is not None),
        float(re.search(r"(?<!\w)#\w+", raw) is not None),
        float(pp.RE_LAUGH.search(clean) is not None),
        len(_UPPER.findall(clean)) / letters if letters else 0.0,
        np.log1p(raw.count("!")),
        np.log1p(raw.count("?")),
    ]


def lexical_features(lower_clean: str) -> list[float]:
    insults = [float(rx.search(lower_clean) is not None) for rx in _INSULT_RX.values()]
    slurs = [float(rx.search(lower_clean) is not None) for rx in _SLUR_RX.values()]
    words = _WORD.findall(lower_clean)
    return insults + slurs + [
        float(sum(insults)),
        float(any(slurs)),
        float(any(w in IMPERATIVE_FORMS for w in words)),
        np.log1p(sum(w in SECOND_PERSON_FORMS for w in words)),
    ]


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """Texto crudo -> matriz densa (n, 39) de rasgos superficiales y léxicos (ver FEATURE_NAMES).

    No aprende nada en fit(): todos los rasgos son reglas fijas. El escalado va después (MaxAbsScaler).
    use_spacy=False omite Person=2 de spaCy (la columna queda a 0).
    """

    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy

    def fit(self, X, y=None):
        self.n_features_in_ = 1
        return self

    def transform(self, X) -> np.ndarray:
        raw = pp._as_text_series(X).tolist()
        lower_clean = [pp.RE_MENTION.sub(" ", pp.RE_URL.sub(" ", t.lower())) for t in raw]
        surface = np.array([surface_features(t) for t in raw])
        lexical = np.array([lexical_features(t) for t in lower_clean])
        person2 = _person2([re.sub(r"\s+", " ", t).strip() for t in lower_clean]) if self.use_spacy else np.zeros(len(raw))
        return np.column_stack([surface, lexical, person2]).astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES, dtype=object)


def handcrafted_frame(texts: pd.Series) -> pd.DataFrame:
    """Los mismos rasgos como DataFrame con nombres (para inspección)."""
    return pd.DataFrame(HandcraftedFeatures().fit_transform(texts), columns=FEATURE_NAMES, index=texts.index)
