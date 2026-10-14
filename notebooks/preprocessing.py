"""Preprocesamiento de tuits para TweetToxic.

Funciones puras (sin estado global mutable), en el orden en que se aplican:

    a) clean_text              -> artefactos del CSV, <USER>, <URL>, hashtags, RT
    b) lowercase
    c) reduce_elongations / normalize_laughs
    d) normalize_censorship
    e) restore_damaged_words   (diccionario construido con build_restore_dict, sin etiquetas)
    f) lemmatize               (spaCy: solo lemma_, pos_ y morph; corrección con LEMMA_FIXES)
    g) apply_variant / build_views
    h) TweetPreprocessor: transformer de scikit-learn que aplica una variante completa, para
       usarlo dentro de un Pipeline y guardarlo junto al modelo (joblib)

Ningún paso usa etiquetas. Todos los diccionarios de este módulo son de elaboración propia
(ver la celda "Recursos externos" de 02_preprocesamiento.ipynb).
"""

from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Iterable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

USER_TOKEN = "<USER>"
URL_TOKEN = "<URL>"
SPECIAL_TOKENS = (USER_TOKEN, URL_TOKEN)

# ---------------------------------------------------------------------------
# a) Limpieza
# ---------------------------------------------------------------------------
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MENTION = re.compile(r"(?<![\w.])@\w+")  # no toca los '@' de los correos
RE_HASHTAG = re.compile(r"#(\w+)")
RE_RT = re.compile(r"\bRT\b:?", flags=re.IGNORECASE)
RE_SPACES = re.compile(r"\s+")
RE_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])|(?<=[A-Z])(?=[A-Z][a-z])")


def remove_non_ascii(text: str) -> str:
    """Borra los caracteres no ASCII, igual que la pérdida que sufrió el train (cállate -> cllate).

    Se aplica a cualquier texto que no venga del train (test, recursos externos) para que
    coincida con él. En el train y el test actuales no cambia nada (no hay no ASCII), salvo
    el '¿' de las celdas '#¿NOMBRE?' del test.
    """
    return text.encode("ascii", "ignore").decode("ascii")


def fix_csv_artifacts(text: str) -> str:
    """Repara artefactos de exportación: '#¿NOMBRE?' (celda de Excel sin texto), comillas
    duplicadas ("" -> ") y entidades HTML (&gt; -> >)."""
    if text.strip() in {"#¿NOMBRE?", "#NOMBRE?"}:
        return ""
    text = html.unescape(text)
    text = re.sub(r'"{2,}', '"', text)
    return text.strip().strip('"').strip()


def split_hashtag(tag: str) -> str:
    """'DefiendeEspaa' -> 'Defiende Espaa'; 'sudaca' -> 'sudaca' (sin diccionario no se puede partir)."""
    return RE_CAMEL.sub(" ", tag).replace("_", " ")


def clean_text(text: str, *, artifacts: bool = True, users: bool = True, urls: bool = True,
               hashtags: bool = True, rt: bool = True) -> str:
    """Paso a: limpieza estructural del tuit.

    - artifacts: repara los artefactos del CSV y borra los caracteres no ASCII.
    - users: '@usuario' -> '<USER>'.
    - urls: 'https://t.co/...' -> '<URL>'.
    - hashtags: '#DefiendeEspaa' -> 'Defiende Espaa' (quita '#' y separa por mayúsculas).
    - rt: elimina la marca de retuit 'RT'.
    """
    if artifacts:
        text = fix_csv_artifacts(remove_non_ascii(text))
    if urls:
        text = RE_URL.sub(f" {URL_TOKEN} ", text)
    if users:
        text = RE_MENTION.sub(f" {USER_TOKEN} ", text)
    if hashtags:
        text = RE_HASHTAG.sub(lambda m: " " + split_hashtag(m.group(1)) + " ", text)
    if rt:
        text = RE_RT.sub(" ", text)
    return RE_SPACES.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# b) Minúsculas
# ---------------------------------------------------------------------------
def lowercase(text: str) -> str:
    """Pasa a minúsculas conservando los tokens especiales (<USER>, <URL>)."""
    text = text.lower()
    for tok in SPECIAL_TOKENS:
        text = text.replace(tok.lower(), tok)
    return text


# ---------------------------------------------------------------------------
# c) Alargamientos y risas
# ---------------------------------------------------------------------------
RE_ELONG = re.compile(r"([^\W\d_])\1{2,}")
# Risa: token con 2+ 'j' formado solo por j/a/e/i/o/s/h/k (jaja, jajjaja, jsjs, jejeje), o 'xd', 'xddd'
RE_LAUGH = re.compile(r"(?i)\b(?=[a-z]*j[a-z]*j)[jaeioshk]{4,}\b|\bx+d+\b")
LAUGH_TOKEN = "jaja"


def reduce_elongations(text: str, max_repeat: int = 2) -> str:
    """Recorta letras repetidas 3+ veces: max_repeat=2 -> 'holaaaa' -> 'holaa'; max_repeat=1 -> 'hola'.

    Con max_repeat=1 también se rompen dígrafos alargados ('callllate' -> 'calate'), por eso
    se comparan ambas opciones.
    """
    return RE_ELONG.sub(lambda m: m.group(1) * max_repeat, text)


def normalize_laughs(text: str) -> str:
    """Unifica todas las risas (jajaja, jsjsjs, jejeje, JAJAJAJ, xd) en un único token 'jaja'."""
    return RE_LAUGH.sub(LAUGH_TOKEN, text)


# ---------------------------------------------------------------------------
# d) Censura y abreviaturas de insultos
# ---------------------------------------------------------------------------
# Diccionario propio y pequeño. Se aplica en minúsculas.
CENSORSHIP_PATTERNS: list[tuple[str, str]] = [
    (r"\bp[\*@\.]+t[a@\*]s?\b|\bput[@\*]|\bpvta\b", "puta"),
    (r"\bhdp\b|\bhdpm\b|\bhdlgp\b|\bhijueputa\b|\bhp\b", "hijo de puta"),
    (r"\bctm\b|\bcsm\b|\bconchetumare\b|\bconchetumadre\b", "concha tu madre"),
    (r"\blpm\b|\blpqtp\b", "la puta madre"),
    (r"\bptm\b", "puta madre"),
    (r"\bm[\*@]+erda\b|\bmrd\b", "mierda"),
    (r"\bpndj[oa]?\b|\bpndjo\b", "pendejo"),
    (r"\bmlp\b", "malparido"),
]
_CENSORSHIP = [(re.compile(p), r) for p, r in CENSORSHIP_PATTERNS]


def normalize_censorship(text: str) -> str:
    """Lleva insultos censurados o abreviados a una forma canónica: p*ta/put@ -> puta, hdp -> hijo de puta,
    ctm -> concha tu madre, lpm -> la puta madre..."""
    for rx, repl in _CENSORSHIP:
        text = rx.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# e) Restauración de palabras dañadas por la pérdida de tildes/ñ
# ---------------------------------------------------------------------------
# Lista base del EDA (sección 4 y 5b): forma dañada -> forma ASCII que escribiría un usuario sin tildes.
# Se restaura a ASCII (no a 'cállate') para fusionar 'cllate' con 'callate', que ya existe en el corpus.
MANUAL_RESTORE: dict[str, str] = {
    "cllate": "callate", "callte": "callate", "ms": "mas", "qu": "que", "t": "tu", "est": "esta", "ests": "estas",
    "espaa": "espana", "espaol": "espanol", "espaola": "espanola", "espaoles": "espanoles",
    "ensear": "ensenar", "ensearte": "ensenarte", "rabe": "arabe", "rabes": "arabes",
    "inmigracin": "inmigracion", "violacin": "violacion", "gero": "guero", "gera": "guera", "geros": "gueros",
    "geras": "gueras", "coo": "cono", "imbcil": "imbecil", "imbciles": "imbeciles", "aqu": "aqui",
    "despus": "despues", "tambin": "tambien", "nio": "nino", "nia": "nina", "nios": "ninos", "nias": "ninas",
    "estpida": "estupida", "estpido": "estupido", "maricn": "maricon", "cabrn": "cabron", "seor": "senor",
    "mam": "mama", "pap": "papa", "pas": "pais", "maana": "manana", "pequea": "pequena",
    "mtete": "metete", "murete": "muerete", "lrgate": "largate", "ndate": "andate", "djate": "dejate",
    "scate": "sacate", "fjate": "fijate", "jdete": "jodete", "mtate": "matate", "chpala": "chupala",
    "slo": "solo", "quin": "quien", "cundo": "cuando", "merecs": "mereces", "vergenza": "verguenza",
    "recin": "recien", "amrica": "america", "polica": "policia", "pelcula": "pelicula", "adems": "ademas",
    "ejrcito": "ejercito", "sper": "super", "msica": "musica", "ltimo": "ultimo", "cmo": "como", "yaa": "ya",
}
DELETED_LETTERS = "aeiou"  # á é í ó ú ü pierden la vocal; la ñ se trata aparte (solo entre vocales)
ABBREVIATIONS = {"rt", "xq", "pq", "q", "k", "hdp", "ctm", "lpm", "ptm", "hp", "csm", "xd", "xdd", "xddd", "mmm",
                 "mm", "tb", "tmb", "cc", "dm", "bb", "pls", "plz", "gt", "lt", "amp", "http", "https", "cdmx",
                 "ddhh", "fc", "tv", "pm", "sr", "mr", "dr", "fb", "pp", "psoe", "mx", "df", "km", "kg", "vs",
                 "eeuu", "user", "url", "mrd", "mlp"}
RE_WORD = re.compile(r"[a-z]+")
# Comienzos de palabra con dos consonantes que sí existen (español y préstamos frecuentes del inglés).
VALID_ONSETS = {"bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "tr", "ch", "ll", "kr", "ps",
                "sh", "st", "sp", "sn", "sm", "sk", "sc", "sw", "th", "tw", "wh", "wr", "gn", "ph", "rh", "kn"}


def is_suspicious(word: str) -> bool:
    """Reglas del EDA (sección 4), más una de comienzo imposible, para detectar palabras que perdieron letras."""
    if len(word) >= 4 and re.match(r"[^aeiouy][^aeiouy]", word) and word[:2] not in VALID_ONSETS \
            and word not in ABBREVIATIONS:
        return True  # comienzo imposible en español: cmo, dnde, vdeo, cllate
    if len(word) >= 5 and re.search(r"[cs]in$", word):
        return True  # -ción/-sión -> -cin/-sin
    if len(word) >= 2 and not re.search(r"[aeiouy]", word) and word not in ABBREVIATIONS \
            and not re.fullmatch(r"(js)+|(\w)\2+", word):
        return True  # sin vocales (ms, cmo no)
    if re.search(r"(aa|ii|uu)", word) and not RE_ELONG.search(word) and word not in ABBREVIATIONS:
        return True  # vocal doble improbable (maana)
    return False


def _candidates(word: str) -> set[str]:
    """Formas que resultan de insertar una vocal en cualquier posición, o una 'n' entre dos vocales (ñ)."""
    out = {word[:i] + c + word[i:] for i in range(len(word) + 1) for c in DELETED_LETTERS}
    out |= {word[:i] + "n" + word[i:] for i in range(1, len(word))
            if word[i - 1] in "aeiou" and word[i] in "aeiou"}
    return out


def build_restore_dict(texts: Iterable[str], min_len: int = 3) -> dict[str, str]:
    """Construye el diccionario de restauración a partir de TEXTOS (sin etiquetas).

    1. Parte de MANUAL_RESTORE (palabras dañadas frecuentes detectadas en el EDA).
    2. Para cada palabra del corpus que parece dañada (is_suspicious) y no está ya en el
       diccionario, busca la forma con una letra más (vocal, o 'n' entre vocales) que sí
       aparezca en el corpus y no parezca dañada; si hay varias, elige la más frecuente.
    3. Si no hay candidata pero termina en -cin (o vocal + -sin), aplica la regla -> -cion/-sion.

    Espera textos ya limpios y en minúsculas.
    """
    vocab = Counter(w for t in texts for w in RE_WORD.findall(t))
    restore = dict(MANUAL_RESTORE)
    for w in vocab:
        if w in restore or len(w) < min_len or not is_suspicious(w):
            continue
        cands = [(vocab[c], c) for c in _candidates(w) if vocab.get(c, 0) > 0 and not is_suspicious(c)]
        if cands:
            restore[w] = max(cands)[1]
        elif w.endswith("cin"):
            restore[w] = w[:-3] + "cion"
        elif re.search(r"[aeiou]sin$", w):  # agresin, invasin (no 'alfonsin')
            restore[w] = w[:-3] + "sion"
    return restore


def restore_damaged_words(text: str, restore: dict[str, str]) -> str:
    """Paso e: sustituye cada palabra dañada por su forma restaurada (cllate -> callate)."""
    return RE_WORD.sub(lambda m: restore.get(m.group(0), m.group(0)), text)


# ---------------------------------------------------------------------------
# f) Lematización con spaCy (solo lemma_, pos_, morph; sin vectores)
# ---------------------------------------------------------------------------
SPACY_MODEL = "es_core_news_sm"
SPACY_EXCLUDE = ["ner", "parser", "senter"]

# Correcciones propias (D9). Clave: forma superficial (tras restaurar) -> lema correcto.
SURFACE_LEMMA_FIXES: dict[str, str] = {
    "callate": "callar", "calla": "callar", "callense": "callar", "callaos": "callar",
    "vete": "ir", "andate": "ir", "largate": "largar", "muerete": "morir", "metete": "meter", "jodete": "joder",
    "matate": "matar", "comemela": "comer", "chupamela": "chupar", "chupala": "chupar", "fijate": "fijar",
    "sacate": "sacar", "dejate": "dejar", "ponte": "poner", "dame": "dar",
    "perra": "perra", "perras": "perra", "zorra": "zorra", "zorras": "zorra", "puta": "puta", "putas": "puta",
    "arabe": "arabe", "arabes": "arabe", "mas": "mas", "jaja": "jaja",
}
# Clave: lema inventado por spaCy -> lema correcto.
LEMMA_FIXES: dict[str, str] = {
    "perrar": "perra", "perrir": "perra", "zorrar": "zorra", "putar": "puta", "rabir": "arabe", "rabar": "arabe",
    "arabir": "arabe", "vetir": "ir", "callatar": "callar", "callatir": "callar", "cllatar": "callar", "jajar": "jaja",
}


def load_spacy():
    """Carga es_core_news_sm sin ner/parser/senter y comprueba que no trae vectores."""
    import spacy

    nlp = spacy.load(SPACY_MODEL, exclude=SPACY_EXCLUDE)
    assert nlp.vocab.vectors.shape[0] == 0, "el modelo no debe traer vectores preentrenados"
    return nlp


_NLP = None


def get_nlp():
    """spaCy cargado una sola vez por proceso (cargarlo cuesta ~0.5 s)."""
    global _NLP
    if _NLP is None:
        _NLP = load_spacy()
    return _NLP


def lemmatize(texts: Iterable[str], nlp=None, keep_second_person: bool = True, batch_size: int = 64) -> list[str]:
    """Paso f: devuelve, por texto, la secuencia de lemas de sus palabras alfabéticas.

    - Usa solo token.lemma_, token.pos_ y token.morph (nunca vectores).
    - Corrige los lemas con SURFACE_LEMMA_FIXES y LEMMA_FIXES.
    - keep_second_person: si el token tiene Person=2 (eres, te, mereces) conserva la forma
      original, porque el lema (ser, tú, merecer) borra la señal de segunda persona (H18).
    - Los tokens especiales <USER>/<URL> se omiten (ya están en la vista de palabras).
    """
    nlp = nlp or get_nlp()
    texts = [t.replace(USER_TOKEN, " ").replace(URL_TOKEN, " ") for t in texts]
    out = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        lemmas = []
        for tok in doc:
            if not tok.is_alpha:
                continue
            low = tok.lower_
            if low in SURFACE_LEMMA_FIXES:
                lemma = SURFACE_LEMMA_FIXES[low]
            elif keep_second_person and "2" in tok.morph.get("Person"):
                lemma = low
            else:
                lemma = tok.lemma_.lower()
                lemma = LEMMA_FIXES.get(lemma, lemma)
            lemmas.append(lemma)
        out.append(" ".join(lemmas))
    return out


# ---------------------------------------------------------------------------
# g) Variantes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Variant:
    """Qué pasos se aplican. Cada flag corresponde a un paso que puede quitarse en la ablación."""
    name: str = "custom"
    artifacts: bool = False
    users: bool = False
    urls: bool = False
    hashtags: bool = False
    rt: bool = False
    lower: bool = False
    elong: int = 0          # 0 = no recortar; 1 o 2 = repeticiones máximas
    laughs: bool = False
    censor: bool = False
    restore: bool = False
    lemma: bool = False     # vista extra de lemas (no sustituye a las palabras)

    def without(self, step: str) -> "Variant":
        """Copia de la variante con un paso desactivado (para la ablación)."""
        value = 0 if step == "elong" else False
        return replace(self, name=f"{self.name} - {step}", **{step: value})

    def steps(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "name"}


V0 = Variant("V0")
V1 = Variant("V1", artifacts=True, users=True, urls=True, hashtags=True, rt=True, lower=True)
V2 = replace(V1, name="V2", elong=2, laughs=True, censor=True)
V2_E1 = replace(V2, name="V2 (alarg.=1)", elong=1)
V3 = replace(V2, name="V3", restore=True)
V4 = replace(V3, name="V4", lemma=True)
VARIANTS = {v.name: v for v in [V0, V1, V2, V2_E1, V3, V4]}
ABLATION_STEPS = ["artifacts", "users", "urls", "hashtags", "rt", "lower", "elong", "laughs", "censor", "restore",
                  "lemma"]


def apply_variant(text: str, variant: Variant, restore: dict[str, str] | None = None) -> str:
    """Aplica a un texto los pasos a-e de la variante (la lematización va aparte: es por lotes)."""
    text = clean_text(text, artifacts=variant.artifacts, users=variant.users, urls=variant.urls,
                      hashtags=variant.hashtags, rt=variant.rt)
    if variant.lower:
        text = lowercase(text)
    if variant.elong:
        text = reduce_elongations(text, max_repeat=variant.elong)
    if variant.laughs:
        text = normalize_laughs(text)
    if variant.censor:
        text = normalize_censorship(text)
    if variant.restore:
        if restore is None:
            raise ValueError("La variante restaura palabras: pasa el diccionario de build_restore_dict().")
        text = restore_damaged_words(text, restore)
    return text


def restore_dict_for(texts: Iterable[str]) -> dict[str, str]:
    """Diccionario de restauración calculado sobre los textos limpios y en minúsculas (sin etiquetas)."""
    return build_restore_dict(lowercase(clean_text(t)) for t in texts)


def build_views(texts: pd.Series, variant: Variant, restore: dict[str, str] | None = None,
                lemma_cache: dict[str, str] | None = None, nlp=None) -> pd.DataFrame:
    """Devuelve un DataFrame con la columna 'text' procesada y, si la variante lo pide, 'lemmas'.

    lemma_cache (texto procesado -> lemas) evita recalcular spaCy: su salida es determinista
    y no depende de las etiquetas, así que se calcula una vez y se reutiliza en todos los folds.
    """
    processed = texts.map(lambda t: apply_variant(t, variant, restore))
    views = pd.DataFrame({"text": processed.values}, index=texts.index)
    if variant.lemma:
        cache = lemma_cache if lemma_cache is not None else {}
        missing = [t for t in pd.unique(processed) if t not in cache]
        if missing:
            cache.update(zip(missing, lemmatize(missing, nlp=nlp)))
        views["lemmas"] = processed.map(cache).values
    return views


def variant_slug(variant: Variant | str) -> str:
    """Nombre apto para archivos: 'V2 (alarg.=1)' -> 'V2-alarg1'."""
    name = variant if isinstance(variant, str) else variant.name
    return re.sub(r"[^A-Za-z0-9]+", "-", name.replace(".=", "")).strip("-")


# ---------------------------------------------------------------------------
# h) Transformer de scikit-learn
# ---------------------------------------------------------------------------
# Caché de lemas compartida en el proceso: texto procesado -> lemas. La salida de spaCy es
# determinista y no depende de etiquetas, así que reutilizarla entre folds no filtra información.
LEMMA_CACHE: dict[str, str] = {}


def load_lemma_cache(path) -> int:
    """Precarga LEMMA_CACHE desde un parquet (columnas text, lemmas). Devuelve cuántas entradas añadió."""
    df = pd.read_parquet(path)
    before = len(LEMMA_CACHE)
    LEMMA_CACHE.update(zip(df["text"], df["lemmas"]))
    return len(LEMMA_CACHE) - before


def save_lemma_cache(path) -> None:
    pd.DataFrame({"text": list(LEMMA_CACHE), "lemmas": list(LEMMA_CACHE.values())}).to_parquet(path, index=False)


def _as_text_series(X) -> pd.Series:
    if isinstance(X, pd.DataFrame):
        X = X["text"]
    if isinstance(X, pd.Series):
        return X.astype(str).reset_index(drop=True)
    return pd.Series(list(X), dtype=str)


class TweetPreprocessor(BaseEstimator, TransformerMixin):
    """Aplica una variante de preprocesamiento dentro de un Pipeline de scikit-learn.

    - variant: nombre de VARIANTS ('V0', 'V1', 'V2', 'V2 (alarg.=1)', 'V3', 'V4') o un objeto Variant.
    - restore_dict: diccionario de restauración fijo. Si es None y la variante restaura palabras,
      se construye en fit() con los textos de entrenamiento (sin etiquetas), así que dentro de una
      validación cruzada se recalcula en cada fold.

    - keep_raw: añade la columna 'raw' con el texto crudo (para rasgos que necesitan, por ejemplo,
      las mayúsculas originales).

    transform() devuelve un DataFrame con la columna 'text' (pasos a-e) y, si la variante lematiza,
    'lemmas' (paso f). Se guarda con joblib junto al modelo; spaCy no se serializa, se carga al usarlo.
    """

    def __init__(self, variant: str | Variant = "V1", restore_dict: dict[str, str] | None = None,
                 keep_raw: bool = False):
        self.variant = variant
        self.restore_dict = restore_dict
        self.keep_raw = keep_raw

    def _resolve_variant(self) -> Variant:
        return VARIANTS[self.variant] if isinstance(self.variant, str) else self.variant

    def fit(self, X, y=None):
        self.variant_ = self._resolve_variant()
        if self.variant_.restore:
            self.restore_dict_ = (dict(self.restore_dict) if self.restore_dict is not None
                                  else restore_dict_for(_as_text_series(X)))
        else:
            self.restore_dict_ = None
        return self

    def transform(self, X) -> pd.DataFrame:
        check_is_fitted(self, "variant_")
        texts = _as_text_series(X)
        views = build_views(texts, self.variant_, self.restore_dict_, lemma_cache=LEMMA_CACHE)
        if getattr(self, "keep_raw", False):
            views["raw"] = texts.values
        return views
