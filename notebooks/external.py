"""Corpus externo etiquetado (11): HatEval, SemEval-2019 Task 5 (Basile et al., 2019), parte en español.

- download_hateval(): copia local sin modificar de los parquet del dataset `valeriobasile/HatEval` en una
  revisión fija, con un manifest.json (revisión, fecha, filas, sha256). Usa el token guardado por
  `huggingface-cli login`; este módulo no lo lee ni lo imprime.
- deduplicate(): quita del corpus externo todo lo que coincide con nuestro train o test (por id, por clave
  normalizada y por similitud coseno de n-gramas de caracteres) y los duplicados internos. Solo mira id y
  texto: las etiquetas se leen después (read_labels) y nunca las de las filas que coinciden con el test.
- fit_augmented() / augmented_cv(): entrenamiento con nuestras filas + las externas (con un peso por fila).
  En la CV las externas solo entran en los folds de entrenamiento y todo lo que se valida (OOF, selección de
  hiperparámetros, early stopping de las redes) usa solo nuestras filas.

torch se importa solo dentro de las funciones que entrenan redes (D33: no convive con xgboost/LightGBM en un
mismo proceso).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier

import labelwise as lw
import modeling as md
import preprocessing as pp
from common import DATA_DIR, LABELS, _near_duplicate_key

HATEVAL_REPO = "valeriobasile/HatEval"
HATEVAL_REVISION = "d170b5cb1844f3614493fce7c7eaaae087a5201c"
HATEVAL_LICENSE = "CC BY-NC 4.0"
HATEVAL_CITATION = ("Basile, V., Bosco, C., Fersini, E., Nozza, D., Patti, V., Rangel Pardo, F. M., Rosso, P. y "
                    "Sanguinetti, M. (2019). SemEval-2019 Task 5: Multilingual Detection of Hate Speech Against "
                    "Immigrants and Women in Twitter. Proceedings of the 13th International Workshop on Semantic "
                    "Evaluation (SemEval-2019), 54-63. https://aclanthology.org/S19-2007")
SPLITS = ("train", "dev", "test")
EXTERNAL_DIR = Path(os.environ.get("TWEETTOXIC_EXTERNAL_DIR", DATA_DIR / "external"))
HATEVAL_DIR = EXTERNAL_DIR / "hateval"
COSINE_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_ok(man: dict, out_dir: Path, revision: str) -> bool:
    return (man.get("revision") == revision
            and all((out_dir / f["file"]).exists() and sha256(out_dir / f["file"]) == f["sha256"]
                    for f in man["files"].values()))


def download_hateval(revision: str = HATEVAL_REVISION, out_dir: Path = HATEVAL_DIR) -> dict:
    """Copia byte a byte los parquet de cada split en out_dir/<split>.parquet y escribe out_dir/manifest.json.
    Si ya existen y su sha256 coincide con el manifest, no descarga nada. Además carga el dataset con
    `datasets` en la misma revisión para comprobar que las filas y las columnas coinciden con los parquet."""
    out_dir = Path(out_dir)
    man_path = out_dir / "manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text())
        if _manifest_ok(man, out_dir, revision):
            man["reused"] = True
            return man
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().dataset_info(HATEVAL_REPO, revision=revision)
    assert info.sha == revision, f"revisión {info.sha} != {revision}"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in SPLITS:
        remote = f"data/{split}-00000-of-00001.parquet"
        cached = hf_hub_download(HATEVAL_REPO, remote, repo_type="dataset", revision=revision)
        local = out_dir / f"{split}.parquet"
        shutil.copyfile(cached, local)
        df = pd.read_parquet(local, columns=["language"])
        files[split] = {"file": local.name, "source": remote, "rows": len(df),
                        "rows_es": int((df["language"] == "es").sum()), "bytes": local.stat().st_size,
                        "sha256": sha256(local)}
    ds = load_dataset(HATEVAL_REPO, revision=revision)
    for split in SPLITS:
        assert ds[split].num_rows == files[split]["rows"], f"{split}: filas distintas entre datasets y el parquet"
    card = info.card_data.to_dict() if info.card_data else {}
    man = {"repo": HATEVAL_REPO, "revision": revision, "downloaded": dt.datetime.now().isoformat(timespec="seconds"),
           "license": card.get("license", "cc-by-nc-4.0"), "gated": info.gated,
           "columns": list(ds["train"].features), "files": files, "citation": HATEVAL_CITATION,
           "note": "Copia sin modificar. No se versiona (data/external/ está en .gitignore)."}
    man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    man["reused"] = False
    return man


def read_texts(out_dir: Path = HATEVAL_DIR, language: str = "es") -> pd.DataFrame:
    """id, text, language y split de todos los splits, SIN etiquetas."""
    parts = [pd.read_parquet(Path(out_dir) / f"{s}.parquet", columns=["id", "text", "language"]).assign(split=s)
             for s in SPLITS]
    df = pd.concat(parts, ignore_index=True)
    return df[df["language"] == language].reset_index(drop=True)


def read_labels(ids, out_dir: Path = HATEVAL_DIR, columns=("HS", "TR", "AG", "target")) -> pd.DataFrame:
    """Etiquetas solo de los ids pedidos (filtro de pyarrow al leer: las demás filas no se cargan)."""
    ids = [int(i) for i in ids]
    parts = [pd.read_parquet(Path(out_dir) / f"{s}.parquet", columns=["id", *columns], filters=[("id", "in", ids)])
             for s in SPLITS]
    return pd.concat(parts, ignore_index=True).set_index("id").loc[ids].reset_index()


# ---------------------------------------------------------------------------
# Deduplicación
# ---------------------------------------------------------------------------
def deduplicate(ext: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
                threshold: float = COSINE_THRESHOLD) -> pd.DataFrame:
    """Clasifica cada fila externa (id, text ya sin no ASCII) en un estado, por orden:

    1. id-test / id-train: su id está en nuestro test / train;
    2. clave-test / clave-train: misma clave normalizada (common._near_duplicate_key) que un tuit nuestro;
    3. coseno-test / coseno-train: similitud coseno >= threshold con algún tuit nuestro (TF-IDF de n-gramas
       de caracteres char_wb 3-5 sobre las claves);
    4. duplicado-interno: misma clave que otra fila externa anterior que se conserva;
    5. se conserva.
    Ante coincidencias con train y test a la vez gana el test (lo más estricto). Devuelve ext con las columnas
    key, status, match_id (id nuestro coincidente), sim_train / sim_test (máxima similitud con el train y con el
    test) y near_train / near_test (id del tuit nuestro más parecido), estas cuatro solo para las filas que
    llegan al paso 3."""
    out = ext.copy().reset_index(drop=True)
    ours = pd.concat([train[["id", "text"]].assign(src="train"), test[["id", "text"]].assign(src="test")],
                     ignore_index=True)
    ours["key"] = ours["text"].map(lambda t: _near_duplicate_key(pp.remove_non_ascii(t)))
    out["key"] = out["text"].map(_near_duplicate_key)
    out["status"], out["match_id"] = "se conserva", pd.NA
    out["sim_train"], out["sim_test"], out["near_train"], out["near_test"] = np.nan, np.nan, pd.NA, pd.NA

    for src in ("test", "train"):  # 1. id
        hit = out["id"].isin(set(ours.loc[ours["src"] == src, "id"])) & (out["status"] == "se conserva")
        out.loc[hit, "status"], out.loc[hit, "match_id"] = f"id-{src}", out.loc[hit, "id"]
    for src in ("test", "train"):  # 2. clave
        key_to_id = dict(zip(ours.loc[ours["src"] == src, "key"], ours.loc[ours["src"] == src, "id"]))
        hit = out["key"].isin(key_to_id) & (out["status"] == "se conserva") & (out["key"] != "")
        out.loc[hit, "status"] = f"clave-{src}"
        out.loc[hit, "match_id"] = out.loc[hit, "key"].map(key_to_id)
    rest = out.index[out["status"] == "se conserva"]  # 3. coseno
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    vec.fit(pd.concat([out.loc[rest, "key"], ours["key"]]))
    A = vec.transform(out.loc[rest, "key"])
    for src in ("test", "train"):
        sub = ours[ours["src"] == src].reset_index(drop=True)
        S = (A @ vec.transform(sub["key"]).T).tocsr()
        mx = np.asarray(S.max(axis=1).todense()).ravel()
        arg = np.asarray(S.argmax(axis=1)).ravel()
        out.loc[rest, f"sim_{src}"] = mx
        out.loc[rest, f"near_{src}"] = sub["id"].to_numpy()[arg]
        hit = (mx >= threshold) & (out.loc[rest, "status"] == "se conserva").to_numpy()
        out.loc[rest[hit], "status"] = f"coseno-{src}"
        out.loc[rest[hit], "match_id"] = sub["id"].to_numpy()[arg[hit]]
    kept = out["status"] == "se conserva"  # 4. duplicados internos
    dup = kept & out["key"].duplicated() & (out["key"] != "")
    out.loc[dup, "status"] = "duplicado-interno"
    return out


# ---------------------------------------------------------------------------
# Entrenamiento con el corpus externo
# ---------------------------------------------------------------------------
def _fresh_workers() -> None:
    """Igual que neural._fresh_workers, sin importar torch: procesos nuevos de joblib (D33)."""
    from joblib.externals.loky import get_reusable_executor

    get_reusable_executor().shutdown(wait=True)


def _stack(X_own, Y_own, X_ext, Y_ext, weight):
    X = pd.concat([pd.Series(X_own).reset_index(drop=True), pd.Series(X_ext).reset_index(drop=True)],
                  ignore_index=True)
    Y = np.vstack([np.asarray(Y_own), np.asarray(Y_ext)]).astype(int)
    w = np.concatenate([np.ones(len(X_own)), np.full(len(X_ext), float(weight))])
    own = np.concatenate([np.ones(len(X_own), bool), np.zeros(len(X_ext), bool)])
    return X, Y, w, own


def _threaded(fn, items, n_threads):
    """map en hilos: liblinear y saga sueltan el GIL, así que n_threads hilos ajustan a la vez (mismos
    resultados que en serie)."""
    if n_threads <= 1:
        return [fn(it) for it in items]
    with ThreadPoolExecutor(n_threads) as pool:
        return list(pool.map(fn, items))


def select_params_augmented(variant, classifier, features, grid, X_own, Y_own, X_ext, Y_ext, weight,
                            n_splits=3, n_threads=1) -> tuple[dict, pd.DataFrame]:
    """Como md.select_params, pero cada fold interno entrena con sus filas propias + todas las externas (con
    peso) y se valida solo con filas propias. Los folds internos son los de md._inner_splits sobre las filas
    propias (estratificados por combinación y agrupando casi duplicados). Las configuraciones de la rejilla
    se ajustan en n_threads hilos."""
    X_own = pd.Series(X_own).reset_index(drop=True)
    Y_own = np.asarray(Y_own)
    inner = {i: np.zeros(Y_own.shape) for i in range(len(grid))}
    for trn, val in md._inner_splits(X_own, Y_own, n_splits):
        X, Y, w, _ = _stack(X_own[trn], Y_own[trn], X_ext, Y_ext, weight)
        featurizer = md.build_featurizer(variant, features).fit(X, Y)
        Xt, Xv = featurizer.transform(X), featurizer.transform(X_own[val])

        def fit_one(params, Xt=Xt, Xv=Xv, Y=Y, w=w):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf = MultiOutputClassifier(md.make_classifier(classifier, params)).fit(Xt, Y, sample_weight=w)
            return np.column_stack([p[:, 1] for p in clf.predict_proba(Xv)])

        for i, p in enumerate(_threaded(fit_one, grid, n_threads)):
            inner[i][val] = p
    scores = pd.DataFrame({"params": [json.dumps(g, sort_keys=True) for g in grid],
                           "macro_interno": [md.tuned_macro(Y_own, inner[i]) for i in range(len(grid))]})
    return grid[int(scores["macro_interno"].idxmax())], scores


def select_label_configs_augmented(X_own, Y_own, X_ext, Y_ext, weight, grid=lw.LABEL_GRID, n_splits=3, n_threads=1):
    """Como lw.select_label_configs (una configuración por etiqueta), con las externas solo en entrenamiento.
    Las 3 etiquetas de cada configuración se ajustan en n_threads hilos."""
    X_own = pd.Series(X_own).reset_index(drop=True)
    Y_own = np.asarray(Y_own)
    inner = {i: np.zeros(Y_own.shape) for i in range(len(grid))}
    for trn, val in md._inner_splits(X_own, Y_own, n_splits):
        X, Y, w, _ = _stack(X_own[trn], Y_own[trn], X_ext, Y_ext, weight)
        for vf in sorted({(g["variant"], g["features"]) for g in grid}):
            featurizer = md.build_featurizer(*vf).fit(X, Y)
            Xt, Xv = featurizer.transform(X), featurizer.transform(X_own[val])
            jobs = [(i, j) for i, g in enumerate(grid) if (g["variant"], g["features"]) == vf for j in range(len(LABELS))]
            fit_one = lambda ij, Xt=Xt, Xv=Xv, Y=Y, w=w: lw._fit_quiet(lw._logreg(grid[ij[0]]["C"]), Xt, Y[:, ij[1]],
                                                                       w).predict_proba(Xv)[:, 1]
            for (i, j), p in zip(jobs, _threaded(fit_one, jobs, n_threads)):
                inner[i][val, j] = p
    scores = pd.DataFrame([{**g, **{lab: lw._best_f1(Y_own[:, j], inner[i][:, j]) for j, lab in enumerate(LABELS)}}
                           for i, g in enumerate(grid)])
    best = {lab: {k: scores.loc[scores[lab].idxmax(), k] for k in ("variant", "features", "C")} for lab in LABELS}
    for lab in LABELS:
        best[lab]["C"] = float(best[lab]["C"])
    return best, scores


def fit_augmented(spec, X_own, Y_own, X_ext, Y_ext, weight, n_threads=2, fixed_epochs=None):
    """Entrena el modelo descrito por spec = (kind, params) con las filas propias + las externas (peso
    `weight`). Devuelve (modelo, info) con lo elegido dentro (hiperparámetros, configuraciones, épocas).

    kind: 'sk' (TweetToxicClassifier(**params)), 'nested' (params = variant, classifier, features, grid; el
    clasificador se elige con select_params_augmented), 'labelwise' (params: grid opcional, por defecto
    lw.LABEL_GRID), 'conditional', 'powerset', 'net'
    (NeuralTweetClassifier(**params)), 'net-text' (red entrenada solo con las filas propias; los textos
    externos solo se usan para preentrenar Word2Vec) y 'trf' (TransformerTweetClassifier(**params)).
    fixed_epochs: épocas fijas para las redes (modelo final, sin validación interna)."""
    kind, params = spec
    X, Y, w, own = _stack(X_own, Y_own, X_ext, Y_ext, weight)
    info = {}
    if kind == "sk":
        model = md.TweetToxicClassifier(**params).fit(X, Y, sample_weight=w)
    elif kind == "nested":
        p = dict(params)
        grid = p.pop("grid")
        best, _ = select_params_augmented(p["variant"], p["classifier"], p["features"], grid, X_own, Y_own, X_ext,
                                          Y_ext, weight, n_threads=n_threads)
        model = md.TweetToxicClassifier(**p, clf_params=best).fit(X, Y, sample_weight=w)
        info["params"] = best
    elif kind == "labelwise":
        configs, _ = select_label_configs_augmented(X_own, Y_own, X_ext, Y_ext, weight, grid=params.get("grid", lw.LABEL_GRID),
                                                    n_threads=n_threads)
        model = lw.LabelwiseClassifier(configs).fit(X, Y, sample_weight=w)
        info["configs"] = configs
    elif kind == "conditional":
        model = lw.ConditionalClassifier().fit(X, Y, sample_weight=w)
    elif kind == "powerset":
        model = lw.PowersetClassifier().fit(X, Y, sample_weight=w)
    elif kind in ("net", "net-text", "trf"):
        import neural as nn_  # solo aquí (D33)

        p = dict(params, n_threads=n_threads, thresholds=0.5, fixed_epochs=fixed_epochs)
        if kind == "trf":
            model = nn_.TransformerTweetClassifier(**p).fit(X, Y, sample_weight=w, own_mask=own)
        elif kind == "net":
            model = nn_.NeuralTweetClassifier(**p).fit(X, Y, sample_weight=w, own_mask=own)
        else:
            model = nn_.NeuralTweetClassifier(**p).fit(pd.Series(X_own).reset_index(drop=True), Y_own,
                                                       pretrain_texts=pd.Series(X_ext).reset_index(drop=True))
        info["best_epochs"] = model.best_epochs_
    else:
        raise ValueError(f"kind desconocido: {kind}")
    return model, info


def _aug_fold(spec, texts, Y, trn, val, X_ext, Y_ext, weight, path, n_threads):
    start = time.perf_counter()
    model, info = fit_augmented(spec, texts[trn], Y[trn], X_ext, Y_ext, weight, n_threads)
    p = model.predict_proba(texts[val])
    info["seconds"] = time.perf_counter() - start
    np.savez(path, proba=p, info=json.dumps(info, default=str))
    return p, info


def _config_hash(spec, weight, ext_ids, folds) -> str:
    payload = json.dumps({"spec": spec, "weight": weight, "ext": [int(i) for i in ext_ids],
                          "folds": np.asarray(folds).tolist()}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def augmented_cv(spec, texts, Y, folds, ext_ids, X_ext, Y_ext, weight, cache_dir, n_jobs=5, n_threads=2) -> dict:
    """OOF (n, 3) en nuestras filas: en el fold k se entrena con nuestras filas de los otros folds + todas las
    externas (peso `weight`) y se predice el fold k. Cada fold se guarda en cache_dir/fold_<k>.npz en cuanto
    termina (se puede interrumpir y reanudar); si la configuración cambia, la caché se borra.
    Devuelve oof, info por fold y el tiempo de reloj de entrenamiento acumulado."""
    cache_dir = Path(cache_dir)
    cfg = _config_hash(spec, weight, ext_ids, folds)
    cfg_file, wall_file = cache_dir / "config.sha256", cache_dir / "wall_seconds.json"
    if cfg_file.exists() and cfg_file.read_text() != cfg:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(cfg)
    texts = pd.Series(texts).reset_index(drop=True)
    folds = np.asarray(folds)
    ks = sorted(np.unique(folds))
    paths = {k: cache_dir / f"fold_{k}.npz" for k in ks}
    missing = [k for k in ks if not paths[k].exists()]
    if missing:
        start = time.perf_counter()
        _fresh_workers()
        Parallel(n_jobs=min(n_jobs, len(missing)))(
            delayed(_aug_fold)(spec, texts, Y, folds != k, folds == k, X_ext, Y_ext, weight, paths[k], n_threads)
            for k in missing)
        _fresh_workers()
        wall = (json.loads(wall_file.read_text())["seconds"] if wall_file.exists() else 0.0) + time.perf_counter() - start
        wall_file.write_text(json.dumps({"seconds": wall}))
    oof, infos = np.zeros(np.asarray(Y).shape), []
    for k in ks:
        d = np.load(paths[k])
        oof[folds == k] = d["proba"]
        infos.append(json.loads(str(d["info"])))
    return {"oof": oof, "info": infos, "fit_seconds": json.loads(wall_file.read_text())["seconds"],
            "cached_folds": len(ks) - len(missing)}


def final_epochs(infos) -> int:
    """Épocas del modelo final de una red = mediana de las mejores épocas de todos los folds y semillas."""
    return max(1, int(np.median([e for i in infos for seed in i["best_epochs"] for e in seed])))


def fit_final_augmented(spec, name, X_own, Y_own, X_ext, Y_ext, weight, X_test, metadata, notebook="11",
                        n_threads=8, fixed_epochs=None):
    """Modelo final (nuestras 1731 filas + las externas), guardado en models/<carpeta del notebook>/<name>.
    Devuelve sus probabilidades en X_test. Se ejecuta en un proceso aparte (D33)."""
    model, info = fit_augmented(spec, X_own, Y_own, X_ext, Y_ext, weight, n_threads, fixed_epochs)
    meta = {"notebook": notebook, "external": {"corpus": HATEVAL_REPO, "revision": HATEVAL_REVISION,
                                               "rows": len(X_ext), "weight": weight, "license": HATEVAL_LICENSE},
            "selected": info, **metadata}
    kind = spec[0]
    if kind in ("sk", "nested"):
        md.save_model(model, name, meta, notebook=notebook)
    elif kind in ("labelwise", "conditional", "powerset"):
        lw.save_labelwise(model, name, meta, notebook=notebook)
    elif kind == "trf":
        import neural as nn_

        nn_.save_transformer_model(model, name, meta, notebook=notebook)
    else:
        import neural as nn_

        nn_.save_neural_model(model, name, meta, notebook=notebook)
    return model.predict_proba(X_test)
