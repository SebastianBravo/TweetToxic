"""Funciones del notebook de Kaggle de TweetToxic (NLP_Grupo3).

Reutiliza los módulos del proyecto (common, modeling, ensemble, external...), que viajan en src/ del dataset de
artefactos. Aquí solo está lo propio del notebook:

- Artifacts: lectura de los artefactos (registro, predicciones OOF y de test, folds, ensambles exportados a JSON).
- Recalcular las cifras de CV desde las OOF guardadas (mismos protocolos que los notebooks 03–11).
- regenerate_submissions(): los 10 envíos seleccionados a partir de las probabilidades de test de los modelos
  base y de los parámetros JSON de cada ensamble (sin deserializar ningún .joblib ni .pt).
- run_full(): reentrenamiento de H5_quote desde los datos crudos (modo FULL) o su versión reducida (SMOKE).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common
import ensemble as en
import modeling as md
from common import LABELS

P_COLS = [f"p_{lab}" for lab in LABELS]
AG = LABELS.index("AG")


# ---------------------------------------------------------------------------
# Datos de la competencia
# ---------------------------------------------------------------------------
def find_competition_files(comp_dir) -> dict:
    """Busca el train (columnas HS, TR y AG), el test y, si existe, el ejemplo de envío en comp_dir."""
    comp_dir = Path(comp_dir)
    files = {}
    for path in sorted(comp_dir.rglob("*.csv")):
        head = path.open("rb").readline().decode("latin-1").strip().lower()
        cols = [c.strip().strip('"') for c in head.replace(",", ";").split(";")]
        if {"hs", "tr", "ag", "text"} <= set(cols):
            files.setdefault("train", path)
        elif "text" in cols and "id" in cols:
            files.setdefault("test", path)
        elif {"id", "hs", "tr", "ag"} <= set(cols):
            files.setdefault("sample", path)
    missing = {"train", "test"} - set(files)
    if missing:
        raise FileNotFoundError(f"No se encontraron los archivos {missing} de la competencia en {comp_dir}")
    return files


def load_competition(comp_dir):
    """train y test con el lector del proyecto (separador ';', UTF-8 o Latin-1, id con comillas corregido)."""
    files = find_competition_files(comp_dir)
    train = common.load_train(files["train"])
    test = common._read_csv(files["test"])
    return train, test, files


# ---------------------------------------------------------------------------
# Artefactos
# ---------------------------------------------------------------------------
class Artifacts:
    """Acceso a los artefactos del dataset (art_dir) alineados con el train y el test."""

    def __init__(self, art_dir, train, test):
        self.dir = Path(art_dir)
        self.train, self.test = train, test
        self.registry = pd.read_csv(self.dir / "models" / "registry.csv").set_index("model_id")
        self._oof = pd.read_parquet(self.dir / "models" / "oof_predictions.parquet")
        self._oof2 = pd.read_parquet(self.dir / "models" / "oof_folds_seed2.parquet")
        self._test = pd.concat([pd.read_parquet(self.dir / "models" / "test_predictions.parquet"),
                                pd.read_parquet(self.dir / "models" / "test_predictions_foldbag.parquet")])
        self.folds = {fs: pd.read_csv(self.dir / "data" / "processed" / f"{fs}.csv").set_index("id").loc[train["id"], "fold"].to_numpy()
                      for fs in ("folds", "folds_seed2")}
        ens = [json.loads(p.read_text()) for p in (self.dir / "ensembles").glob("*.json")]
        self.ensembles = {E["tag"]: E for E in sorted(ens, key=lambda E: E["order"])}
        path = self.dir / "models" / "ensemble_crossfit.parquet"
        self.crossfit = pd.read_parquet(path) if path.exists() else None

    def oof(self, model_id, foldset="folds") -> np.ndarray:
        store = self._oof if foldset == "folds" else self._oof2
        df = store[store["model_id"] == model_id]
        if df.empty:
            raise KeyError(f"No hay OOF de {model_id} en {foldset}")
        return df.set_index("id").loc[self.train["id"], P_COLS].to_numpy()

    def has_oof(self, model_id, foldset="folds") -> bool:
        store = self._oof if foldset == "folds" else self._oof2
        return bool((store["model_id"] == model_id).any())

    def test_proba(self, model_id) -> np.ndarray:
        df = self._test[self._test["model_id"] == model_id]
        if df.empty:
            raise KeyError(f"No hay probabilidades de test de {model_id}")
        return df.set_index("id").loc[self.test["id"], P_COLS].to_numpy()


# ---------------------------------------------------------------------------
# Protocolos de evaluación
# ---------------------------------------------------------------------------
def f1_registry_protocol(art: Artifacts, model_id, Y, foldset="folds") -> dict:
    """F1 con el protocolo con el que se registró el modelo:
    - notebooks 03–06: umbral que maximiza el F1 en las OOF de los otros folds (argmax) y jerarquía;
    - notebooks 07–11: protocolo del 07 (crossfit: umbral de meseta elegido sin mirar el fold y jerarquía).
    Devuelve la media por fold de F1_HS, F1_TR, F1_AG, F1_macro y F1_micro."""
    p, f = art.oof(model_id, foldset), art.folds[foldset]
    if int(art.registry.loc[model_id, "notebook"]) <= 6:
        _, tuned = md._summarize_folds(p, Y, pd.Series(f))
        return tuned.mean().to_dict()
    r = en.crossfit(en.Single(0), p[:, None, :], Y, f, md.empty_mask(art.train["text"]))
    return r["folds"][("plateau", True)].mean().to_dict()


def crossfit_rule(art: Artifacts, ids, method, Y, foldset="folds", rule=True):
    """Protocolo del 07 para un ensamble (o un modelo con method='single') y, con rule, la regla de comillas
    en las decisiones fuera de fold (como en el 08–11). Devuelve (resultado del crossfit, métricas por fold)."""
    P = np.stack([art.oof(m, foldset) for m in ids], axis=1)
    comb = en.Single(0) if method == "single" else en.make_combiner(method, "platt")
    f = art.folds[foldset]
    r = en.crossfit(comb, P, Y, f, md.empty_mask(art.train["text"]))
    pred = r["pred"].copy()
    if rule:
        pred[md.quoted_mask(art.train["text"]), AG] = 1
    per_fold = pd.DataFrame([en.metrics(Y[f == k], pred[f == k]) for k in sorted(np.unique(f))])
    return r, per_fold, pred


def cv_from_crossfit(art: Artifacts, tag, foldset="folds", rule=None) -> pd.DataFrame:
    """Métricas por fold de un ensamble a partir de su crossfit guardado (probabilidades fuera de fold del
    combinador y umbral de meseta elegido para cada fold sin mirarlo): solo umbrales, jerarquía, tuits vacíos y
    regla de comillas, con numpy. No depende de la versión de scikit-learn (el reajuste del stacker sí).
    rule: None = la del ensamble; False = sin la regla de comillas (p. ej. FINAL_A a partir de FINAL_A_quote)."""
    E = art.ensembles[tag]
    cf = art.crossfit[(art.crossfit["tag"] == tag) & (art.crossfit["foldset"] == foldset)].set_index("id").loc[art.train["id"]]
    proba, f = cf[P_COLS].to_numpy(), art.folds[foldset]
    assert (cf["fold"].to_numpy() == f).all()
    Y = art.train[LABELS].to_numpy()
    empty, quote = md.empty_mask(art.train["text"]), md.quoted_mask(art.train["text"])
    rows = []
    for k in sorted(np.unique(f)):
        v = f == k
        use_rule = E["quote_rule"] if rule is None else rule
        pred = en.decide(proba[v], E["crossfit_thresholds"][foldset][str(k)], E["hierarchy"], empty[v],
                         quote[v] if use_rule else None)
        rows.append(en.metrics(Y[v], pred))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Envíos
# ---------------------------------------------------------------------------
def regenerate_submissions(art: Artifacts, out_dir) -> pd.DataFrame:
    """Regenera los 10 envíos seleccionados desde las probabilidades de test de sus modelos base y los
    parámetros JSON de cada ensamble, los escribe en out_dir y los compara celda a celda con los enviados."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, E in art.ensembles.items():
        P = np.stack([art.test_proba(m) for m in E["base_ids"]], axis=1)
        pred = en.decide_from_dict(E, P, art.test["text"])
        sub = pd.DataFrame({"id": art.test["id"].to_numpy(), **{lab: pred[:, j].astype(int) for j, lab in enumerate(LABELS)}})
        sub.to_csv(out_dir / E["file"], index=False)
        sent = pd.read_csv(art.dir / "submissions" / E["file"])
        rows.append({"envío": tag, "archivo": E["file"], "filas": len(sub),
                     "ids en el orden del test": bool((sub["id"].to_numpy() == sent["id"].to_numpy()).all()),
                     "celdas distintas del enviado": int((sub[LABELS].to_numpy() != sent[LABELS].to_numpy()).sum()),
                     "idéntico al enviado": bool(sub.equals(sent))})
    return pd.DataFrame(rows).set_index("envío")


# ---------------------------------------------------------------------------
# Modo FULL / SMOKE: reentrenar H5_quote desde los datos crudos
# ---------------------------------------------------------------------------
TORCH_KINDS = ("net", "net-text", "trf")


def _batch_of(spec):
    if spec[0] in TORCH_KINDS:
        return "redes"
    return "xgboost" if spec[1].get("classifier") in ("xgb", "lgbm") else "scikit-learn"


def _smoke_spec(spec):
    kind, params = spec[0], dict(spec[1])
    if kind == "nested":
        params["grid"] = params["grid"][2:4]
    elif kind == "labelwise":
        params["grid"] = [{"variant": "V2", "features": "tfidf", "C": 10.0}, {"variant": "V1", "features": "char", "C": 10.0}]
    elif kind in TORCH_KINDS:
        params.update(n_seeds=1, max_epochs=2, patience=1)
    return [kind, params]


def _final_job(name, spec, X_own, Y_own, X_ext, Y_ext, weight, X_test, epochs, threads):
    import external as ex_  # torch solo se importa en el proceso de las redes (no convive con xgboost)

    return ex_.fit_final_augmented(spec, name, X_own, Y_own, X_ext, Y_ext, weight, X_test,
                                   {"origin": "notebook de Kaggle (modo FULL/SMOKE)"}, notebook="11",
                                   n_threads=threads, fixed_epochs=epochs)


def run_full(mode, art: Artifacts, work_dir, hateval_local=None, n_jobs=None, log=print) -> dict:
    """Reentrena H5_quote (pool v2 con HatEval a peso 0.5 y combinador por pesos) desde los datos crudos con
    las funciones del proyecto:

    1. HatEval: descarga de la revisión fijada (token HF_TOKEN de Kaggle Secrets) y sha256 contra el manifest del
       11; o, solo en pruebas locales, una copia local (hateval_local);
    2. borrado de no ASCII y deduplicación estricta (id, clave, coseno ≥ 0.85) contra el train y el test de la competencia;
    3. los 14 modelos base: CV en folds.csv con HatEval solo en entrenamiento (OOF) y modelo final con todo;
    4. combinador por pesos, umbrales de meseta, jerarquía, tuits vacíos y regla de comillas;
    5. comparación con el CSV enviado.

    SMOKE: el mismo camino con 1 fold, 1 semilla, rejillas mínimas y una submuestra (para comprobar que corre)."""
    import external as ex

    assert mode in ("FULL", "SMOKE")
    smoke = mode == "SMOKE"
    start = time.perf_counter()
    work_dir = Path(work_dir)
    train, test = art.train, art.test
    full = json.loads((art.dir / "full" / "h5_specs.json").read_text())
    manifest = json.loads((art.dir / "hateval_manifest.json").read_text())
    n_jobs = n_jobs or max(1, min(5, os.cpu_count() or 2))
    threads = max(1, (os.cpu_count() or 2) // n_jobs)

    # 1. HatEval
    he_dir = work_dir / "hateval"
    he_dir.mkdir(parents=True, exist_ok=True)
    if hateval_local is not None:
        log("HatEval: copia local (solo para la prueba fuera de Kaggle)")
        for s in ex.SPLITS:
            shutil.copyfile(Path(hateval_local) / f"{s}.parquet", he_dir / f"{s}.parquet")
    else:
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("El modo FULL/SMOKE necesita el token de Hugging Face: añade el secreto HF_TOKEN en "
                               "Kaggle (Add-ons > Secrets), activa Internet y acepta los términos de "
                               "https://huggingface.co/datasets/valeriobasile/HatEval. HatEval no se incluye en el dataset.")
        ex.download_hateval(manifest["revision"], he_dir)
    for s, f in manifest["files"].items():
        got = ex.sha256(he_dir / f"{s}.parquet")
        assert got == f["sha256"], f"HatEval {s}: sha256 distinto del manifest del 11"
    log(f"HatEval verificado: revisión {manifest['revision'][:12]}…, sha256 de los 3 splits = manifest")

    # 2. Deduplicación (sin etiquetas) y etiquetas solo de los conservados
    import preprocessing as pp

    H = ex.read_texts(he_dir)
    H["text"] = H["text"].map(pp.remove_non_ascii)  # el mismo borrado de no ASCII que sufrió el train
    D = ex.deduplicate(H[["id", "text", "split"]], train, test)
    keep = D[D["status"] == "se conserva"].reset_index(drop=True)
    labels = ex.read_labels(keep["id"], he_dir)
    log(f"Deduplicación: {dict(D['status'].value_counts())}")
    if not smoke:
        assert len(keep) == full["expected_kept"], f"se conservan {len(keep)} tuits, el 11 conservó {full['expected_kept']}"

    texts, Y = train["text"].reset_index(drop=True), train[LABELS].to_numpy()
    folds = art.folds["folds"]
    X_ext, Y_ext = keep["text"], labels[LABELS].to_numpy().astype(int)
    folds_to_run, ext_ids = None, keep["id"].to_numpy()
    if smoke:  # submuestra estratificada: 2 folds del train, 300 tuits de HatEval, solo el fold 0 se valida
        own = np.where(np.isin(folds, [0, 1]))[0]
        texts, Y, folds = texts.iloc[own].reset_index(drop=True), Y[own], folds[own]
        sub = np.random.default_rng(common.SEED).choice(len(X_ext), size=min(300, len(X_ext)), replace=False)
        X_ext, Y_ext, ext_ids = X_ext.iloc[np.sort(sub)].reset_index(drop=True), Y_ext[np.sort(sub)], ext_ids[np.sort(sub)]
        folds_to_run = [0]
    log(f"Entrenamiento: {len(texts)} tuits del train + {len(X_ext)} de HatEval (peso {full['weight']}) | "
        f"{n_jobs} procesos × {threads} hilos")

    # 3. CV (OOF) de los 14 modelos base, por tandas de procesos (torch y xgboost no comparten proceso)
    specs = {name: (_smoke_spec(spec) if smoke else spec) for name, spec in full["specs"].items()}
    cache = work_dir / "cache"
    oof, infos = {}, {}
    order = sorted(specs, key=lambda n: ("scikit-learn", "xgboost", "redes").index(_batch_of(specs[n])))
    for name in order:
        t0 = time.perf_counter()
        res = ex.augmented_cv(specs[name], texts, Y, folds, ext_ids, X_ext, Y_ext, full["weight"], cache / name,
                              n_jobs=n_jobs, n_threads=threads, folds_to_run=folds_to_run)
        oof[name], infos[name] = res["oof"], res["info"]
        log(f"  CV {name}: {time.perf_counter() - t0:.0f} s")

    # Modelos finales (todo el train + HatEval) y sus probabilidades de test
    from joblib import Parallel, delayed

    P_test = {}
    for batch in ("scikit-learn", "xgboost", "redes"):
        names = [n for n in order if _batch_of(specs[n]) == batch]
        if not names:
            continue
        t0 = time.perf_counter()
        ex._fresh_workers()
        out = Parallel(n_jobs=n_jobs)(
            delayed(_final_job)(n, specs[n], texts, Y, X_ext, Y_ext, full["weight"], test["text"],
                                ex.final_epochs(infos[n]) if specs[n][0] in TORCH_KINDS else None, threads)
            for n in names)
        ex._fresh_workers()
        P_test.update(zip(names, out))
        log(f"  finales ({batch}): {len(names)} modelos, {time.perf_counter() - t0:.0f} s")

    # 4. Combinador (pesos >= 0 por log-loss sobre las OOF), umbrales y reglas
    ids = full["base_ids"]
    rows = folds == 0 if smoke else np.ones(len(Y), bool)
    P = np.stack([oof[m] for m in ids], axis=1)[rows]
    Yr, fr, txt = Y[rows], folds[rows], texts[rows].reset_index(drop=True)
    if smoke:  # con un solo fold validado no hay crossfit: 3 grupos al azar solo para recorrer el código
        fr = np.random.default_rng(common.SEED).integers(0, 3, size=len(Yr))
    r = en.crossfit(en.make_combiner("weights", "platt"), P, Yr, fr, md.empty_mask(txt))
    pred_oof = r["pred"].copy()
    pred_oof[md.quoted_mask(txt), AG] = 1
    cv = np.mean([en.metrics(Yr[fr == k], pred_oof[fr == k])["F1_macro"] for k in np.unique(fr)])
    fitted, thr = en.fit_final(en.make_combiner("weights", "platt"), P, Yr, fr, r["proba"])
    P_te = np.stack([P_test[m] for m in ids], axis=1)
    pred = en.decide(fitted.predict(P_te), thr, True, md.empty_mask(test["text"]), md.quoted_mask(test["text"]))
    sub = pd.DataFrame({"id": test["id"].to_numpy(), **{lab: pred[:, j].astype(int) for j, lab in enumerate(LABELS)}})
    out_path = work_dir / f"{mode}_H5_quote_reentrenado.csv"
    sub.to_csv(out_path, index=False)

    # 5. Comparación con el envío original
    sent = pd.read_csv(art.dir / "submissions" / art.ensembles["H5_quote"]["file"])
    same = (sub[LABELS].to_numpy() == sent[LABELS].to_numpy())
    result = {"modo": mode, "tuits HatEval conservados": len(keep), "F1 macro CV (con la regla)": round(float(cv), 4),
              "umbrales": thr, "% celdas idénticas al envío H5_quote": round(100 * same.mean(), 2),
              "tuits idénticos": int(same.all(axis=1).sum()), "minutos": round((time.perf_counter() - start) / 60, 1),
              "archivo": out_path.name}
    log(json.dumps(result, ensure_ascii=False, default=str))
    return result
