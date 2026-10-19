"""Construye kaggle/dataset/: el dataset privado de Kaggle "tweettoxic-nlp-grupo3-artifacts".

Contiene todo lo que necesita el notebook de Kaggle además de los datos de la competencia: los módulos del
proyecto (src/), el registro y las predicciones OOF y de test, los folds, los 10 envíos seleccionados, sus
ensambles exportados a JSON, tablas y figuras, y el manifest de HatEval (sin datos de HatEval). No incluye
ningún .joblib ni .pt ni data/external/.

Comprueba antes de escribir:
- que cada ensamble exportado a JSON da exactamente las mismas probabilidades que el objeto original
  (diferencia 0.0, en las OOF y en el test);
- que las decisiones regeneradas desde el JSON coinciden con el CSV enviado;
- que la CV recalculada desde las OOF coincide con el registro (o con el notebook 08) a 1e-4.

Uso (desde la raíz del repositorio): uv run python kaggle/build_dataset.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "kaggle"))

import common  # noqa: E402
import ensemble as en  # noqa: E402
import kaggle_support as ks  # noqa: E402
import modeling as md  # noqa: E402
from common import LABELS, PROCESSED_DIR  # noqa: E402

OUT = ROOT / "kaggle" / "dataset"
MODELS = ROOT / "models"
SRC_MODULES = ["common", "preprocessing", "modeling", "ensemble", "neural", "handcrafted", "embeddings", "external",
               "labelwise"]
P_COLS = ks.P_COLS

CORE = ["V2__logreg__nested", "V1__logreg__nested", "V3__linsvm__nested"]
CLASSIC = ["V3__et__word__k2000", "V3__xgb__word__k500", "V3__nbsvm", "V3__mnb__word"]
NETS = ["V1__nn_cnn_char__tune-d0.3-lr0.002", "V1__nn_cnn_word__w2v", "V3__nn_mlp__tune-d0.5-lr0.002"]
POOL_D = CORE + CLASSIC + NETS
S10 = {NETS[0]: NETS[0] + "__s10", NETS[1]: NETS[1] + "__s10"}
NEW_09 = ["V2__labelwise", "V2__conditional", "V2__powerset", "V3__nn_cnn_char__tune-d0.3-lr0.002__s10"]
POOL_V2 = [S10.get(m, m) for m in POOL_D] + NEW_09
GRIDS_04 = {
    "logreg": [{"C": c} for c in (0.3, 1.0, 3.0, 10.0, 30.0)] + [{"C": c, "l1_ratio": 0.5} for c in (1.0, 3.0, 10.0)]
              + [{"C": c, "l1_ratio": 1.0} for c in (3.0, 10.0)],
    "linsvm": [{"C": c} for c in (0.03, 0.1, 0.3, 1.0, 3.0)],
}

# Los 10 envíos seleccionados, en el orden de la lista final, con su puntaje público
SELECTED = [
    ("H5_quote", "ens11__v2he05__weights__quote", "14 modelos + HatEval (peso 0.5) · pesos ≥ 0", 0.80331),
    ("H2_quote", "ens11__v2he05__stack-all__quote", "14 modelos + HatEval (peso 0.5) · stacking", 0.80706),
    ("H6_quote", "ens11__v2he05-new__stack-all__quote", "H2 + Transformer y CNN con Word2Vec ampliado · stacking", 0.80856),
    ("H4_quote", "ens11__mix05__stack-all__quote", "14 modelos sin y con HatEval (28) · stacking", 0.80562),
    ("H1_quote", "ens11__v2he10__stack-all__quote", "14 modelos + HatEval (peso 1.0) · stacking", 0.81235),
    ("A2_quote", "ens10__v2__stack-all__quote", "14 modelos sin HatEval · stacking", 0.78783),
    ("FINAL_A_quote", "ens08__D__stack-all__quote", "ensamble original (10 modelos) · stacking", 0.78598),
    ("H3_quote", "ens11__Dhe05__stack-all__quote", "10 modelos del ensamble original + HatEval (peso 0.5) · stacking", 0.80207),
    ("D_s10_quote", "ens10__Ds10__stack-all__quote", "ensamble original con CNN de 10 semillas · stacking", 0.79035),
    ("A_foldbag_quote", "ens10__D__stack-all__foldbag__quote", "ensamble original con modelos de fold promediados en el test", 0.78454),
]
# Figuras de los notebooks 01–09 que muestra el notebook de Kaggle
FIGURES = ['01_label_distribution.png', '01_label_phi.png', '01_length_by_label.png', '01_lexicon_rates.png', '01_log_odds_unigrams.png', '01_pos_by_label.png', '01_surface_features_by_label.png', '01_train_vs_test_length.png', '02_ablation.png', '02_variants_f1.png', '03_f1_heatmap.png', '04_error_diversity.png', '04_finalists_nested.png', '04_screening.png', '05_diversity.png', '05_learning_curves.png', '05_train_vs_validation.png', '06_learning_curves.png', '06_stage1.png', '07_ensemble_grid.png', '07_error_correlation.png', '07_families.png', '07_reliability_pool.png', '07_threshold_plateau.png', '09_learning_curve.png']
# CV de FINAL_A_quote (no está en el registro): notebook 08
FINAL_A_QUOTE_CV = {"folds": {"F1_macro": 0.8060}, "folds_seed2": {"F1_macro": 0.8019}}
# Tabla del 02: regresión logística con TF-IDF de palabras + caracteres, umbral 0.5
VARIANTS_02 = [("V0", "texto crudo", 0.741, 0.767, 0.725, 0.744, 0.020),
               ("V1", "limpieza + minúsculas (referencia)", 0.761, 0.800, 0.731, 0.764, 0.027),
               ("V2", "V1 + alargamientos a 2, risas, censura", 0.759, 0.797, 0.730, 0.762, 0.029),
               ("V2 (alarg.=1)", "V2 con alargamientos a 1", 0.760, 0.796, 0.731, 0.762, 0.030),
               ("V3", "V2 + restauración de palabras dañadas", 0.764, 0.797, 0.732, 0.764, 0.028),
               ("V4", "V3 + lemas (spaCy) como vista extra", 0.760, 0.803, 0.740, 0.768, 0.030)]


def he(m, tag):
    return f"{m.replace('__s10', '')}__he{tag}"


def collect_seed2(train) -> pd.DataFrame:
    """OOF de folds_seed2 de las cachés del 07, 09 y 11 (no están en oof_predictions.parquet)."""
    folds2 = pd.read_csv(PROCESSED_DIR / "folds_seed2.csv").set_index("id").loc[train["id"], "fold"].to_numpy()
    arrays = {}
    for cache in (MODELS / "07_ensamble" / "cache", MODELS / "09_mejoras" / "cache"):
        for f in sorted(cache.glob("*__folds_seed2.npy")):
            arrays[f.name[: -len("__folds_seed2.npy")]] = np.load(f)
    for d in sorted((MODELS / "11_hateval" / "cache").glob("*__folds_seed2")):
        oof = np.full((len(train), 3), np.nan)
        for k in range(5):
            oof[folds2 == k] = np.load(d / f"fold_{k}.npz")["proba"]
        arrays[d.name[: -len("__folds_seed2")]] = oof
    frames = []
    for m, a in arrays.items():
        assert a.shape == (len(train), 3) and not np.isnan(a).any(), m
        frames.append(pd.DataFrame({"model_id": m, "id": train["id"].to_numpy(), "fold": folds2,
                                    **{c: a[:, j] for j, c in enumerate(P_COLS)}}))
    return pd.concat(frames, ignore_index=True)


def collect_foldbag(test) -> pd.DataFrame:
    frames = []
    for f in sorted((MODELS / "10_candidatos" / "cache").glob("*__foldbag__test.npy")):
        a = np.load(f)
        frames.append(pd.DataFrame({"model_id": f.name[: -len("__test.npy")], "id": test["id"].to_numpy(),
                                    **{c: a[:, j] for j, c in enumerate(P_COLS)}}))
    return pd.concat(frames, ignore_index=True)


def full_specs(reg) -> dict:
    """Configuración de los 14 modelos base de H5_quote (como spec_of del 11) para el modo FULL."""
    def net_params(mid, **changes):
        params = md.load_model(mid).get_params()
        for k in ("n_threads", "thresholds", "fixed_epochs", "log_path"):
            params.pop(k, None)
        return dict(params, **changes)

    specs = {}
    for m in POOL_V2:
        base = m.replace("__s10", "")
        if base in CORE:
            r = reg.loc[base]
            spec = ["nested", dict(variant=r["variant"], classifier=r["classifier"], features=r["features"],
                                   grid=GRIDS_04[r["classifier"]])]
        elif base in CLASSIC:
            r = reg.loc[base]
            spec = ["sk", dict(variant=r["variant"], classifier=r["classifier"], features=r["features"],
                               clf_params=json.loads(r["params"]) or None)]
        elif base in ("V2__labelwise", "V2__conditional", "V2__powerset"):
            spec = [base.split("__")[1], {}]
        elif base == "V3__nn_cnn_char__tune-d0.3-lr0.002":
            spec = ["net", net_params(NETS[0], variant="V3", n_seeds=3)]
        else:
            spec = ["net", net_params(base, n_seeds=3)]
        specs[he(m, "05")] = spec
    return {"weight": 0.5, "base_ids": [he(m, "05") for m in POOL_V2], "specs": specs, "expected_kept": 3621,
            "combiner": "weights", "note": "Pool v2 del 09 reentrenado con HatEval a peso 0.5 (notebook 11); "
                                           "las CNN con 3 semillas; C y configuración por etiqueta por CV anidada."}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in ("src", "models", "data/processed", "ensembles", "submissions", "reports/tables", "reports/figures", "full"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    train, test = common.load_train(), common.load_test()
    Y = train[LABELS].to_numpy()
    reg = pd.read_csv(MODELS / "registry.csv").set_index("model_id")

    for m in SRC_MODULES:
        shutil.copy(ROOT / "notebooks" / f"{m}.py", OUT / "src" / f"{m}.py")
    shutil.copy(ROOT / "kaggle" / "kaggle_support.py", OUT / "src" / "kaggle_support.py")
    for f in ("registry.csv", "oof_predictions.parquet", "test_predictions.parquet"):
        shutil.copy(MODELS / f, OUT / "models" / f)
    collect_seed2(train).to_parquet(OUT / "models" / "oof_folds_seed2.parquet", index=False)
    collect_foldbag(test).to_parquet(OUT / "models" / "test_predictions_foldbag.parquet", index=False)
    for f in ("folds.csv", "folds_seed2.csv"):
        shutil.copy(PROCESSED_DIR / f, OUT / "data" / "processed" / f)
    for f in ("resultados_finales.csv", "resultados_finales.md"):
        shutil.copy(ROOT / "reports" / "tables" / f, OUT / "reports" / "tables" / f)
    pd.DataFrame(VARIANTS_02, columns=["variante", "pasos", "F1 HS", "F1 TR", "F1 AG", "F1 macro", "desv."]).to_csv(
        OUT / "reports" / "tables" / "variantes_02.csv", index=False)
    for f in FIGURES:
        shutil.copy(ROOT / "reports" / "figures" / f, OUT / "reports" / "figures" / f)
    shutil.copy(ROOT / "data" / "external" / "hateval" / "manifest.json", OUT / "hateval_manifest.json")
    (OUT / "full" / "h5_specs.json").write_text(json.dumps(full_specs(reg), indent=1, ensure_ascii=False, default=str))

    # Ensambles: exportación, comprobaciones y CV esperada
    art_tmp = None
    for i, (tag, name, desc, lb) in enumerate(SELECTED):
        model = md.load_model(name)
        params = json.loads(json.dumps(en.ensemble_to_dict(model)))
        file = next(p.name for p in (ROOT / "submissions").glob(f"{tag}_ens*.csv"))
        shutil.copy(ROOT / "submissions" / file, OUT / "submissions" / file)
        meta = json.loads(md.find_model_path(name).with_suffix(".json").read_text())
        cv_ids = POOL_D if tag == "A_foldbag_quote" else params["base_ids"]
        params.update(tag=tag, order=i + 1, file=file, description=desc, public_lb=lb, method=meta["method"], cv_ids=cv_ids)
        (OUT / "ensembles" / f"{tag}.json").write_text(json.dumps(params, indent=1, ensure_ascii=False))
    art = ks.Artifacts(OUT, train, test)
    crossfit_frames = []
    for tag, E in art.ensembles.items():
        model = md.load_model(E["name"])
        P = np.stack([art.test_proba(m) for m in E["base_ids"]], axis=1)
        diff = float(np.abs(model.predict_proba_from_base(P) - en.combiner_from_dict(E["combiner"]).predict(P)).max())
        assert diff == 0.0, (tag, diff)
        expected, thresholds = {}, {}
        for fs in ("folds", "folds_seed2"):
            r, per_fold, _ = ks.crossfit_rule(art, E["cv_ids"], E["method"], Y, fs, rule=True)
            expected[fs] = {k: round(float(v), 6) for k, v in per_fold.mean().items()}
            thresholds[fs] = {str(k): v for k, v in r["thresholds"]["plateau"].items()}
            crossfit_frames.append(pd.DataFrame({"tag": tag, "foldset": fs, "id": train["id"].to_numpy(),
                                                 "fold": art.folds[fs], **{c: r["proba"][:, j] for j, c in enumerate(P_COLS)}}))
        if E["name"] in reg.index:  # folds.csv: el registro guarda las métricas con la regla (10 y 11)
            for k in ("F1_HS", "F1_TR", "F1_AG", "F1_macro", "F1_micro"):
                assert abs(expected["folds"][k] - reg.loc[E["name"], k]) <= 1e-4, (tag, k)
            meta = json.loads(md.find_model_path(E["name"]).with_suffix(".json").read_text())
            assert abs(expected["folds_seed2"]["F1_macro"] - meta["cv_F1_macro_with_rule"]["folds_seed2"]) <= 1e-4, tag
            source = "models/registry.csv (folds.csv) y el .json del ensamble (folds_seed2)"
        else:
            for fs in ("folds", "folds_seed2"):
                assert abs(expected[fs]["F1_macro"] - FINAL_A_QUOTE_CV[fs]["F1_macro"]) <= 1e-4, (tag, fs)
            source = "notebook 08 (FINAL_A_quote no está en el registro)"
        E.update(expected_cv=expected, expected_source=source, crossfit_thresholds=thresholds)
        (OUT / "ensembles" / f"{tag}.json").write_text(json.dumps(E, indent=1, ensure_ascii=False))
        print(f"{tag:16s} JSON = objeto (0.0) | CV {expected['folds']['F1_macro']:.4f} / {expected['folds_seed2']['F1_macro']:.4f}")
    pd.concat(crossfit_frames, ignore_index=True).to_parquet(OUT / "models" / "ensemble_crossfit.parquet", index=False)
    art = ks.Artifacts(OUT, train, test)
    for tag in art.ensembles:  # la CV desde el crossfit guardado reproduce la esperada (numpy, sin reajustar)
        for fs in ("folds", "folds_seed2"):
            got = ks.cv_from_crossfit(art, tag, fs).mean()
            assert all(abs(got[k] - art.ensembles[tag]["expected_cv"][fs][k]) <= 1e-6 for k in got.index), (tag, fs)
    check = ks.regenerate_submissions(art, ROOT / "kaggle" / "working" / "check_submissions")
    assert check["idéntico al enviado"].all(), check
    shutil.rmtree(ROOT / "kaggle" / "working" / "check_submissions")
    print("Los 10 CSV regenerados desde los JSON son idénticos a los enviados")

    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "tweettoxic-nlp-grupo3-artifacts",
        "id": "TU_USUARIO_KAGGLE/tweettoxic-nlp-grupo3-artifacts",
        "subtitle": "Artefactos del notebook de TweetToxic (NLP_Grupo3): OOF, ensambles en JSON y codigo",
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
        "description": ("Artefactos para reproducir el notebook de Kaggle de TweetToxic (NLP_Grupo3): modulos del "
                        "proyecto (src/), registro de modelos, predicciones OOF y de test, folds, los 10 envios "
                        "seleccionados y sus ensambles exportados a JSON. No contiene datos de HatEval (solo su "
                        "manifest) ni modelos serializados (.joblib/.pt)."),
    }, indent=2, ensure_ascii=False))
    for bad in ("*.joblib", "*.pt"):
        assert not list(OUT.rglob(bad)), bad
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"Dataset: {sum(1 for p in OUT.rglob('*') if p.is_file())} archivos, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
