# TweetToxic

Competencia de Kaggle **TweetToxic** (Maestría en IA, Pontificia Universidad Javeriana): clasificación multietiqueta de tuits en español.

| Etiqueta | Significado |
|---|---|
| **HS** | Discurso de odio |
| **TR** | Dirigido contra una persona concreta (1) o contra un grupo (0) |
| **AG** | Agresividad |

**Métrica:** F1 macro por columnas, es decir, la media del F1 de HS, TR y AG.

**Reglas de la competencia:** no se permite ningún modelo preentrenado (de lenguaje, embeddings, clasificadores ni transformers). Todo se entrena desde cero en este repositorio.

## Entorno

Se usa [uv](https://docs.astral.sh/uv/) con Python 3.14:

```bash
uv sync                      # instala las dependencias fijadas en uv.lock
uv run jupyter lab           # o abrir los notebooks desde VS Code con el kernel de .venv
```

En macOS, XGBoost y LightGBM necesitan OpenMP: `brew install libomp`.

## Estructura

- **`data/`:** los CSV de la competencia (`toxic_train.csv`, `toxic_test.csv`), que no se modifican nunca.
  - `data/processed/`: derivados, como los folds compartidos y los textos preprocesados.
  - `data/external/`: corpus externos; no se versiona, ver abajo.
- **`notebooks/`:** los notebooks numerados del `01` al `11`, que se ejecutan en orden, y los módulos que comparten (`common.py`, `preprocessing.py`, `modeling.py`, `neural.py`, `ensemble.py`, `labelwise.py`, `external.py`, …).
- **`models/`:** una carpeta por notebook con los modelos guardados.
  - Se versionan solo los `.json` de metadatos, `registry.csv` y los parquet de predicciones.
  - Los `.joblib`/`.pt` se regeneran ejecutando los notebooks.
- **`submissions/`:** los CSV enviados a Kaggle.
- **`reports/`:** figuras y tablas.

## Notebook de Kaggle

`kaggle/notebook/tweettoxic-nlp-grupo3.ipynb` es el notebook que se sube a Kaggle: resume el proyecto, recalcula las cifras de validación desde las predicciones guardadas y regenera los 10 envíos seleccionados.

- **Dataset de artefactos:** necesita el dataset privado `tweettoxic-nlp-grupo3-artifacts`, que se genera con `uv run python kaggle/build_dataset.py` en `kaggle/dataset/`.
- **Para ejecutarlo en Kaggle:** se crea el dataset privado con el contenido de `kaggle/dataset/` y se añade como input del notebook, junto con la competencia `tweet-toxic-2026-puj`.

## Datos externos: HatEval (notebook 11)

El notebook `11_hateval.ipynb` usa la parte en español de **HatEval** como corpus etiquetado adicional, con autorización del profesor (2026-10-15).

- **Deduplicación:** se deduplica estrictamente contra nuestro train y nuestro test.
- **Tuits que coinciden con el test:** se descartan sin leer sus etiquetas.

**Cómo obtenerlo.** El dataset es de acceso restringido en Hugging Face:

1. Crear una cuenta en Hugging Face y **aceptar los términos** del dataset en <https://huggingface.co/datasets/valeriobasile/HatEval>.
2. Iniciar sesión en la máquina: `uv run hf auth login` (en versiones antiguas de `huggingface_hub`, `huggingface-cli login`). El token queda guardado localmente; no hace falta escribirlo en ningún archivo del proyecto.
3. Ejecutar la sección 1 del notebook 11 (`external.download_hateval()`). Esta sección:
   - descarga la revisión fija `d170b5cb1844f3614493fce7c7eaaae087a5201c`;
   - guarda una copia **sin modificar** en `data/external/hateval/` (`train.parquet`, `dev.parquet`, `test.parquet`);
   - escribe un `manifest.json` con la revisión, la fecha, las filas y el sha256 de cada archivo.

   Si los archivos ya existen y su sha256 coincide, no descarga nada.

`data/external/` está en `.gitignore`: el corpus no se redistribuye con este repositorio.

**Cita y licencia (CC BY-NC 4.0):**

> Basile, V., Bosco, C., Fersini, E., Nozza, D., Patti, V., Rangel Pardo, F. M., Rosso, P. y Sanguinetti, M. (2019). SemEval-2019 Task 5: Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter. *Proceedings of the 13th International Workshop on Semantic Evaluation (SemEval-2019)*, 54–63. <https://aclanthology.org/S19-2007>
