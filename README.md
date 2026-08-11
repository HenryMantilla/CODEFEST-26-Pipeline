# CODEFEST AD ASTRA 2026 — Etapa 1: Base de Conocimiento

Pipeline de extracción, indexación y recuperación sobre el corpus de ADL
(1,826 documentos, ~3.14 GB, fenómenos F1/F2/F3). El entregable vive en
[`entrega/`](entrega/README.md); este README cubre cómo reproducir todo
desde cero.

## Setup

```bash
uv sync                    # instala el entorno base (torch pinned, faiss-cpu)
uv sync --extra grafo      # + networkx/transformers/sentencepiece, solo para el grafo
uv sync --extra docling    # + docling, solo si vas a reproducir esa comparación
```

Verifica que torch y faiss quedaron sanos antes de nada más — este proyecto
perdió días a instalaciones que descuadraban el stack CUDA:

```bash
python -c "import torch, faiss; print(torch.__version__, torch.cuda.is_available())"
python tools/diagnose_nccl.py
```

Si algo falla ahí, ver `tools/fix_cuda_stack.py` antes de tocar cualquier otra cosa.

## Pipeline completo

```bash
# 1. triage: clasifica cada PDF (digital / escaneado / disperso / two-up)
python triage.py --corpus "CORPUS CODEFEST AD ASTRA 2026" --out triage.json

# 2. inventario oficial: doc_id de Indice_Datos_Codefest.xlsx
python inventory.py --xlsx "CORPUS CODEFEST AD ASTRA 2026/Indice_Datos_Codefest.xlsx" \
    --corpus "CORPUS CODEFEST AD ASTRA 2026" --out inventory.json

# 3. (opcional) medir dónde Docling gana sobre PyMuPDF, y con cuáles archivos
python tools/compare_extractors.py --corpus "CORPUS CODEFEST AD ASTRA 2026" \
    --triage triage.json --decide docling_files.txt

# 4. construir el índice
python build_index.py --corpus "CORPUS CODEFEST AD ASTRA 2026" \
    --triage triage.json --inventory inventory.json \
    --docling off \
    --out entrega/base_vectorial \
    --encoder intfloat/multilingual-e5-large BAAI/bge-m3

# 5. (opcional, bonus) grafo de conocimiento
python tools/build_graph.py --index-dir entrega/base_vectorial

# 6. generar resultados.jsonl
python entrega/generador.py

# 7. verificar antes de empaquetar
python tools/make_entrega.py

# 8. verificar antes de subir a GitHub
python tools/check_git_ready.py
```

## Evaluación

```bash
# contra la muestra de 7 consultas que dieron los organizadores
python eval.py --gt "CORPUS CODEFEST AD ASTRA 2026/F3_Dinamicas_Territoriales/FASE ORDENADA CODEFEST.xlsx"

# barrido de configuraciones sin recargar el índice en cada corrida
python tools/sweep.py --gt "...FASE ORDENADA CODEFEST.xlsx" --grid doc

# construir juicios propios sobre las 50 consultas (recomendado: la muestra
# de 7 no alcanza resolución para distinguir la mayoría de cambios)
python tools/build_pool.py --index-dir entrega/base_vectorial \
    --queries entrega/consultas_50.jsonl --out pool.xlsx
# ... juzgar la columna `relevancia` a mano ...
python eval.py --judgments pool.xlsx
```

## Estructura del repo

```
build_index.py       orquesta el pipeline de extracción + indexación
triage.py            clasifica cada archivo antes de extraer
inventory.py         mapea archivo -> doc_id oficial
chunking.py          fragmentación con metadata Tabla 1
extraction.py        extracción por formato (pdf/html/json/csv/xlsx/img)
layout.py            orden de lectura: XY-cut, two-up, self-check geométrico
rich_layout.py        extracción con jerarquía de encabezados
docling_extract.py   extracción vía Docling (layout model, sin componentes generativos)
eval.py              evaluación contra el xlsx de muestra o contra pool.xlsx

entrega/              el entregable -- ver entrega/README.md
  generador.py         script único, autocontenido, entry point (1.4)
  base_vectorial/       índices FAISS + metadata + grafo/
  consultas_50.jsonl
  informe_tecnico.pdf

tools/                utilidades de desarrollo, NO forman parte del entregable
  build_graph.py         construye el grafo de conocimiento (bonus, sección 7)
  build_pool.py           juicios de relevancia agrupados (50 consultas)
  sweep.py                barre configuraciones de generador.py sin recargar el índice
  compare_extractors.py   PyMuPDF vs Docling, medido por alternancia de columnas
  make_entrega.py         valida entrega/ antes de empaquetar
  check_git_ready.py      valida el repo antes de git push
  diagnose_nccl.py, fix_cuda_stack.py, fix_torch_cuda.py   diagnóstico de entorno CUDA
  check_graph_linking.py, find_unmatched.py, inspect_failed.py   diagnósticos puntuales
```

## Decisiones de diseño (resumen; el detalle está en `informe_tecnico.pdf`)

- **Encoders**: `intfloat/multilingual-e5-large` + `BAAI/bge-m3`, fusionados
  por RRF (fragmentos) / CombSUM o max-pooling (documentos, `--doc-agg`).
- **Léxico**: BM25 sobre `metadata.jsonl`, restringido al vocabulario de las
  consultas -- permitido explícitamente por la FAQ.
- **Reranker**: `BAAI/bge-reranker-v2-m3` (cross-encoder), confirmado
  permitido por la FAQ ("la restricción aplica para arquitecturas decoder").
- **Grafo**: NER + co-ocurrencia con patrones verbales, entidades con techo
  de frecuencia (`--max-entity-frac`) para evitar que términos genéricos
  dominen el grafo. Integrado a la recuperación como canal adicional (8.5),
  no solo construido -- la FAQ es explícita en que construirlo sin
  integrarlo no puntúa.
- **Extracción**: enrutamiento por archivo, medido, entre PyMuPDF y Docling
  (`tools/compare_extractors.py --decide`), no por regla fija.

## Licencia y corpus

El corpus (`CORPUS CODEFEST AD ASTRA 2026/`) es de ADL y no se distribuye en
este repositorio -- ver `.gitignore`. Los modelos usados son de licencia
permisiva (MIT/Apache-2.0); ver `informe_tecnico.pdf` para la lista completa
con licencias.



# entrega/ — CODEFEST AD ASTRA 2026, Etapa 1

Section 1.4: *"El objetivo es reproducir los resultados. Si no es posible
reproducir los resultados, se excluirá de la evaluación."* Everything needed
to regenerate `resultados.jsonl` byte-for-byte is inside this directory.

```
entrega/
  resultados.jsonl          50 lines, q001..q050 (9.3)
  generador.py              retrieval script (1.4), the entry point graders
                            run. Self-contained: imports nothing else in
                            this repo. The FAQ permits an extra lib/ folder
                            ("Pueden agregar una carpeta extra lib que
                            consuma generador.py") -- this project keeps
                            everything in one file instead, by choice, so
                            there is exactly one file to audit.
  consultas_50.jsonl        the 50 queries, so the run is self-contained
  informe_tecnico.pdf       design document, <= 8 pages (1.4)
  base_vectorial/
    encoder_multilingual-e5-large/{index.faiss, metadata.jsonl, encoder.json}
    encoder_bge-m3/{index.faiss, metadata.jsonl, encoder.json}
    grafo/grafo.graphml     knowledge graph, Section 7 (bonus)
```

`encoder.json` is an extra file, not required by 1.4. It records the model
name and the query/passage prefixes used at index time so `generador.py`
cannot encode queries differently from how the documents were encoded — the
failure mode in 8.1 that degrades retrieval silently rather than erroring.

## Reproduce

```bash
pip install faiss-cpu sentence-transformers numpy
cd entrega
python generador.py                       # defaults match the submitted run
```

Defaults: `--index-dir base_vectorial --queries consultas_50.jsonl
--out resultados.jsonl`, all relative to this directory.

`--index-dir` is the folder **containing** the `encoder_*/` subfolders, not
one of those subfolders. Inputs are resolved against the current directory
first and then against the script's own directory, so running from one level
up also works and needs no flags:

```bash
python entrega/generador.py               # from the repository root
```

The script prints the absolute index, query and output paths it settled on,
then validates the output against the 9.3.2 schema before exiting.

First run downloads the encoders from HuggingFace (~2.5 GB) and builds the
BM25 channel from `metadata.jsonl` (about a minute). No network access is
needed afterwards.

## Configuration actually submitted

| flag | value |
|---|---|
| `--doc-score` | `combsum` |
| `--doc-pool` | 30 |
| `--doc-hit-bonus` / `--doc-hit-cap` | 0.02 / 3 |
| `--dedupe-threshold` | 0.45 |
| `--phenomenon-boost` | 0.08 (multiplicative, RRF space) |
| `--phenomenon-boost-doc` | 0.03 (additive, CombSUM space) |
| `--bm25-weight` | 1.0 |
| `--reranker` | *(off)* |
| `--depth` | 1000 |

## Compliance notes (8.3)

No generative model is used anywhere. The pipeline is: encoder → FAISS
`IndexFlatIP` cosine search → BM25 term statistics → rank/score fusion
(RRF and CombSUM, both named in 8.4) → metadata post-filters (8.7) → score
aggregation to document level (8.6). Both encoders are BERT-family
bidirectional encoders (4.2). Nothing is generated, summarised, rewritten or
reranked by a decoder.