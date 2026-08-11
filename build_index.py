"""
build_index.py — Vector store construction (CODEFEST AD ASTRA 2026, 5-6).

Pipeline: documents -> extraction -> sentence chunking -> encoder ->
normalized vectors -> IndexFlatIP + metadata.jsonl

------------------------------------------------------------------------
DESIGN NOTES
------------------------------------------------------------------------

1. NO OCR ON THE CPU PATH, EVER.
   The old routing sent unclassified files (and every "sparse" file) to
   extract_pdf(mode="auto"), which rasterizes and Tesseracts any page with
   under 100 characters of text layer. Digital reports are full of those:
   figure spreads, chapter dividers, image covers. On 36k pages that was
   thousands of 300-dpi OCR runs nobody asked for -- the actual reason a
   60-second job ran for hours. Every PDF now defaults to mode "digital",
   which cannot call OCR. Scanned and sparse files go to Phase B.

2. LARGE PDFs ARE SHARDED ACROSS WORKERS.
   A 1330-page PDF as one task occupies one worker while the rest sit idle.
   Any PDF over --shard-min-pages is split into --shard-pages tasks that run
   in parallel and are reassembled in page order before chunking. Verified
   byte-identical to unsharded extraction.

   SCHEDULING is longest-processing-time-first. A long task started last has
   nothing to overlap with; started first it is absorbed while short tasks
   fill in around it.

3. MULTI-COLUMN IS HANDLED ON DIGITAL PAGES. triage.json reports 490 of 760
   PDFs with multi-column sampled pages; the old digital -> "plain" routing
   never applied XY-cut, so four-column spreads came out read straight
   across.

4. THREE EXTRACTION TIERS, cheapest first:
     PyMuPDF + XY-cut   geometry only. Free. Handles ordinary documents.
     rich_layout        span-level, font-metric based. Handles stat callouts,
                        rotated running heads, heading context, captions.
     Docling            trained layout model. Handles what geometry cannot.
                        Off by default: it is a neural pass per page.

5. .pbf TILES ARE EXCLUDED. Their contents are already present, decoded,
   inside AMAZONUW_amazonunderworld-data.csv with the tile coordinates as
   columns. Indexing both would put two copies of every municipality in the
   index -- the near-duplicate problem that wastes top-10 slots.
   pbf_extract.py stays in the repo for the case where you want the tiles
   without the CSV.

6. OCR BACKEND DEFAULTS TO RAPIDOCR, NOT PADDLEOCR.
   PaddleOCR needs the PaddlePaddle framework, which has no prebuilt wheel
   for sm_120 (Blackwell), and whose CUDA dependency pins repeatedly
   displaced the nvidia-nccl-cu12 that torch links against -- breaking the
   ENCODER, which is the one component this pipeline cannot do without.
   RapidOCR runs the SAME PP-OCRv5 detection and recognition models through
   ONNX Runtime, which shares no shared objects with torch. Same models,
   same accuracy, no framework collision.

   If you reinstall PaddleOCR, put it in a SEPARATE venv. Phase B is already
   a single isolated process, so running it standalone costs nothing.

------------------------------------------------------------------------
PHASE STRUCTURE
------------------------------------------------------------------------
    Phase A  (CPU, multiprocess)  every file with a usable text layer
    Phase A2 (CPU, multiprocess)  reassemble + chunk sharded documents
    Phase B  (GPU, SINGLE proc)   triage-flagged scanned/sparse + images
    Phase C  (GPU, SINGLE proc)   Docling, when --docling is not off

    Phases B and C cannot be folded into the pool. Loading a GPU model inside
    15 worker processes makes 15 CUDA contexts fight over one device: at best
    it serialises anyway while paying 15x the memory cost, at worst it OOMs.

DEADLOCK SAFETY (this was regressed once already -- do not undo it)
    torch is NOT imported before the process pool. torch.cuda.is_available()
    creates a CUDA context, and a CUDA context cannot survive fork(): the
    children deadlock on their first allocator call, which looks exactly like
    a run that freezes mid-progress with no error message. The pool uses the
    spawn start method for the same reason.

DECODER-FREE (spec sections 4.2, 8.3)
    PaddleOCR and Tesseract are CTC-based, not autoregressive. Do not
    substitute TrOCR or any seq2seq OCR model. Docling's TableFormer is
    encoder-decoder and is force-disabled in docling_extract.py; its VLM
    pipeline is autoregressive and is not exposed at all.

LICENSING (spec 4.3: "Se prefieren licencias Apache 2.0, MIT o CC BY")
    PaddleOCR Apache-2.0, Tesseract Apache-2.0, Docling code MIT. Surya is
    deliberately not offered: its weights are AI Pubs Open RAIL-M.

Usage:
    python build_index.py --corpus "CORPUS CODEFEST AD ASTRA 2026" \
        --triage triage.json --inventory inventory.json \
        --ocr-backend rapid --workers 15

    # add Docling for the design-heavy subset only
    python build_index.py ... --docling design

    # smoke test
    python build_index.py --corpus ./corpus --limit 40 --ocr-backend none

Requires:
    pip install sentence-transformers faiss-cpu numpy torch pymupdf
    pip install rapidocr                    # --ocr-backend rapid (default)
    # OCR runs on the GPU through RapidOCR's TORCH engine, reusing the torch
    # that already works here. Do NOT install onnxruntime-gpu: its wheels
    # have no sm_120 kernels and are built for CUDA 12, so on this box it is
    # a silent CPU fallback at best and a cu12/cu13 library fight at worst.
    # Plain CPU onnxruntime is fine to have; it is the fallback.
    pip install docling                     # --docling

    NOT in this venv: paddleocr / paddlepaddle-gpu. They pin
    nvidia-nccl-cu12 to a version torch does not link against, and installing
    them breaks the encoder. Keep a constraints file to catch it early:
        echo "nvidia-nccl-cu12==2.28.9" > constraints.txt   # match YOUR torch
        uv pip install <anything> --constraint constraints.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# faiss / torch / sentence_transformers / docling are imported lazily.
# With the spawn start method every worker re-imports this module, and paying
# a multi-second torch import in each of 15 children would cost more than the
# extraction itself. Module level stays light on purpose.
from extraction import MODE_BY_CLASS, DEFAULT_PDF_MODE

ENCODERS = {
    "intfloat/multilingual-e5-large": {"doc_prefix": "passage: ", "query_prefix": "query: "},
    "intfloat/multilingual-e5-base":  {"doc_prefix": "passage: ", "query_prefix": "query: "},
    "BAAI/bge-m3":                    {"doc_prefix": "", "query_prefix": ""},
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2":
                                      {"doc_prefix": "", "query_prefix": ""},
}

# Part of the extraction cache key (_cache_key). BUMP THIS whenever
# chunking.py changes how a record is built -- boundaries, overlap, or the
# metadata field set. The cache stores finished record dicts, so without a
# bump a rebuild happily replays the OLD records and the fix you just made
# never reaches metadata.jsonl. That failure is silent: the run is fast, the
# counts match, and the field is still wrong.
# NOTE: only THIS constant is in the key. Editing layout.py or extraction.py
# changes the text without changing the key, so a cached run replays the old
# text and the fix appears to have done nothing. Bump on any change to
# extraction, layout, chunk boundaries, or the metadata field set.
#   v2-hardcap  force-split for sentences with no boundary
#   v3-fuente   `fuente` = file name + `ruta_relativa`; overlap no longer
#               collapses to zero after a sentence longer than overlap_chars
#   v4-layout   XY-cut prefers the column gutter over paragraph-band splits,
#               GUTTER_FRACTION 0.035 -> 0.025. Two-column pages were being
#               read across, affecting 490 of 760 PDFs.
CHUNKER_VERSION = "v4-layout"
CHARS_PER_TOKEN = 2.8      # conservative for XLM-R vocabularies on es/en/pt

# 60 pages is roughly 2-4 seconds of PyMuPDF work: small enough that the
# slowest shard cannot dominate the run, large enough that per-task overhead
# (spawn pickling, PDF open) stays negligible.
SHARD_MIN_PAGES = 120
SHARD_PAGES = 60

# --------------------------------------------------------------- exclusions
#
# Files that are in the corpus but should NOT become documents in the index.
# Every one of these produces text that is structurally valid and
# semantically worthless, which is the dangerous kind: it never errors, it
# just competes for top-10 slots against real prose and wins sometimes.
#
#   catalogs/manifests  13 JSON + 4 CSV of scrape bookkeeping (status,
#                       size_bytes, scraped_at). Useful as metadata, never as
#                       an answer.
#   PubMed exports      111,775 + 61,521 + 46,514 + 17,054 rows of
#                       bibliographic listings. No query asks about biomedical
#                       literature. Indexing them would roughly double the
#                       chunk count for zero recall.
#   .pbf tiles          73 Mapbox vector tiles whose contents are ALREADY in
#                       AMAZONUW_amazonunderworld-data.csv, decoded, with the
#                       tile coordinates as columns. Indexing both puts two
#                       copies of every municipality in the index -- exactly
#                       the near-duplicate problem that wastes top-10 slots.
#                       pbf_extract.py stays in the repo for the case where
#                       you want the tiles without the CSV.
#   evaluation files    the 50-question PDF, the sample ground truth, and the
#                       official inventory are about the challenge, not part
#                       of the corpus.
EXCLUDE_PATTERNS = [
    "*catalog*.json", "*catalogo*.json", "*catalog*.csv", "*catalogo*.csv",
    "*tiles-index.json", "*publicaciones-2.csv",
    "*pubmed-*.csv", "*pubmed-*.xlsx", "*lit-covid-ai*.csv", "*lit-covid-ai*.xlsx",
    # clinicaltrials exports: same shape and same argument as the PubMed ones
    # -- registry listings (NCT id, title, sponsor, phase, status). Six files
    # of trial bookkeeping that no query in the 50 asks about, and every row
    # is close to every other row, so they crowd the top 10 with rows that
    # differ only in an identifier.
    "*clinicaltrials-*.csv", "*clinicaltrials-*.xlsx",
    "*.pbf",
    "Extracto_Preguntas*.pdf", "FASE ORDENADA*.xlsx", "Indice_Datos_Codefest.xlsx",
]



# Image suffixes. `.avif` is here because one SWF asset uses it and
# FORMAT_BY_SUFFIX would otherwise return None, dropping the file silently --
# the same class of bug as the .pbf files that vanished without a log line.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff",
                  ".bmp", ".avif"}

# Loose images whose NAME says they carry data rather than decoration. Used
# only to warn: dropping "table-5-1-web.jpg" is a different decision from
# dropping a photograph of a launch, and the two should not be made silently
# together.
_DATA_IMAGE = ("table", "chart", "stoplight", "by-country", "figure", "fig-",
               "graph", "matrix", "timeline")


def image_carries_data(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in _DATA_IMAGE)


def is_excluded(relative: str, patterns: list[str]) -> str:
    """Return the pattern that excluded this file, or ""."""
    from fnmatch import fnmatch
    name = relative.replace("\\", "/").split("/")[-1]
    for pattern in patterns:
        if fnmatch(name, pattern) or fnmatch(name.lower(), pattern.lower()):
            return pattern
    return ""


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 2


# --------------------------------------------------------------- OCR language

# triage.py hands out Tesseract-style 3-letter codes. PaddleOCR 3.x uses
# 2-letter ISO 639-1 codes and rejects the 3-letter forms.
TESSERACT_TO_PADDLE = {"spa": "es", "eng": "en", "por": "pt",
                       "fra": "fr", "deu": "de", "ita": "it"}
TESSERACT_TO_EASYOCR = {"spa": "es", "eng": "en", "por": "pt",
                        "fra": "fr", "deu": "de", "ita": "it"}

OCR_LANGS = {
    "Alertas_Tempranas": "spa", "MAPP_OEA": "spa", "CEEEP": "spa",
    "ILIA_Latam": "spa", "RutaN_GEIAL": "spa", "CENIA": "spa+eng",
    "INPE": "por", "Amazon_Underworld": "spa+por", "RESDAL": "eng+spa",
}
DEFAULT_OCR_LANG = "eng+spa"


def ocr_language_for(relative_path: str) -> str:
    for key, lang in OCR_LANGS.items():
        if key in relative_path:
            return lang
    return DEFAULT_OCR_LANG


def to_paddle_lang(tesseract_codes: str) -> str:
    """PaddleOCR takes ONE language per call, not Tesseract's 'spa+eng'."""
    first = tesseract_codes.split("+")[0].strip()
    return TESSERACT_TO_PADDLE.get(first, "es")


def to_easyocr_langs(tesseract_codes: str) -> list[str]:
    """EasyOCR (Docling's default OCR engine) takes a LIST of 2-letter codes."""
    out = [TESSERACT_TO_EASYOCR.get(c.strip())
           for c in tesseract_codes.split("+")]
    return [c for c in out if c] or ["es"]


# --------------------------------------------------------------- design-heavy


# Language suffixes used in this corpus, in both conventions seen: separated
# (SWF_gcsr-2026-execsum-spa.pdf) and glued (MAPPOEA_37...mappoeaesp.pdf).
_LANG_SUFFIX = re.compile(
    r"[-_]?(esp|eng|spa|por|ing|chi|rus|fra|espanol|english|ingles|portugues)$",
    re.IGNORECASE)

# Order of preference. Queries are 100% Spanish; the encoders are
# cross-lingual, so an English sibling is still useful, but a Chinese or
# Russian one is a near-duplicate nobody can retrieve deliberately.
_LANG_RANK = {"spa": 0, "esp": 1, "espanol": 1,
              "eng": 2, "ing": 3, "english": 2, "ingles": 3,
              "por": 4, "portugues": 4, "fra": 5, "chi": 9, "rus": 9}


def redundant_translations(files: list[Path], corpus: Path,
                           keep_ranks_below: int = 5) -> set[str]:
    """
    Paths that are a translation of a document already in the corpus, in a
    language no query uses.

    THE PROBLEM THEY CAUSE
        SWF_gcsr-2026-execsum ships as chi/eng/por/rus/spa. Those five are the
        same report, so they compete for the same slots with near-identical
        embeddings -- and the multilingual encoders match a Spanish query to
        the Chinese version well enough to rank it. Only one can ever be the
        ground-truth answer, so the other four can consume document slots and
        never score. Section 10.2.2 gives three document slots per query;
        spending two on translations of the right answer is the same loss as
        retrieving the wrong document.

    WHAT IS DELIBERATELY NOT DROPPED
        A Portuguese document with no Spanish or English sibling. INPE's
        Brazilian material is Portuguese-only and is the sole source for its
        topics -- dropping it would remove content, not duplication. Only
        variants of a document that ALREADY has a preferred-language sibling
        are removed, so nothing unique is ever lost.

    Small in this corpus: two groups, seven files. Included because one of
    them is a ground-truth document for q036 and its English twin is not.
    """
    groups: dict[str, list[tuple[int, str, str]]] = {}
    for path in files:
        if path.suffix.lower() != ".pdf":
            continue
        stem = path.stem
        match = _LANG_SUFFIX.search(stem)
        if not match:
            continue
        language = match.group(1).lower()
        base = stem[:match.start()].lower().rstrip("-_")
        if len(base) < 8:                     # too short to be a real match
            continue
        groups.setdefault(base, []).append(
            (_LANG_RANK.get(language, 6), language,
             str(path.relative_to(corpus))))

    drop: set[str] = set()
    for base, members in groups.items():
        if len({lang for _r, lang, _p in members}) < 2:
            continue
        best = min(rank for rank, _l, _p in members)
        if best >= keep_ranks_below:
            continue                          # no preferred-language sibling
        for rank, language, relative in sorted(members):
            if rank >= keep_ranks_below:
                drop.add(relative)
    return drop


def is_design_heavy(profile: dict) -> bool:
    """
    Decide whether a PDF needs span-level (rich_layout) or model-based
    (Docling) parsing rather than plain geometry.

    The signature of an annual-report spread, as opposed to ordinary
    multi-column prose: moderate text density (big type, lots of white
    space), several images per page, and columns. A dense two-column
    academic paper hits the column test but not the other two, and is better
    served by plain extraction, which is what the reference fragments were
    built from. Being conservative matters: both alternatives change chunk
    boundaries, so applying them where they are not needed moves output away
    from the ground truth for no gain.
    """
    sampled = max(profile.get("sampled_pages", 0), 1)
    chars = profile.get("chars_per_page", 0.0)
    images = profile.get("images_per_page", 0.0)
    multicolumn = profile.get("multicolumn_pages", 0) / sampled
    return chars < 2200 and images >= 1.0 and multicolumn >= 0.4


def is_complex_layout(profile: dict) -> bool:
    """
    Files where plain get_text() produces SPLICED text -- the first line of
    column one joined to the first line of column two -- rather than merely
    imperfect text.

    Broader than is_design_heavy(), and deliberately so. That function asks
    "would a layout model read this better?", which is a quality judgement
    worth being conservative about. This one asks "is plain extraction
    actively corrupting the text?", and the answer is yes whenever a page has
    more than one column or holds two logical pages. A spliced sentence is
    fluent Spanish that nobody wrote: it embeds to a plausible vector, sits in
    the index looking healthy, and can never match a ground-truth fragment.

    `two_up` alone qualifies even at one column per half, because the fold is
    a column boundary that get_text() cannot see.
    """
    sampled = max(profile.get("sampled_pages", 0), 1)
    multicolumn = profile.get("multicolumn_pages", 0) / sampled
    return bool(profile.get("two_up")) or multicolumn >= 0.3


# --------------------------------------------------------------- worker (CPU)

def _pin_worker_threads() -> None:
    """
    NumPy/BLAS and any OpenMP library spawn their own thread pools. With N
    worker processes already saturating the machine, each fanning out M
    internal threads oversubscribes the CPU by N*M and makes a high core
    count perform WORSE than a conservative one. Pin every worker to one
    internal thread so parallelism is additive instead of self-competing.
    """
    for var in ("OMP_THREAD_LIMIT", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def _default_workers() -> int:
    """Physical cores - 1. SMT gives little to CPU-bound single-thread work."""
    logical = os.cpu_count() or 4
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or logical
    except ImportError:
        physical = max(1, logical // 2)
    return max(1, physical - 1)


def _cache_key(path: Path, root: Path, suffix: str = "") -> str:
    stat = path.stat()
    raw = (f"{path.relative_to(root)}|{stat.st_size}|{int(stat.st_mtime)}"
           f"|{CHUNKER_VERSION}{suffix}")
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _read_cache(cache_dir: str, path: Path, root: Path, suffix: str = ""):
    if not cache_dir:
        return None
    try:
        cache_file = Path(cache_dir) / f"{_cache_key(path, root, suffix)}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        pass                                   # corrupt or unreadable: redo it
    return None


def _write_cache(cache_dir: str, path: Path, root: Path, payload,
                 suffix: str = "") -> None:
    if not cache_dir:
        return
    try:
        cache_file = Path(cache_dir) / f"{_cache_key(path, root, suffix)}.json"
        cache_file.write_text(json.dumps(payload, ensure_ascii=False),
                              encoding="utf-8")
    except Exception:
        pass


# A task is a plain tuple so it pickles cheaply into spawned children.
#   kind: "file"  -> extract + chunk one whole file, return chunk records
#         "shard" -> extract a PAGE RANGE, return raw text for reassembly
#         "rich"  -> rich_layout parse of a page range, return chunk records
# index: (kind, path, root, doc_id, phenomenon, cache_dir, max_chars,
#         overlap_chars, mode, ocr_lang, first_page, last_page, weight)

def process_one(task) -> tuple:
    """
    Phase A worker. Runs in a spawned process, imports its own dependencies,
    returns plain dicts or plain text. Never touches the GPU, never OCRs.

    Returns (kind, doc_id, first_page, payload, note, seconds). The timing is
    measured HERE rather than in the parent because as_completed() reports
    when a result was collected, not how long the work took -- a task that
    finished early can sit in the result queue behind a slow one and look
    slow itself, which would poison the slow-file report.
    """
    started = time.time()
    (kind, path_str, root_str, doc_id, phenomenon, cache_dir, max_chars,
     overlap_chars, mode, ocr_lang, first_page, last_page, _weight) = task
    path, root = Path(path_str), Path(root_str)

    suffix = "" if kind == "file" else f"|{kind}|{first_page}|{last_page}"
    cached = _read_cache(cache_dir, path, root, suffix)
    if cached is not None:
        return kind, doc_id, first_page, cached, "cached", time.time() - started

    try:
        if kind == "shard":
            from extraction import extract_pdf
            text = extract_pdf(path, mode=mode, ocr_lang=ocr_lang,
                               first_page=first_page, last_page=last_page)
            _write_cache(cache_dir, path, root, text, suffix)
            return kind, doc_id, first_page, text, "", time.time() - started

        if kind == "rich":
            from chunking import chunk_text_units
            from rich_layout import parse_document, is_useful
            from extraction import guess_title

            units = [u for u in parse_document(path, first_page=first_page,
                                               last_page=last_page)
                     if is_useful(u)]
            if not units:
                return (kind, doc_id, first_page, [], "no usable units",
                        time.time() - started)
            # FIX: filename-derived title. rich_layout units already carry
            # their own heading trail (unit.context_prefix()), so this only
            # adds the document-level anchor a heading trail cannot give --
            # cheap and safe even though it is coarser than the JSON path's
            # real title.
            doc_title = guess_title(units[0].text if units else "", path.name)
            chunks = chunk_text_units(
                units, doc_id=doc_id, source=str(path.relative_to(root)),
                phenomenon=phenomenon, count_tokens=estimate_tokens,
                file_name=path.name, official_doc_id=doc_id,
                max_chars=max_chars, overlap_chars=overlap_chars,
                doc_title=doc_title)
            records = [c.to_dict() for c in chunks]
            _write_cache(cache_dir, path, root, records, suffix)
            return kind, doc_id, first_page, records, "", time.time() - started

        # kind == "file"
        from chunking import chunk_document
        from extraction import load_document

        doc = load_document(path, doc_id=doc_id, phenomenon=phenomenon,
                            mode=mode, ocr_lang=ocr_lang,
                            official_doc_id=doc_id)
        if doc is None:
            return (kind, doc_id, first_page, [], "no usable text",
                    time.time() - started)
        # `source` carries the RELATIVE PATH and `file_name` the basename.
        # Which of the two lands in `fuente` is decided in one place only --
        # Chunk.to_dict() in chunking.py -- so do not special-case it here.
        # (It is the file name; the path goes to `ruta_relativa`. See the
        # docstring there for why the path was the riskier choice.)
        doc.source = str(path.relative_to(root))
        doc.file_name = path.name
        chunks = chunk_document(doc, estimate_tokens, max_chars=max_chars,
                                overlap_chars=overlap_chars)
        records = [c.to_dict() for c in chunks]
        _write_cache(cache_dir, path, root, records, suffix)
        return kind, doc_id, first_page, records, "", time.time() - started

    except Exception as exc:
        return (kind, doc_id, first_page, [],
                f"{kind} failed: {type(exc).__name__}: {exc}"[:160],
                time.time() - started)


def chunk_merged(args) -> tuple[str, list[dict], str, float]:
    """
    Second pass: chunk a document that was extracted as several shards.

    Chunking happens here rather than in the shard workers because a chunk
    that straddles a shard boundary must see both sides. Splitting the text
    at page 60 and chunking each half independently would put a hard cut
    there, losing the overlap window and truncating whatever sentence spans
    the boundary.
    """
    started = time.time()
    (doc_id, source, phenomenon, text, max_chars, overlap_chars,
     cache_dir, path_str, root_str) = args
    from dataclasses import dataclass as _dc
    from chunking import chunk_document
    from extraction import guess_title

    @_dc
    class _Doc:
        doc_id: str
        source: str
        file_format: str
        phenomenon: int
        text: str
        file_name: str = ""
        official_doc_id: str = ""
        title: str = ""

    try:
        doc = _Doc(doc_id, source, "pdf", phenomenon, text,
                   Path(path_str).name, doc_id,
                   guess_title(text, Path(path_str).name))
        chunks = chunk_document(doc, estimate_tokens, max_chars=max_chars,
                                overlap_chars=overlap_chars)
        records = [c.to_dict() for c in chunks]
        _write_cache(cache_dir, Path(path_str), Path(root_str), records)
        return doc_id, records, "", time.time() - started
    except Exception as exc:
        return (doc_id, [],
                f"merge-chunk failed: {type(exc).__name__}: {exc}"[:160],
                time.time() - started)


# --------------------------------------------------------------- timing log

class TimingLog:
    """
    Records how long every unit of work took, so the slow tail can be looked
    at by hand instead of guessed at.

    Kept in memory rather than streamed to disk: even at 2000+ entries this
    is a few hundred KB, and writing once at the end means a crashed run does
    not leave a half-flushed file that reads like a complete report.
    """

    def __init__(self, path: Path | None, threshold: float = 5.0):
        self.path = path
        self.threshold = threshold
        self.rows: list[dict] = []

    def add(self, seconds: float, phase: str, kind: str, source: str,
            doc_id: str, span: str, chunks: int, note: str) -> None:
        self.rows.append({
            "seconds": round(seconds, 2), "phase": phase, "kind": kind,
            "doc_id": doc_id, "span": span, "chunks": chunks,
            "note": note, "source": source,
        })

    def write(self, top: int = 300) -> None:
        if not self.path or not self.rows:
            return
        rows = sorted(self.rows, key=lambda r: -r["seconds"])
        slow = [r for r in rows if r["seconds"] >= self.threshold]
        total = sum(r["seconds"] for r in rows)

        lines = [
            "# Slow-task report — sorted by wall time, slowest first.",
            "# Times are per TASK, measured inside the worker, so a task is",
            "# not blamed for time it spent queued behind another one.",
            "#",
            "# phase: A=CPU pool  A2=shard reassembly  B=OCR  C=Docling",
            "# pages: page range for a shard, page count for OCR/Docling",
            "#",
            f"# tasks: {len(rows)}   cumulative worker time: {total:.0f}s",
            f"# at or above {self.threshold:g}s: {len(slow)}",
            "#",
            "# seconds\tphase\tkind\tdoc_id\tpages\tchunks\tnote\tsource",
        ]
        for row in rows[:top]:
            lines.append(
                f"{row['seconds']:.2f}\t{row['phase']}\t{row['kind']}\t"
                f"{row['doc_id']}\t{row['span']}\t{row['chunks']}\t"
                f"{row['note'] or '-'}\t{row['source']}")

        if len(rows) > top:
            lines.append(f"# ... {len(rows) - top} faster tasks omitted")

        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"      slow-task report -> {self.path} "
              f"({len(slow)} tasks >= {self.threshold:g}s)")


def _progress_line(done: int, total: int, seconds: float, doc_id: str,
                   label: str, chunks: int, name: str, note: str) -> str:
    flag = "" if not note else ("  [cached]" if note == "cached"
                                else f"  !! {note}")
    return (f"  [{done:5d}/{total}] {seconds:6.2f}s  {doc_id:<18} "
            f"{label:<16} {chunks:5d} ch  {name[:50]}{flag}")


# --------------------------------------------------------------- OCR (GPU)

class PaddleOcrEngine:
    """
    Single-process PaddleOCR wrapper. One model load, reused across pages.
    Instantiated only in the main process, only after Phase A's pool closed.
    """

    def __init__(self, lang: str = "es"):
        from paddleocr import PaddleOCR
        self.lang = lang
        # Orientation/unwarping sub-models are extra passes we do not need on
        # ordinary report scans, and each one costs time per page.
        self.reader = PaddleOCR(lang=lang,
                                use_doc_orientation_classify=False,
                                use_doc_unwarping=False,
                                use_textline_orientation=False)

    def page_text(self, page, dpi: int = 300) -> str:
        import tempfile
        from layout import order_blocks

        pixmap = page.get_pixmap(dpi=dpi)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            pixmap.save(tmp_path)
            result = self.reader.predict(tmp_path)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

        blocks: list[tuple[float, float, float, float, str]] = []
        for page_result in result:
            texts = page_result.get("rec_texts") or []
            boxes = page_result.get("rec_boxes")
            polys = page_result.get("rec_polys")

            for i, text in enumerate(texts):
                if not text or not text.strip():
                    continue
                if boxes is not None and i < len(boxes):
                    x0, y0, x1, y1 = (float(v) for v in boxes[i][:4])
                elif polys is not None and i < len(polys):
                    xs = [float(p[0]) for p in polys[i]]
                    ys = [float(p[1]) for p in polys[i]]
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                else:
                    x0 = y0 = float(i); x1 = y1 = float(i) + 1
                blocks.append((x0, y0, x1, y1, text.strip()))

        if not blocks:
            return ""

        # Same ordering used for digital pages -- order_blocks(), not a bare
        # _xy_cut(). Detector output is in detection order, which interleaves
        # columns; _xy_cut alone fixes that but skips furniture stripping and
        # the two-up split, so a SCANNED spread stayed spliced long after the
        # digital path was fixed. A scanned two-column page and a digital one
        # are the same layout problem and now get the same answer.
        ordered = order_blocks(blocks, float(pixmap.width),
                               float(pixmap.height))
        return "\n".join(b[4] for b in ordered)

    def image_text(self, path: Path) -> str:
        result = self.reader.predict(str(path))
        parts = []
        for page_result in result:
            parts.extend(t.strip() for t in (page_result.get("rec_texts") or [])
                         if t and t.strip())
        return "\n".join(parts)



def _rapidocr_enum(class_name: str, *candidates):
    """
    Resolve a RapidOCR 3.x config enum member by name or by value.

    RapidOCR validates several params as Enum instances, not strings:
    `Rec.ocr_version` rejects "PP-OCRv5" with "must be Enum Type". The member
    names have moved between releases (PPOCRV5 / PP_OCRV5 / PPOCRv5), so match
    on the name first and then on the VALUE, which is the stable part.

    Returns None when rapidocr, the enum class, or every candidate is absent,
    and callers then fall back to the plain string -- which older builds still
    accept.
    """
    try:
        import rapidocr
    except ImportError:
        return None
    enum_class = getattr(rapidocr, class_name, None)
    if enum_class is None:
        return None

    for candidate in candidates:
        member = getattr(enum_class, candidate, None)
        if member is not None:
            return member

    def flat(text: str) -> str:
        return str(text).lower().replace("-", "").replace("_", "")

    wanted = {flat(c) for c in candidates}
    try:
        for member in enum_class:
            if flat(member.value) in wanted or flat(member.name) in wanted:
                return member
    except TypeError:
        pass
    return None


def rapidocr_latin_params(version: str = "v5") -> dict:
    """
    Recognition-head params asking for the Latin dictionary, expressed the way
    THIS installed RapidOCR wants them.

    PP-OCRv6's small recognition model dropped lang_type='latin'; PP-OCRv5
    still ships it, and Latin covers Spanish, English and Portuguese in one
    head. So pin the version -- as an enum where the library demands one.
    """
    params: dict = {}
    ocr_version = _rapidocr_enum("OCRVersion", "PPOCRV5", "PP_OCRV5",
                                 "PPOCRv5", "PP-OCRv5")
    if ocr_version is not None:
        params["Rec.ocr_version"] = ocr_version

    lang_rec = _rapidocr_enum("LangRec", "LATIN", "latin")
    params["Rec.lang_type"] = lang_rec if lang_rec is not None else "latin"

    # model_type must be pinned too, and to MOBILE specifically. Not every
    # (version, language, size) triple exists as a published model: probing
    # rapidocr 3.x, PP-OCRv5 + latin + mobile validates while
    # PP-OCRv5 + latin + server raises "Invalid OCR configuration", and the
    # v6 default (small) is what produced "Unsupported rec.lang_type='latin'
    # for PP-OCRv6 small model". Leaving it unset means inheriting whatever
    # default the current release picked, which is how this broke.
    # Recognition stays on MOBILE. Probed against this rapidocr build:
    #   Rec PP-OCRv5 + latin + mobile   valid
    #   Rec PP-OCRv5 + latin + server   INVALID -- no such published model
    # The server recognition models exist only for ch and en. Choosing one
    # would trade every Spanish accent for a marginally better dictionary,
    # which is the wrong trade on a corpus that is 26% Spanish.
    model_type = _rapidocr_enum("ModelType", "MOBILE", "mobile")
    params["Rec.model_type"] = model_type if model_type is not None else "mobile"

    # DETECTION, though, has a server model and detection is language-free.
    # Text detection quality drives OCR accuracy at least as much as the
    # recognition dictionary: a line the detector misses or splits is lost
    # regardless of how good the recogniser is, and the server detector is
    # markedly better on small type, tables and multi-column scans -- exactly
    # the 65 files that reach OCR here. It costs more VRAM and time per page,
    # which is affordable for 735 pages.
    det_version = _rapidocr_enum("OCRVersion", "PPOCRV5", "PP_OCRV5",
                                 "PPOCRv5", "PP-OCRv5")
    det_model = _rapidocr_enum("ModelType", "SERVER", "server")
    if det_version is not None:
        params["Det.ocr_version"] = det_version
    params["Det.model_type"] = det_model if det_model is not None else "server"
    return params


def describe_rapidocr_options() -> str:
    """Valid enum members, printed when every candidate config fails."""
    try:
        import rapidocr
    except ImportError:
        return "rapidocr is not installed"
    lines = []
    for name in ("OCRVersion", "LangRec", "ModelType", "EngineType"):
        enum_class = getattr(rapidocr, name, None)
        if enum_class is None:
            continue
        try:
            members = ", ".join(f"{m.name}={m.value!r}" for m in enum_class)
        except TypeError:
            continue
        lines.append(f"        {name}: {members}")
    return "\n".join(lines) or "        no config enums exposed"


class RapidOcrEngine:
    """
    PP-OCRv5 models via ONNX Runtime. THE DEFAULT BACKEND.

    This is not a downgrade from PaddleOCR: RapidOCR ships the same PP-OCRv5
    detection and recognition models, converted to ONNX. Detection quality,
    layout handling and recognition accuracy are those of the Paddle models.
    What is dropped is the PaddlePaddle framework, which on Blackwell (sm_120)
    has no prebuilt wheel and whose CUDA pins overwrite the nvidia-nccl-cu12
    that torch links against.

    ONNX Runtime and torch share no shared objects, so the two coexist.
    Apache-2.0 and CTC-based: satisfies spec 4.2 (no decoder) and 4.3.

    ON BLACKWELL, USE THE TORCH BACKEND, NOT ONNX-GPU. (--ocr-engine auto)

    RapidOCR can run its PP-OCR models through several inference engines.
    The default is ONNX Runtime, and on sm_120 the GPU path does not work:

      * the official onnxruntime-gpu PyPI wheel ships kernels up to
        sm_89/sm_90. Blackwell is absent, so CUDAExecutionProvider either
        never appears or dies with cudaErrorNoKernelImageForDevice /
        cudaErrorInvalidPtx;
      * onnxruntime-gpu is built against CUDA 12 while this box runs CUDA
        13, which is an independent reason for a silent CPU fallback;
      * installing the cu12 nvidia-* libraries next to a cu13 torch is the
        SAME dependency collision that drove this project off PaddleOCR (see
        design note 6). Fixing OCR by breaking the encoder is not a fix.

    RapidOCR's own documentation recommends the CPU build of ONNX Runtime
    and advises against the GPU build inside rapidocr.

    The torch engine sidesteps all of it. torch already works on this GPU --
    it is what encodes the corpus in phase [5/6] -- so OCR reuses one CUDA
    stack that is known good instead of introducing a second one. The models
    are the same PP-OCR weights in .pth form, so OUTPUT is unchanged; only
    the wall clock differs.

    Order tried by --ocr-engine auto:
        torch + CUDA   -> when torch.cuda.is_available()
        onnx + CUDA    -> when CUDAExecutionProvider is present
        CPU            -> always works, just slower

    Importing torch here is safe: Phase B runs after the Phase A pool has
    closed, so no CUDA context can be inherited across fork(). Do not hoist
    this import to module level -- see DEADLOCK SAFETY at the top of the file.
    """

    def __init__(self, lang: str = "es", use_gpu: bool = True,
                 engine: str = "auto"):
        from rapidocr import RapidOCR
        self.lang = lang

        torch_cuda = False
        if use_gpu and engine in ("auto", "torch"):
            try:
                import torch
                torch_cuda = torch.cuda.is_available()
            except Exception:
                torch_cuda = False

        providers = []
        if engine in ("auto", "onnx"):
            try:
                import onnxruntime
                providers = onnxruntime.get_available_providers()
            except Exception:
                providers = []
        onnx_cuda = use_gpu and "CUDAExecutionProvider" in providers

        # EngineType is an enum in rapidocr 3.x; older builds take the plain
        # string. Try the enum, fall back to "torch".
        try:
            from rapidocr import EngineType
            torch_type = EngineType.TORCH
        except Exception:
            torch_type = "torch"

        torch_params = {"Det.engine_type": torch_type,
                        "Cls.engine_type": torch_type,
                        "Rec.engine_type": torch_type,
                        "EngineConfig.torch.use_cuda": True,
                        "EngineConfig.torch.gpu_id": 0}

        # The default 'ch' recognition dictionary is Chinese+English and does
        # not reliably cover á é í ó ú ñ. Silently stripping accents would
        # corrupt every Spanish document in the OCR set, so ask for the Latin
        # head explicitly -- but never at the cost of failing to start, hence
        # each engine is tried with and then without it.
        # RapidOCR made PP-OCRv6 the default, and its *small* recognition
        # model DROPPED lang_type='latin'. The ladder below then fell through
        # to a bare RapidOCR(), which silently selects the Chinese+English
        # dictionary -- and OCR'd the entire Spanish set with it. The fallback
        # was worse than the failure and nothing in the log said so.
        #
        # Pin PP-OCRv5, which still ships the Latin head, before accepting any
        # default. On a 26%-Spanish corpus this is not cosmetic: "informacion"
        # and "información" are different tokens to BM25, and a mangled accent
        # is a mangled word to the encoder too.
        latin = {"Rec.lang_type": "latin"}
        latin_v5 = rapidocr_latin_params("v5")
        attempts: list[tuple[str, dict | None]] = []
        # Each engine is tried with the pinned-v5 Latin head, then the plain
        # Latin head, and only then without one. Order matters: an accented
        # CPU result beats an unaccented GPU one.
        if torch_cuda and engine in ("auto", "torch"):
            attempts += [("torch/CUDA", {**torch_params, **latin_v5}),
                         ("torch/CUDA", {**torch_params, **latin}),
                         ("torch/CUDA", dict(torch_params))]
        if onnx_cuda and engine in ("auto", "onnx"):
            attempts += [("onnx/CUDA",
                          {**latin_v5, "EngineConfig.onnxruntime.use_cuda": True}),
                         ("onnx/CUDA",
                          {**latin, "EngineConfig.onnxruntime.use_cuda": True})]
        attempts += [("CPU", dict(latin_v5)), ("CPU", dict(latin)), ("CPU", None)]

        errors: list[str] = []
        self.engine_name = ""
        for name, params in attempts:
            try:
                self.reader = RapidOCR() if params is None else RapidOCR(params=params)
                self.engine_name = name
                self.latin = bool(params) and params.get("Rec.lang_type") == "latin"
                break
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}"[:100])
        else:
            raise RuntimeError("RapidOCR would not initialise. Tried: "
                               + " | ".join(errors))

        self.on_gpu = self.engine_name != "CPU"
        self.latin = getattr(self, "latin", False)
        print(f"      RapidOCR: {self.engine_name}"
              f"  (torch.cuda={torch_cuda}, ort={providers or 'not installed'})")
        for line in errors:
            print(f"        tried and failed -> {line}")
        if not self.latin:
            # Louder than the GPU warning on purpose. CPU only costs time;
            # the wrong recognition dictionary corrupts the text permanently
            # and the damage is invisible in the logs -- it just quietly
            # lowers every score that depends on a Spanish word matching.
            print("        !! NOT using the Latin recognition head. Spanish "
                  "accents (á é í ó ú ñ) will be\n"
                  "           unreliable in all OCR'd text, which is a "
                  "PERMANENT corpus defect, not a\n"
                  "           speed problem. Fix before trusting this build:\n"
                  "             pip install -U rapidocr      # PP-OCRv5 Latin "
                  "head\n"
                  "             --ocr-backend tesseract      # handles "
                  "Spanish natively (-l spa)")
            print("        valid values in THIS rapidocr build:")
            print(describe_rapidocr_options())
        if not self.on_gpu and (torch_cuda or providers):
            print("        OCR is on CPU. On Blackwell/CUDA 13 the fix is the "
                  "torch engine, not onnxruntime-gpu:\n"
                  "          pip install -U rapidocr        # needs >=3.0 for "
                  "Det/Cls/Rec.engine_type\n"
                  "        torch itself is already working -- it encodes the "
                  "corpus in step [5/6].")

    @staticmethod
    def _blocks(result) -> list[tuple[float, float, float, float, str]]:
        """RapidOCR 3.x returns an object; 1.x/2.x returned a list of tuples."""
        blocks: list[tuple[float, float, float, float, str]] = []

        boxes = getattr(result, "boxes", None)
        texts = getattr(result, "txts", None)
        pairs = (zip(boxes, texts) if boxes is not None and texts is not None
                 else ((item[0], item[1]) for item in (result or [])))

        for box, text in pairs:
            if not text or not str(text).strip():
                continue
            try:
                xs = [float(point[0]) for point in box]
                ys = [float(point[1]) for point in box]
            except (TypeError, IndexError):
                continue
            blocks.append((min(xs), min(ys), max(xs), max(ys), str(text).strip()))
        return blocks

    @staticmethod
    def _image(page, dpi: int):
        """
        Rasterize straight to a numpy array. PaddleOcrEngine wrote a PNG to
        /tmp for every page; this avoids that round trip entirely.
        """
        import numpy as np
        pixmap = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n)
        if pixmap.n == 4:
            img = img[:, :, :3]
        return img[:, :, ::-1].copy(), float(pixmap.width)   # RGB -> BGR

    def page_text(self, page, dpi: int = 300) -> str:
        from layout import order_blocks
        img, width = self._image(page, dpi)
        blocks = self._blocks(self.reader(img))
        if not blocks:
            return ""
        # Detector order interleaves columns on a multi-column scan.
        # order_blocks() restores reading order AND handles two-up spreads and
        # page furniture -- the bare _xy_cut() this used to call did only the
        # first, so scanned spreads kept splicing the two logical pages.
        ordered = order_blocks(blocks, float(width), float(img.shape[0]))
        return "\n".join(block[4] for block in ordered)

    def image_text(self, path: Path) -> str:
        """A scanned page saved as a loose image is still a scanned page."""
        from layout import order_blocks

        blocks = self._blocks(self.reader(str(path)))
        if not blocks:
            return ""
        width = max(b[2] for b in blocks)
        height = max(b[3] for b in blocks)
        return "\n".join(b[4] for b in order_blocks(blocks, width, height))


class TesseractOcrEngine:
    """CPU fallback, same interface. Kept so --ocr-backend tesseract works."""

    def __init__(self, lang: str = "spa"):
        self.lang = lang

    def page_text(self, page, dpi: int = 300) -> str:
        from extraction import ocr_full_page
        return ocr_full_page(page, lang=self.lang, dpi=dpi)

    def image_text(self, path: Path) -> str:
        from extraction import extract_image
        return extract_image(path)


def ocr_one_file(path: Path, root: Path, doc_id: str, phenomenon: int,
                 engine, max_chars: int, overlap_chars: int,
                 cache_dir: str, dpi: int = 300) -> tuple[list[dict], str]:
    """Phase B: OCR every page of one scanned file, then chunk it."""
    import fitz
    from chunking import chunk_document
    from extraction import Document, clean, FORMAT_BY_SUFFIX, guess_title

    cached = _read_cache(cache_dir, path, root)
    if cached is not None:
        return cached, "cached"

    if FORMAT_BY_SUFFIX.get(path.suffix.lower()) == "img":
        try:
            text = clean(engine.image_text(path))
        except Exception as exc:
            return [], f"image ocr failed: {type(exc).__name__}: {exc}"[:160]
        file_format = "img"
    else:
        try:
            doc_pdf = fitz.open(path)
        except Exception as exc:
            return [], f"open failed: {type(exc).__name__}: {exc}"[:160]

        pages = []
        try:
            for page in doc_pdf:
                pages.append(engine.page_text(page, dpi=dpi))
        except Exception as exc:
            doc_pdf.close()
            return [], f"ocr failed: {type(exc).__name__}: {exc}"[:160]
        doc_pdf.close()
        text = clean("\n\n".join(pages))
        file_format = "pdf"

    if len(text.split()) < 20:
        return [], "ocr produced no usable text"

    doc = Document(doc_id=doc_id, source=str(path.relative_to(root)),
                   file_format=file_format, text=text, phenomenon=phenomenon,
                   file_name=path.name, official_doc_id=doc_id,
                   extra={"extractor": type(engine).__name__,
                          "ocr_lang": engine.lang},
                   title=guess_title(text, path.name))

    chunks = chunk_document(doc, estimate_tokens, max_chars=max_chars,
                            overlap_chars=overlap_chars)
    records = [c.to_dict() for c in chunks]
    _write_cache(cache_dir, path, root, records)
    return records, ""


# --------------------------------------------------------------- planning

def plan_tasks(files, corpus: Path, inventory: dict, profiles: dict,
               cache_str: str, max_chars: int, overlap_chars: int,
               ocr_backend: str, rich_mode: str, docling_mode: str,
               shard_min_pages: int, shard_pages: int,
               exclude_patterns: list[str] | None = None,
               docling_paths: set[str] | None = None,
               images_mode: str = "skip"):
    """
    Split the corpus into Phase A tasks, Phase B (OCR) tasks and Phase C
    (Docling) tasks. A file appears in exactly one of them.

    Returns (cpu_tasks, ocr_tasks, docling_tasks, sharded_docs, stats).
    sharded_docs maps doc_id -> (path, source, phenomenon) for reassembly.
    """
    from extraction import FORMAT_BY_SUFFIX

    cpu_tasks, ocr_tasks, docling_tasks = [], [], []
    sharded_docs: dict[str, tuple[Path, str, int]] = {}
    stats = {"sharded": 0, "shards": 0, "rich": 0, "ocr": 0,
             "docling": 0, "plain": 0, "excluded": 0}
    exclude_patterns = exclude_patterns if exclude_patterns is not None \
        else EXCLUDE_PATTERNS
    excluded_examples: list[str] = []
    skipped_data_images: list[str] = []
    docling_skipped_scanned = 0

    for i, path in enumerate(files):
        relative = str(path.relative_to(corpus))

        pattern = is_excluded(relative, exclude_patterns)
        if pattern:
            stats["excluded"] += 1
            if len(excluded_examples) < 8:
                excluded_examples.append(f"{path.name} ({pattern})")
            continue

        record = inventory.get(relative, {})
        doc_id = record.get("doc_id") or f"DOC-{i:05d}"
        phenomenon = record.get("phenomenon", 0)
        profile = profiles.get(relative, {})
        klass = profile.get("klass", "")
        file_format = FORMAT_BY_SUFFIX.get(path.suffix.lower())
        lang = ocr_language_for(relative)

        if file_format is None:
            continue

        # Loose images. Skipping them is the default: of the nine in this
        # corpus, five are photographs (a launch, a portrait, a header banner)
        # that OCR turns into nothing or into caption noise, and the OCR pass
        # costs a rasterize plus a detector run each.
        #
        # The other four are DATA -- table-5-1-web.jpg, stoplight-chart-
        # execsummary-web.jpg, asat-by-country-2026.jpg -- and CORPUS_ANALISIS
        # ties them to q018, q024, q028 and q031. They are named below so the
        # loss is visible rather than silent. --images ocr keeps all of them.
        if file_format == "img" and images_mode == "skip":
            stats["excluded"] += 1
            if image_carries_data(path.name):
                skipped_data_images.append(path.name)
            continue

        scanned = klass in ("scanned", "sparse") or file_format == "img"
        design = bool(profile) and is_design_heavy(profile)
        complex_layout = bool(profile) and is_complex_layout(profile)

        # ---- Phase C: Docling claims files before anything else, because
        # the whole point of enabling it is to override the cheaper tiers.
        if file_format == "pdf" and docling_mode != "off":
            # `list` is the mode to prefer once you have measured. Docling
            # is NOT uniformly better: across ten complex-layout reports it
            # won on six and lost on four, badly on the SWF files. Routing by
            # rule sends the losers to it too; routing by an explicit list
            # from tools/compare_extractors.py --decide sends only the files
            # where it demonstrably helps.
            wanted = (docling_mode == "all"
                      or (docling_mode == "list"
                          and relative in (docling_paths or set()))
                      or (docling_mode == "design" and design)
                      or (docling_mode == "complex" and complex_layout)
                      or (docling_mode == "scanned" and scanned))

            # SCANNED FILES STAY ON PHASE B, unless Docling was asked for them
            # by name. Docling's OCR engine is EasyOCR, which is a SECOND OCR
            # stack: another model, another set of CUDA wheels, and none of
            # the configuration that took days to get right here -- the
            # PP-OCRv5 Latin recognition head that reads Spanish accents
            # correctly. Installing easyocr to satisfy Docling would also pull
            # its own torch pins, which is exactly the collision that broke
            # this environment once already.
            #
            # Docling is here for LAYOUT on digital pages. There is no reason
            # to let it own the OCR path as a side effect of claiming a file
            # first.
            if wanted and scanned and docling_mode not in ("scanned", "all"):
                wanted = False
                docling_skipped_scanned += 1
            if wanted:
                docling_tasks.append((path, doc_id, phenomenon, lang, scanned,
                                      profile.get("pages") or 0))
                stats["docling"] += 1
                continue

        # ---- Phase B: no text layer worth reading
        if scanned and ocr_backend != "none":
            ocr_tasks.append((path, doc_id, phenomenon, lang,
                              profile.get("pages") or 0))
            stats["ocr"] += 1
            continue

        mode = MODE_BY_CLASS.get(klass, DEFAULT_PDF_MODE)

        if file_format != "pdf":
            size = path.stat().st_size
            cpu_tasks.append(("file", str(path), str(corpus), doc_id,
                              phenomenon, cache_str, max_chars, overlap_chars,
                              mode, lang, 0, None, max(1.0, size / 50_000)))
            stats["plain"] += 1
            continue

        pages = profile.get("pages") or 0
        if not pages:
            from extraction import pdf_page_count
            pages = pdf_page_count(path)

        use_rich = rich_mode == "all" or (rich_mode == "auto" and design)
        kind = "rich" if use_rich else "shard"
        stats["rich" if use_rich else "plain"] += 1

        if pages <= shard_min_pages:
            whole_kind = "rich" if use_rich else "file"
            weight = float(max(pages, 1)) * (3.0 if use_rich else 1.0)
            cpu_tasks.append((whole_kind, str(path), str(corpus), doc_id,
                              phenomenon, cache_str, max_chars, overlap_chars,
                              mode, lang, 0, None, weight))
            continue

        # ---- big file: split into page-range shards
        stats["sharded"] += 1
        weight_per_page = 3.0 if use_rich else 1.0
        if not use_rich:
            # Rich shards emit chunk records directly and are renumbered
            # rather than reassembled from text.
            sharded_docs[doc_id] = (path, relative, phenomenon)
        for start in range(0, pages, shard_pages):
            stop = min(start + shard_pages, pages)
            cpu_tasks.append((kind, str(path), str(corpus), doc_id, phenomenon,
                              cache_str, max_chars, overlap_chars, mode, lang,
                              start, stop, (stop - start) * weight_per_page))
            stats["shards"] += 1

    stats["excluded_examples"] = excluded_examples
    if docling_skipped_scanned:
        print(f"      {docling_skipped_scanned} scanned files kept on Phase B "
              f"(RapidOCR) instead of Docling:\n"
              f"        Docling would OCR them with EasyOCR, a second engine "
              f"without the Latin\n        head. --docling scanned overrides "
              f"this if you want it anyway.")

    if skipped_data_images:
        print(f"      NOTE: {len(skipped_data_images)} skipped images look "
              f"like DATA, not decoration:")
        for name in skipped_data_images:
            print(f"        {name}")
        print("      CORPUS_ANALISIS ties these to q018/q024/q028/q031. "
              "--images ocr keeps them.")

    return cpu_tasks, ocr_tasks, docling_tasks, sharded_docs, stats


# --------------------------------------------------------------- build

def preflight(docling_mode: str = "off", docling_device: str = "cuda") -> None:
    """
    Check the environment BEFORE extraction, not at step [5/6].

    Extraction takes hours. Encoding needs torch. A build that extracts the
    whole corpus and only then discovers torch is broken has thrown all of it
    away -- which is exactly what happened when installing docling pulled
    torch's pinned nvidia-* wheels out from under it and every phase
    afterwards reported per-file failures instead of one environment fault.

    Fails on torch, warns on CUDA. torch on CPU is slow but correct; torch
    absent produces nothing at all.
    """
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            f"PREFLIGHT: torch is not importable.\n"
            f"  {exc}\n"
            f"  Step [5/6] cannot encode anything without it, so this run "
            f"would waste hours of extraction.\n"
            f"  'undefined symbol: nccl*' means torch's pinned nvidia-* "
            f"wheels were replaced by another package's.\n"
            f"  Reinstall torch for YOUR CUDA version (check nvidia-smi; "
            f"Blackwell/sm_120 needs cu128 or newer):\n"
            f"      pip install --force-reinstall torch "
            f"--index-url https://download.pytorch.org/whl/cu128\n"
            f"  Verify: python -c \"import torch; print(torch.cuda.is_available())\"")

    cuda = torch.cuda.is_available()
    print(f"PREFLIGHT: torch {torch.__version__}, cuda={cuda}"
          + (f", device={torch.cuda.get_device_name(0)}" if cuda else ""))

    if not cuda:
        print("  WARNING: no CUDA. Encoding 145k chunks on CPU takes hours "
              "rather than minutes.\n"
              "           The build is still CORRECT -- same models, same "
              "vectors -- just slow.")
        if docling_mode != "off" and docling_device == "cuda":
            raise SystemExit(
                "  --docling-device cuda with no working CUDA. Docling would "
                "fail on every file.\n"
                "  Use --docling-device cpu, or --docling off, or fix CUDA.")


def build(corpus: Path, out: Path,encoder_names: list[str] | str, batch_size: int = 128,
          max_chars: int = 1000, overlap_chars: int = 350,
          inventory_path: Path | None = None, triage_path: Path | None = None,
          workers: int | None = None,
          cache_dir: Path | None = Path(".cache_extract"),
          fp16: bool = True, limit: int | None = None,
          ocr_backend: str = "paddle", ocr_dpi: int = 300,
          ocr_engine: str = "auto",
          file_timeout: float = 900.0, rich_mode: str = "auto",
          docling_mode: str = "off", docling_device: str = "cuda",
          docling_list: Path | None = None,
          drop_translations: bool = True,
          images_mode: str = "skip",
          shard_order: str = "largest-first",
          shard_min_pages: int = SHARD_MIN_PAGES,
          shard_pages: int = SHARD_PAGES,
          max_seq_length: int | None = 320,
          progress: str = "file",
          slow_log: Path | None = Path("slow_files.txt"),
          slow_threshold: float = 5.0,
          exclude_patterns: list[str] | None = None) -> None:

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    if isinstance(encoder_names, str):
        encoder_names = [encoder_names]

    preflight(docling_mode, docling_device)

    workers = workers or _default_workers()
    cache_str = str(cache_dir) if cache_dir else ""
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    timings = TimingLog(slow_log, slow_threshold)

    inventory = {}
    if inventory_path and inventory_path.exists():
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        print(f"Using official inventory: {len(inventory)} entries")

    profiles: dict[str, dict] = {}
    if triage_path and triage_path.exists():
        triage = json.loads(triage_path.read_text(encoding="utf-8"))
        profiles = {f["path"]: f for f in triage["files"]}
        counts: dict[str, int] = {}
        for f in profiles.values():
            counts[f["klass"]] = counts.get(f["klass"], 0) + 1
        print(f"Using triage: {counts}")
    else:
        print("No triage.json -- every PDF is treated as digital and NOTHING "
              "is routed to OCR. Run triage.py first.")

    files = sorted(p for p in corpus.rglob("*") if p.is_file())

    if drop_translations:
        translations = redundant_translations(files, corpus)
        if translations:
            print(f"Dropping {len(translations)} non-Spanish/English "
                  f"translations of documents already in the corpus:")
            for relative in sorted(translations):
                print(f"  {relative}")
            files = [f for f in files
                     if str(f.relative_to(corpus)) not in translations]

    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"No files under {corpus.resolve()}")

    docling_paths: set[str] = set()
    if docling_mode == "list":
        if not docling_list or not Path(docling_list).exists():
            raise SystemExit(
                "--docling list needs --docling-list FILE.\n"
                "  Produce it with:\n"
                "    python tools/compare_extractors.py --corpus <corpus> "
                "--triage triage.json \\\n"
                "        --decide docling_files.txt")
        docling_paths = {line.strip() for line
                         in Path(docling_list).read_text(encoding="utf-8").splitlines()
                         if line.strip()}
        print(f"Docling routing list: {len(docling_paths)} files from "
              f"{docling_list}")

    cpu_tasks, ocr_tasks, docling_tasks, sharded_docs, stats = plan_tasks(
        files, corpus, inventory, profiles, cache_str, max_chars,
        overlap_chars, ocr_backend, rich_mode, docling_mode,
        shard_min_pages, shard_pages, exclude_patterns,
        docling_paths=docling_paths, images_mode=images_mode)

    if docling_mode == "list" and len(docling_tasks) < len(docling_paths):
        print(f"  WARNING: {len(docling_paths) - len(docling_tasks)} paths in "
              f"the list matched no file. The list stores paths RELATIVE to "
              f"--corpus;\n           check it was produced against this "
              f"same corpus root.")

    # Longest-processing-time-first: a long task started last has nothing to
    # overlap with; started first it is absorbed while short tasks fill in.
    cpu_tasks.sort(key=lambda t: t[-1], reverse=shard_order == "largest-first")

    print(f"      {len(files)} files -> {len(cpu_tasks)} Phase A tasks, "
          f"{len(ocr_tasks)} Phase B (OCR), {len(docling_tasks)} Phase C "
          f"(Docling)")
    print(f"      routing: {stats['plain']} plain, {stats['rich']} rich-layout, "
          f"{stats['docling']} docling, {stats['sharded']} large files split "
          f"into {stats['shards']} shards, {stats['ocr']} OCR")
    if stats.get("excluded"):
        print(f"      excluded {stats['excluded']} files (catalogs, scrape "
              f"manifests, PubMed listings, redundant .pbf tiles, challenge "
              f"files):")
        for example in stats.get("excluded_examples", []):
            print(f"        - {example}")

    if docling_tasks:
        docling_pages = sum(t[5] for t in docling_tasks)
        print(f"      Docling will process ~{docling_pages} pages. At a "
              f"typical 0.4-2 s/page on GPU that is roughly "
              f"{docling_pages*0.4/60:.0f}-{docling_pages*2/60:.0f} minutes.")
        if docling_mode == "all":
            print("      WARNING: --docling all runs a neural pass over the "
                  "WHOLE corpus. Try --docling design first.")

    records: list[dict] = []
    failures: list[tuple[str, str]] = []
    shard_texts: dict[str, dict[int, str]] = {}
    rich_shard_records: dict[str, dict[int, list[dict]]] = {}

    # ---------------- [1/6] Phase A: CPU pool
    print(f"[1/6] Phase A: {len(cpu_tasks)} tasks across {workers} workers "
          f"({os.cpu_count()} logical CPUs) ...")
    cached = done = 0
    started = time.time()
    last_report = started

    # spawn, not fork: a spawned child starts from a clean interpreter and
    # cannot inherit a CUDA context or a held lock from the parent.
    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=workers, initializer=_pin_worker_threads,
                             mp_context=context) as pool:
        futures = {pool.submit(process_one, t): t for t in cpu_tasks}
        pending = dict(futures)

        for future in as_completed(futures, timeout=file_timeout * max(len(cpu_tasks), 1)):
            task = pending.pop(future, None)
            kind, doc_id, first_page, payload, note, seconds = future.result()
            done += 1
            if note == "cached":
                cached += 1
            elif note:
                failures.append((task[1] if task else doc_id, note))

            # task[11] is last_page: not None means this was a page-range
            # shard rather than a whole file.
            is_shard = task is not None and task[11] is not None
            span = f"{task[10]}-{task[11]}" if is_shard else "-"
            source = str(Path(task[1]).relative_to(corpus)) if task else doc_id
            n_chunks = len(payload) if isinstance(payload, list) else 0

            if kind == "shard":
                shard_texts.setdefault(doc_id, {})[first_page] = payload or ""
                n_chunks = 0
            elif kind == "rich" and is_shard:
                rich_shard_records.setdefault(doc_id, {})[first_page] = payload or []
            else:
                records.extend(payload or [])

            timings.add(seconds, "A", kind, source, doc_id, span, n_chunks, note)

            if progress == "file":
                label = f"{kind}[{span}]" if is_shard else kind
                print(_progress_line(done, len(cpu_tasks), seconds, doc_id,
                                     label, n_chunks, Path(source).name, note),
                      flush=True)
            elif done % 25 == 0 or done == len(cpu_tasks) or time.time() - last_report > 20:
                last_report = time.time()
                elapsed = time.time() - started
                rate = done / max(elapsed, 1e-3)
                print(f"      {done}/{len(cpu_tasks)}  {elapsed:5.0f}s  "
                      f"~{(len(cpu_tasks) - done) / max(rate, 1e-3):5.0f}s left  "
                      f"{len(records):7d} chunks  ({cached} cached)", flush=True)

    print(f"      Phase A done in {time.time()-started:.0f}s "
          f"({cached} tasks served from cache)")

    # ---------------- [2/6] reassemble sharded documents
    if shard_texts or rich_shard_records:
        print(f"[2/6] Reassembling {len(shard_texts)} sharded documents ...")
        merge_args = []
        for doc_id, parts in shard_texts.items():
            path, source, phenomenon = sharded_docs.get(doc_id, (None, "", 0))
            if path is None:
                continue
            text = "\n\n".join(parts[k] for k in sorted(parts) if parts[k])
            if len(text.split()) < 20:
                failures.append((str(path), "sharded doc produced no text"))
                continue
            merge_args.append((doc_id, source, phenomenon, text, max_chars,
                               overlap_chars, cache_str, str(path), str(corpus)))

        if merge_args:
            with ProcessPoolExecutor(max_workers=min(workers, len(merge_args)),
                                     initializer=_pin_worker_threads,
                                     mp_context=context) as pool:
                for doc_id, doc_records, note, seconds in pool.map(chunk_merged,
                                                                   merge_args):
                    if note:
                        failures.append((doc_id, note))
                    records.extend(doc_records)
                    timings.add(seconds, "A2", "merge-chunk", doc_id, doc_id,
                                "-", len(doc_records), note)
                    if progress == "file":
                        print(f"      merged {doc_id:<18} {seconds:6.2f}s  "
                              f"{len(doc_records):5d} chunks", flush=True)

        # Rich shards already carry chunks; renumber positions across shards
        # so chunk_ids stay unique and ordered by page.
        for doc_id, parts in rich_shard_records.items():
            position = 0
            for key in sorted(parts):
                for record in parts[key]:
                    record["posicion"] = position
                    record["chunk_id"] = f"{doc_id}-chunk-{position:04d}"
                    position += 1
                    records.append(record)
    else:
        print("[2/6] No sharded documents to reassemble.")

    # ---------------- [3/6] Phase B: GPU OCR, single process
    if ocr_tasks:
        print(f"[3/6] Phase B: OCR-ing {len(ocr_tasks)} files with "
              f"{ocr_backend} ...")
        engines: dict[str, object] = {}
        engine_classes = {"rapid": RapidOcrEngine, "paddle": PaddleOcrEngine,
                          "tesseract": TesseractOcrEngine}
        backend_failed = ""
        started = time.time()
        # FIX: was read at "consecutive_failures += 1" below without ever
        # being initialised in THIS loop -- only Phase C's loop further down
        # sets it to 0. Phase B ran fine as long as every file's `note` was
        # falsy or "cached" (nothing ever hit the += 1 line); the first real
        # OCR failure or non-cached note then hit an UnboundLocalError
        # instead of being recorded and counted. Each phase counts its own
        # consecutive failures, so each needs its own counter.
        consecutive_failures = 0

        for n, (path, doc_id, phenomenon, tess_lang, pages) in enumerate(ocr_tasks, 1):
            task_started = time.time()

            # A backend that will not load is a DEPENDENCY problem, not a
            # document problem. By this point Phases A and A2 are finished and
            # cached; aborting here throws away a whole extracted corpus for
            # the sake of 68 files, and leaves no index at all. Record it,
            # skip the phase, let the encoder run on what we have.
            if backend_failed:
                failures.append((str(path), f"OCR unavailable: {backend_failed}"))
                continue

            key = (tess_lang if ocr_backend == "tesseract"
                   else to_paddle_lang(tess_lang))
            engine = engines.get(key)
            if engine is None:
                try:
                    if ocr_backend == "rapid":
                        engine = RapidOcrEngine(
                            key, use_gpu=ocr_engine != "cpu",
                            engine=ocr_engine)
                    else:
                        engine = engine_classes[ocr_backend](key)
                except Exception as exc:
                    backend_failed = f"{type(exc).__name__}: {exc}"[:120]
                    print(f"\n      !! {ocr_backend} failed to initialise: "
                          f"{backend_failed}")
                    print(f"      Skipping Phase B. {len(ocr_tasks)} scanned "
                          f"files will be ABSENT from the index. The rest of "
                          f"the pipeline continues.")
                    print(f"      Fix the backend and re-run: Phases A/A2 are "
                          f"cached, so only these {len(ocr_tasks)} files "
                          f"repeat.\n")
                    failures.append((str(path), f"OCR init: {backend_failed}"))
                    continue
                engines[key] = engine

            doc_records, note = ocr_one_file(path, corpus, doc_id, phenomenon,
                                             engine, max_chars, overlap_chars,
                                             cache_str, dpi=ocr_dpi)
            if note and note != "cached":
                failures.append((str(path), note))
                consecutive_failures += 1
                # Twenty in a row is not twenty bad PDFs. Stop and say so
                # rather than burning the queue and reporting a total that
                # looks like a corpus problem.
                if consecutive_failures >= 20:
                    raise SystemExit(
                        f"[3/6] 20 consecutive OCR failures, last was:\n"
                        f"      {note}\n"
                        f"      Aborting: this is an environment fault, not a "
                        f"file fault. Re-run with --ocr-backend none to build "
                        f"without Phase B, or fix the install -- the cache "
                        f"keeps everything already extracted.")
            else:
                consecutive_failures = 0
            records.extend(doc_records)

            seconds = time.time() - task_started
            source = str(path.relative_to(corpus))
            timings.add(seconds, "B", f"ocr-{ocr_backend}", source, doc_id,
                        str(pages or "-"), len(doc_records), note)
            print(_progress_line(n, len(ocr_tasks), seconds, doc_id,
                                 f"ocr/{pages}p", len(doc_records),
                                 path.name, note), flush=True)

        print(f"      Phase B done in {time.time()-started:.0f}s")
    else:
        print("[3/6] No files routed to OCR.")

    # ---------------- [4/6] Phase C: Docling, single process
    if docling_tasks:
        print(f"[4/6] Phase C: Docling on {len(docling_tasks)} files "
              f"({docling_device}) ...")
        from docling_extract import DoclingEngine, docling_one_file

        # PREFLIGHT. Phase C failed 268 files in 11 seconds once, every one
        # with the same torch ImportError, and the run continued to the
        # encoder before dying there -- after discarding every chunk those
        # files would have produced. An environment fault is not a per-file
        # failure and must not be retried 268 times or swallowed by the
        # per-file try/except.
        try:
            import torch
            if docling_device == "cuda" and not torch.cuda.is_available():
                raise SystemExit(
                    "[4/6] --docling-device cuda but torch.cuda.is_available() "
                    "is False. Use --docling-device cpu, or fix the CUDA "
                    "install before spending the run.")
        except ImportError as exc:
            raise SystemExit(
                f"[4/6] torch is not importable: {exc}\n"
                f"      Docling needs it, and so does step [5/6]. Installing "
                f"docling can move torch's pinned nvidia-* wheels out from "
                f"under it; 'undefined symbol: nccl*' is that.\n"
                f"      Reinstall torch for your CUDA version, verify with "
                f"`python -c \"import torch; print(torch.cuda.is_available())\"`, "
                f"then re-run. The extraction cache is intact, so nothing is "
                f"recomputed.")

        docling_engines: dict[str, object] = {}
        started = time.time()
        consecutive_failures = 0

        for n, (path, doc_id, phenomenon, tess_lang, scanned, pages) in \
                enumerate(docling_tasks, 1):
            task_started = time.time()
            key = f"{scanned}|{tess_lang}"
            engine = docling_engines.get(key)
            if engine is None:
                engine = DoclingEngine(device=docling_device, do_ocr=scanned,
                                       ocr_lang=to_easyocr_langs(tess_lang))
                docling_engines[key] = engine

            cached_records = _read_cache(cache_str, path, corpus, "|docling")
            if cached_records is not None:
                doc_records, note = cached_records, "cached"
            else:
                doc_records, note = docling_one_file(
                    path, corpus, doc_id, phenomenon, engine, estimate_tokens,
                    max_chars, overlap_chars)
                if not note:
                    _write_cache(cache_str, path, corpus, doc_records, "|docling")

            if note and note != "cached":
                failures.append((str(path), note))
            records.extend(doc_records)

            seconds = time.time() - task_started
            source = str(path.relative_to(corpus))
            timings.add(seconds, "C", "docling", source, doc_id,
                        str(pages or "-"), len(doc_records), note)
            print(_progress_line(n, len(docling_tasks), seconds, doc_id,
                                 f"docling/{pages}p", len(doc_records),
                                 path.name, note), flush=True)

        print(f"      Phase C done in {time.time()-started:.0f}s")
    elif docling_mode != "off":
        print("[4/6] Docling enabled but no file matched the selection.")
    else:
        print("[4/6] Docling disabled.")

    timings.write()

    if not records:
        raise SystemExit("No chunks were produced. Check the corpus path.")
    print(f"      TOTAL: {len(records)} chunks, {len(failures)} tasks failed")
    ocr_missing = sum(1 for _p, n in failures if n.startswith("OCR"))
    if ocr_missing:
        print(f"      WARNING: {ocr_missing} scanned files are MISSING from "
              f"this index. Most are Alertas_Tempranas, the densest Fenomeno 3 "
              f"material (q033-q050). Do not treat F1@3 on those queries as "
              f"meaningful until Phase B succeeds.")
    for path, note in failures[:15]:
        print(f"        FAILED {Path(path).name[:50]}: {note}")

    records.sort(key=lambda r: (r["doc_id"], r["posicion"]))

    # ---------------- [5/6] encoders
    # Safe to touch CUDA now: no pool is open, nothing else will fork.
    #
    # Every encoder is fed the SAME `records` list in the same order. This is
    # what makes RRF valid: fuse_rrf keys on chunk_id, so if two indexes came
    # from separate runs, one file failing OCR in run A and succeeding in run
    # B would shift every chunk_id after it and the fusion would combine
    # unrelated rows -- silently, with plausible-looking output.
    import torch
    import faiss
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[5/6] Encoding with {len(encoder_names)} encoder(s) on "
          f"{device.upper()}: {', '.join(encoder_names)}")

    # chunk_id = THE FAISS INTERNAL ID. Assigned here, once, over the shared
    # record list -- so it is identical across every encoder folder, which is
    # what makes cross-encoder fusion on chunk_id sound.
    #
    # FAQ: "aprovecho para indicar que deberían usar como chunk_id el mismo
    # obtenido del indice FAISS." IndexFlatIP assigns ids 0..n-1 in insertion
    # order, and 1.4 already requires metadata.jsonl line order to match those
    # ids, so line i IS id i. Writing it explicitly costs nothing and removes
    # a discrepancy a judge would otherwise have to reason about.
    #
    # This CANNOT be done in chunking.py: a chunk does not know its global
    # position until every file has been extracted and the records ordered.
    #
    # The descriptive id survives as `chunk_uid`. It is what makes a debugging
    # session tractable -- "F3-MAPPOEA-014-chunk-0007" says which document and
    # which passage, "137482" says nothing -- and Table 1 permits extra fields.
    for position, record in enumerate(records):
        record["chunk_uid"] = record["chunk_id"]
        record["chunk_id"] = str(position)
    print(f"      chunk_id renumbered to the FAISS index: 0..{len(records)-1}")

    for encoder_name in encoder_names:
        config = dict(ENCODERS.get(encoder_name,
                                   {"doc_prefix": "", "query_prefix": ""}))
        remote = bool(config.get("trust_remote_code"))
        print(f"      loading {encoder_name} ...")
        model = SentenceTransformer(encoder_name, device=device,
                                    trust_remote_code=remote)
        if fp16 and device == "cuda":
            model = model.half()

        # Chunks target ~1000 characters. Leaving the model at its 512
        # default pads every batch to 512 and burns encode time on padding.
        # Check the truncation percentage below before lowering this: a
        # truncated chunk loses its closing sentences from the VECTOR while
        # keeping them in metadata.jsonl, so the audit finds the text and
        # the ranker never does.
        if max_seq_length:
            model.max_seq_length = min(max_seq_length,
                                       model.get_max_seq_length() or 512)
        model_limit = model.get_max_seq_length() or 512
        dim = model.get_sentence_embedding_dimension()
        print(f"      dim={dim}  max_seq_len={model_limit}  "
              f"fp16={fp16 and device == 'cuda'}")

        texts = [
            config["doc_prefix"] + (f"{r['contexto']}. {r['texto']}"
                                    if r.get("contexto") else r["texto"])
            for r in records
        ]

        # num_tokens is per-encoder (different tokenizers, different counts),
        # so it is recomputed here and written into that encoder's own
        # metadata.jsonl rather than shared across folders.
        lengths: list[int] = []
        for i in range(0, len(texts), 1000):
            encoded = model.tokenizer(texts[i:i + 1000], add_special_tokens=True,
                                      truncation=False)["input_ids"]
            lengths.extend(len(ids) for ids in encoded)
        for record, n in zip(records, lengths):
            record["num_tokens"] = n
        over = sum(1 for n in lengths if n > model_limit)
        print(f"      {over} chunks exceed {model_limit} tokens "
              f"({100*over/max(len(lengths),1):.1f}%; encoder truncates, "
              f"stored text unchanged)")
        if over > len(lengths) * 0.02:
            print("      WARNING: over 2% truncated. Raise --max-seq-length.")

        # ---------------- [6/6] encode + index
        print(f"[6/6] Encoding {len(texts)} chunks ...")
        started = time.time()
        vectors = model.encode(
            texts, batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype("float32")
        print(f"      encoded in {time.time() - started:.0f}s")

        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        target = out / f"encoder_{encoder_name.split('/')[-1]}"
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(target / "index.faiss"))

        with (target / "metadata.jsonl").open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        (target / "encoder.json").write_text(json.dumps(
            {"model": encoder_name, "dim": dim, "n_vectors": int(index.ntotal),
             **config}, ensure_ascii=False, indent=2), encoding="utf-8")

        assert index.ntotal == len(records), "Index/metadata misalignment"
        print(f"OK  {index.ntotal} vectors -> {target}")

        # Free the GPU before the next encoder loads. Two large models
        # resident at once is how a run that encodes fine alone OOMs when
        # chained.
        del model, vectors, index
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("entrega/base_vectorial"))
    parser.add_argument("--encoder", nargs="+", dest="encoders",
                        default=["intfloat/multilingual-e5-large"],
                        help="One or more encoders. All of them are encoded "
                             "from the SAME chunk set in this run, which is "
                             "what makes RRF in generador.py valid.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument("--overlap-chars", type=int, default=350)
    parser.add_argument("--inventory", type=Path, default=Path("inventory.json"))
    parser.add_argument("--triage", type=Path, default=Path("triage.json"))
    parser.add_argument("--workers", type=int, default=None,
                        help="CPU processes for Phase A. Default: physical "
                             "cores - 1. Does not affect Phases B/C, which "
                             "are single-process by design.")
    parser.add_argument("--ocr-backend",
                        choices=["rapid", "paddle", "tesseract", "none"],
                        default="rapid",
                        help="rapid=PP-OCRv5 via ONNX Runtime (DEFAULT; same "
                             "models as paddle, no PaddlePaddle framework, "
                             "does not collide with torch). paddle=PaddleOCR "
                             "(needs its own venv on Blackwell). "
                             "tesseract=CPU fallback. none=skip scanned files.")
    parser.add_argument("--ocr-dpi", type=int, default=300)
    parser.add_argument("--ocr-engine", choices=["auto", "torch", "onnx", "cpu"],
                        default="auto",
                        help="Inference engine for --ocr-backend rapid. "
                             "auto=torch+CUDA if available, else onnx+CUDA, "
                             "else CPU. On Blackwell (sm_120) / CUDA 13 the "
                             "onnx GPU path does not work -- see "
                             "RapidOcrEngine. cpu=force CPU.")
    parser.add_argument("--rich-layout", choices=["auto", "off", "all"],
                        default="auto", dest="rich_mode",
                        help="auto=span-level parsing for design-heavy PDFs "
                             "only (default). all=every PDF. off=never.")
    parser.add_argument("--docling",
                        choices=["off", "list", "design", "complex", "scanned", "all"],
                        default="off", dest="docling_mode",
                        help="Trained-layout-model extraction. off=default. "
                             "design=the design-heavy subset (recommended). "
                             "scanned=Docling instead of PaddleOCR on scanned "
                             "files. all=every PDF (hours).")
    parser.add_argument("--images", choices=["skip", "ocr"], default="skip",
                        dest="images_mode",
                        help="loose .jpg/.png/.avif files. skip (default) "
                             "drops them; ocr rasterizes and reads them. Most "
                             "are photographs, but a few are data tables -- "
                             "the skipped ones are named at plan time.")
    parser.add_argument("--keep-translations", action="store_true",
                        help="index Chinese/Russian translations of documents "
                             "that also exist in Spanish or English. Off by "
                             "default: they are near-duplicates that can "
                             "occupy a document slot and never match.")
    parser.add_argument("--docling-list", type=Path, default=None,
                        help="file of corpus-relative paths for --docling "
                             "list, from tools/compare_extractors.py --decide")
    parser.add_argument("--docling-device", choices=["cuda", "cpu"],
                        default="cuda")
    parser.add_argument("--shard-min-pages", type=int, default=SHARD_MIN_PAGES,
                        help="PDFs above this page count are split across "
                             "workers instead of held by one.")
    parser.add_argument("--shard-pages", type=int, default=SHARD_PAGES)
    parser.add_argument("--shard-order",
                        choices=["largest-first", "smallest-first"],
                        default="largest-first")
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="Encoder input length. Chunks are ~300 tokens; "
                             "the 512 default pays for padding. 0 disables.")
    parser.add_argument("--progress", choices=["file", "summary"],
                        default="file",
                        help="file=one line per task (default). "
                             "summary=periodic aggregate only.")
    parser.add_argument("--slow-log", type=Path, default=Path("slow_files.txt"),
                        help="Where to write the per-task timing report.")
    parser.add_argument("--slow-threshold", type=float, default=5.0,
                        help="Seconds above which a task counts as slow in "
                             "the report header.")
    parser.add_argument("--exclude", nargs="+", default=[], metavar="GLOB",
                        help="Extra fnmatch patterns, matched against the FILE "
                             "NAME (not the path), appended to "
                             "EXCLUDE_PATTERNS. Lets you ablate a file class "
                             "without editing source, e.g. --exclude '*.csv' "
                             "or --exclude '*clinicaltrials-*'. Rebuilding is "
                             "not needed to test the idea -- run it with "
                             "--limit first and read the excluded count.")
    parser.add_argument("--no-exclusions", action="store_true",
                        help="index catalogs, scrape manifests, PubMed "
                             "listings and .pbf tiles too. Not recommended: "
                             "see EXCLUDE_PATTERNS for why each is dropped.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache_extract"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N files (smoke test)")
    parser.add_argument("--file-timeout", type=float, default=900.0)
    args = parser.parse_args()

    build(args.corpus, args.out, args.encoders, batch_size=args.batch_size,
          max_chars=args.max_chars, overlap_chars=args.overlap_chars,
          inventory_path=args.inventory, triage_path=args.triage,
          workers=args.workers,
          cache_dir=None if args.no_cache else args.cache_dir,
          fp16=not args.no_fp16, limit=args.limit,
          ocr_backend=args.ocr_backend, ocr_dpi=args.ocr_dpi,
          ocr_engine=args.ocr_engine,
          file_timeout=args.file_timeout, rich_mode=args.rich_mode,
          docling_mode=args.docling_mode, docling_device=args.docling_device,
          docling_list=args.docling_list,
          drop_translations=not args.keep_translations,
          images_mode=args.images_mode,
          shard_order=args.shard_order,
          shard_min_pages=args.shard_min_pages, shard_pages=args.shard_pages,
          max_seq_length=args.max_seq_length or None,
          progress=args.progress,
          slow_log=args.slow_log if str(args.slow_log) else None,
          slow_threshold=args.slow_threshold,
          exclude_patterns=([] if args.no_exclusions
                            else EXCLUDE_PATTERNS + list(args.exclude)))