"""
extraction.py — Per-format text extraction (CODEFEST AD ASTRA 2026, section 2).

Each file is one document with a unique doc_id. Returns a Document object with
cleaned text plus source metadata.

PATCH NOTES
  1. NEW MODE "digital" -- and it is now the default for every PDF.
     The old "auto" mode called ocr_full_page() on any page with < 100 chars
     of text layer. Digital reports are full of such pages: full-bleed figure
     spreads, chapter dividers, image covers, near-blank pages. On a 36k-page
     corpus that silently fired thousands of 300-dpi Tesseract runs and turned
     a 60-second job into an overnight one. "digital" does the useful half of
     "auto" (per-page column detection -> XY-cut) and CANNOT invoke OCR. OCR
     now happens only in build_index.py's Phase B, on the ~59 files triage
     actually flagged.

  2. "plain" is no longer the routing target for digital files.
     triage.json reports 490 of 760 PDFs with multi-column sampled pages, but
     MODE_BY_CLASS mapped digital -> "plain", which never applies XY-cut.
     A 4-column spread came out read straight across: fluent, ordered
     plausibly, and completely wrong -- the worst failure mode, because
     nothing flags it for review. The ground-truth-alignment argument for
     plain get_text() holds on single-column prose, which is what the 15
     annotated reference fragments were; it does not license plain extraction
     on a magazine spread. "digital" keeps plain output on 1-column pages and
     only reorders where there is a real gutter.

  3. PAGE RANGES. extract_pdf() takes first_page/last_page so build_index can
     shard a 1330-page PDF across many workers instead of parking it in one.

  4. .PBF IS NO LONGER SILENTLY DROPPED. See pbf_extract.py.

Requires:
    pip install pymupdf pdfplumber pytesseract pillow beautifulsoup4 lxml \
                pandas openpyxl mapbox-vector-tile
    sudo apt install tesseract-ocr tesseract-ocr-spa tesseract-ocr-por
    (tesseract is only needed for --ocr-backend tesseract; the default
     PaddleOCR GPU path does not use it)
"""

from __future__ import annotations

import io
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF

from layout import (detect_columns, document_is_two_up,
                    page_text_in_reading_order)


# ---------------------------------------------------------------- structure

@dataclass
class Document:
    doc_id: str
    source: str          # original file name or URL
    file_format: str     # pdf | html | json | csv | xlsx | md | txt | img | pbf
    text: str
    phenomenon: int = 0  # 1, 2 or 3
    file_name: str = ""          # bare original filename (hedge, see chunking)
    official_doc_id: str = ""    # DOC_ID from Indice_Datos_Codefest.xlsx
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- cleaning

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t\u00a0\u2007\u202f]+")
_BLANK_LINES = re.compile(r"\n{3,}")
# A single newline is a layout line break, not a paragraph break
_WRAP = re.compile(r"(?<!\n)\n(?!\n)")
# Word split by a hyphen at end of line: "innova-\ntion" -> "innovation"
_HYPHEN = re.compile(r"(\w)-\n(\w)")


def clean(text: str) -> str:
    """
    Minimal normalization only. NFC matters: 35 JSON files in this corpus use
    NFD combining accents, and the same word in NFC and NFD produces different
    vectors. Everything else here is whitespace hygiene -- no content is
    removed, because the reference fragments retain page numbers and footnote
    markers.
    """
    text = unicodedata.normalize("NFC", text)
    text = _CTRL.sub(" ", text)
    text = _HYPHEN.sub(r"\1\2", text)     # before touching line breaks
    text = _BLANK_LINES.sub("\n\n", text)
    text = _WRAP.sub(" ", text)           # join lines within a paragraph
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def strip_boilerplate(pages: list[str], threshold: float = 0.6) -> list[str]:
    """Drop headers/footers repeating on >= threshold of pages (section 2.2)."""
    if len(pages) < 4:
        return pages

    headers, footers = Counter(), Counter()
    for page in pages:
        lines = [l.strip() for l in page.split("\n") if l.strip()]
        if not lines:
            continue
        headers[lines[0]] += 1
        footers[lines[-1]] += 1

    minimum = threshold * len(pages)
    repeated = {t for t, n in (headers + footers).items()
                if n >= minimum and len(t) < 120}

    result = []
    for page in pages:
        lines = [l for l in page.split("\n") if l.strip() not in repeated]
        # bare page numbers
        lines = [l for l in lines if not re.fullmatch(r"\s*\d{1,4}\s*", l)]
        result.append("\n".join(lines))
    return result


# ---------------------------------------------------------------- PDF

def _ocr(image_bytes: bytes, langs: str = "eng+spa") -> str:
    """OCR one embedded image. Returns "" when OCR is unavailable or noisy."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.width < 150 or img.height < 150:     # icons, logos, bullets
            return ""
        text = clean(pytesseract.image_to_string(img, lang=langs))
        # reject OCR noise: too few words or too many odd characters
        if len(text.split()) < 8:
            return ""
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
        return text if alpha_ratio > 0.7 else ""
    except Exception:
        return ""


def _tables_as_markdown(pdf_path: Path, page_numbers: list[int]) -> dict[int, list[str]]:
    """
    Extract tables as Markdown for a SET of pages in one pdfplumber open.

    The previous signature took a single page number and opened the whole PDF
    each time, which is O(n^2) in page count -- on a 1330-page file that is
    1330 full document parses. Never re-enable table extraction with the old
    per-page open.
    """
    try:
        import pdfplumber
    except ImportError:
        return {}

    out: dict[int, list[str]] = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number in page_numbers:
                if page_number >= len(pdf.pages):
                    continue
                tables = []
                for raw in pdf.pages[page_number].extract_tables():
                    rows = [[(cell or "").replace("\n", " ").strip() for cell in row]
                            for row in raw if any(row)]
                    if len(rows) < 2:
                        continue
                    header, body = rows[0], rows[1:]
                    md = ["| " + " | ".join(header) + " |",
                          "| " + " | ".join("---" for _ in header) + " |"]
                    md += ["| " + " | ".join(r) + " |" for r in body]
                    tables.append("\n".join(md))
                if tables:
                    out[page_number] = tables
    except Exception:
        pass
    return out


def ocr_full_page(page, lang: str = "eng+spa", dpi: int = 300) -> str:
    """
    OCR an entire page. Scanned pages are a single full-bleed image, so the
    per-image OCR used for figures finds nothing useful: rasterize instead.
    300 dpi is the accuracy/speed knee for Tesseract on report scans.

    ONLY called from mode="ocr", i.e. only from build_index.py Phase B or an
    explicit --ocr-backend tesseract run. Never from the digital path.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        pixmap = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        return pytesseract.image_to_string(img, lang=lang).strip()
    except Exception:
        return ""


def pdf_page_count(path: Path) -> int:
    """Page count without extracting anything. Used for shard planning."""
    try:
        doc = fitz.open(path)
        n = doc.page_count
        doc.close()
        return n
    except Exception:
        return 0


def extract_pdf(path: Path, mode: str = "digital", ocr_lang: str = "eng+spa",
                figure_ocr: bool = False, tables: bool = False,
                drop_boilerplate: bool = False,
                first_page: int = 0, last_page: int | None = None) -> str:
    """
    Extract a PDF's text, optionally restricted to pages [first_page, last_page).

    mode:
      "digital" per page: plain text layer, switching to XY-cut reading order
                when the page has more than one column. NEVER calls OCR.
                This is the default and the correct choice for anything triage
                did not flag as scanned.
      "plain"   always the raw PyMuPDF text layer. Kept for reproducing the
                organizers' single-column extraction exactly, e.g. when
                validating against the annotated fragments.
      "layout"  always XY-cut reading order.
      "auto"    legacy: like "digital" but falls back to full-page OCR on
                text-poor pages. RETAINED ONLY FOR EXPLICIT USE. Routing
                anything here by default is what made the pipeline take hours.
      "ocr"     always full-page OCR (Phase B / tesseract backend).

    figure_ocr, tables and drop_boilerplate stay off by default: each inserts
    or removes text the reference fragments do not have, moving output away
    from the ground truth.
    """
    doc = fitz.open(path)
    total = doc.page_count
    start = max(0, first_page)
    stop = total if last_page is None else min(total, last_page)

    page_indices = list(range(start, stop))
    table_map = _tables_as_markdown(path, page_indices) if tables else {}

    # Decided ONCE per file, not per page. A two-up scan (two logical pages
    # side by side on one physical page) is a property of the scan, and
    # per-page detection fails on exactly the pages that need it: a full-bleed
    # graphic leaves too few blocks to find the fold. ILIA_2023.pdf is 160
    # such pages; per-page detection caught 60 of them, the document-level
    # vote catches all of them, and interleaved pages went 38 -> 0.
    two_up = document_is_two_up(doc)
    if two_up:
        print(f"      {path.name}: two-up spread, splitting pages at the fold")

    pages: list[str] = []
    for i in page_indices:
        page = doc[i]
        raw = page.get_text("text")

        if mode == "ocr":
            body = ocr_full_page(page, ocr_lang)
        elif mode == "plain":
            body = raw
        elif mode == "layout":
            body = page_text_in_reading_order(page, two_up=two_up)
        elif mode == "auto":
            # legacy path, kept for explicit opt-in only
            if len(raw.strip()) < 100:
                body = ocr_full_page(page, ocr_lang) or raw
            elif detect_columns(page) > 1:
                body = page_text_in_reading_order(page, two_up=two_up)
            else:
                body = raw
        else:  # "digital" -- the default
            if len(raw.strip()) < 100:
                # No usable text layer on THIS page. Do not rasterize: a
                # handful of image pages inside an otherwise digital report
                # is not worth an OCR pass, and files that are genuinely
                # scanned went to Phase B instead.
                body = raw
            elif two_up or detect_columns(page) > 1:
                # `two_up or` matters: on a spread, plain get_text() reads
                # straight across the fold and splices the two logical pages
                # line by line. That must be reordered even when each half is
                # single-column and detect_columns therefore returns 1.
                body = page_text_in_reading_order(page, two_up=two_up)
            else:
                # Single column: plain extraction already matches the
                # reference pipeline, so do not second-guess it.
                body = raw

        parts = [body]

        for md in table_map.get(i, []):
            parts.append(f"\n[TABLE]\n{md}\n")

        if figure_ocr:
            for xref, *_ in page.get_images(full=True):
                try:
                    embedded = doc.extract_image(xref)
                except Exception:
                    continue
                text = _ocr(embedded["image"], ocr_lang)
                if text:
                    parts.append(f"\n[FIGURE] {text}\n")

        pages.append("\n".join(parts))

    doc.close()
    if drop_boilerplate:
        pages = strip_boilerplate(pages)
    return clean("\n\n".join(pages))


# ---------------------------------------------------------------- JSON

TEXT_FIELDS = ("title", "titulo", "headline", "summary", "abstract", "resumen",
               "body_text", "body", "content", "contenido", "text", "texto",
               "body_paragraphs", "paragraphs", "parrafos")
META_FIELDS = ("url", "link", "date", "fecha", "published", "authors", "autor",
               "autores", "tags", "keywords", "source", "fuente", "lang", "idioma")


def _flatten(value: Any) -> str:
    """Join lists of paragraphs while preserving their order."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n\n".join(_flatten(v) for v in value if v)
    if isinstance(value, dict):
        return "\n\n".join(_flatten(v) for v in value.values() if v)
    return ""


# 363 of the 954 JSON files are Alertas Tempranas, and they are the densest
# Fenomeno 3 subcorpus -- the phenomenon covering q033-q050, 18 of 50 queries.
# Their `title` is always the literal string "Mapa" and `fields` is always {},
# so the generic extractor produces a document whose most prominent line is
# noise. The retrievable signal is in alerta_meta: municipality names, alert
# type, date, key theme. A query like "corredores de movilidad" or
# "reclutamiento de menores" matches on those, not on "Mapa".
def _alertas_tempranas(data: dict) -> tuple[str, dict] | None:
    meta_block = data.get("alerta_meta")
    if not isinstance(meta_block, dict):
        return None

    header = []
    for key, label in (("codigo", "Alerta"), ("tipo", "Tipo"),
                       ("fecha_emision", "Fecha"), ("municipios", "Municipios"),
                       ("tema_clave", "Tema")):
        value = str(meta_block.get(key, "")).strip()
        if value:
            header.append(f"{label}: {value}")

    body = _flatten(data.get("body_paragraphs") or []).strip()
    text = "\n\n".join(p for p in [". ".join(header), body] if p)
    return text, {k: v for k, v in meta_block.items() if v}


# CENIA (15 files) uses sections[{heading, paragraphs[]}] and has no body_text.
def _sectioned(data: dict) -> tuple[str, dict] | None:
    sections = data.get("sections")
    if not isinstance(sections, list) or not sections:
        return None
    parts = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        heading = str(section.get("heading", "")).strip()
        body = _flatten(section.get("paragraphs") or []).strip()
        if heading and body:
            parts.append(f"{heading}. {body}")
        elif body:
            parts.append(body)
    return ("\n\n".join(parts), {}) if parts else None


def extract_json(path: Path) -> tuple[str, dict]:
    """
    Section 2.1: explicitly select the text fields and concatenate them in
    order; descriptive fields go to metadata, NOT into the body text.
    Returns (text, metadata).

    Three corpus-specific schemas are handled before the generic path,
    because the generic path produces near-useless text for all three.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):          # array of articles -> single document
        data = {"items": data}

    for handler in (_alertas_tempranas, _sectioned):
        result = handler(data)
        if result and len(result[0].split()) >= 20:
            return clean(result[0]), result[1]

    parts, meta = [], {}
    for field_name in TEXT_FIELDS:                  # canonical order
        if data.get(field_name):
            value = _flatten(data[field_name]).strip()
            if value:
                parts.append(value)
    for field_name in META_FIELDS:
        if data.get(field_name):
            meta[field_name] = data[field_name]

    if not parts:                                   # unknown schema
        parts.append(_flatten(data))

    return clean("\n\n".join(parts)), meta


# ---------------------------------------------------------------- HTML

def extract_html(path: Path) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    # "\n" as separator keeps block structure as a signal for chunking
    return clean(soup.get_text("\n"))


# ---------------------------------------------------------------- tabular

# Tile plumbing: the same feature is repeated once per zoom level, exactly as
# spec 2.1 warns. Keeping every copy fills the index with identical rows that
# crowd each other out of the top 10.
_TILE_COLUMNS = {"tile_zoom", "tile_x", "tile_y", "zoom", "z", "x", "y"}
_DEDUP_KEYS = ("au_ID_concatenated", "au_id_concatenated", "id_concatenated")


def extract_tabular(path: Path, max_rows: int | None = None) -> str:
    """
    CSV/XLSX: one row per line as 'column: value | column: value' (2.1).

    Two corpus-specific behaviours:

      Deduplication. AMAZONUW_amazonunderworld-data.csv is the decoded tile
      pyramid: 4369 rows where the same municipality recurs at several zoom
      levels. Rows are deduplicated on au_ID_concatenated, keeping the first,
      and the tile coordinate columns are dropped from the emitted text since
      "tile_x: 143" is not something any query will ever match.

      Row cap. The AI Index PubMed exports run to 111,775 rows of
      bibliographic listings (PMID, title, journal). None of the 50 queries
      ask about biomedical literature, so they are excluded outright in
      build_index.py; max_rows exists as a second line of defence if one
      slips through.
    """
    import pandas as pd

    if path.suffix.lower() in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None, dtype=str)
    else:
        sheets = {"": pd.read_csv(path, dtype=str, on_bad_lines="skip")}

    lines = []
    for name, df in sheets.items():
        if name:
            lines.append(f"[SHEET] {name}")

        key = next((k for k in _DEDUP_KEYS if k in df.columns), None)
        if key:
            before = len(df)
            df = df.drop_duplicates(subset=[key], keep="first")
            if len(df) < before:
                lines.append(f"[NOTE] {before - len(df)} duplicate rows "
                             f"collapsed on {key}")

        columns = [c for c in df.columns if str(c).lower() not in _TILE_COLUMNS]
        df = df[columns]
        if max_rows:
            df = df.head(max_rows)

        for _, row in df.fillna("").iterrows():
            pairs = [f"{col}: {val}" for col, val in row.items() if str(val).strip()]
            if pairs:
                lines.append(" | ".join(pairs) + ".")
    return clean("\n".join(lines))


def extract_image(path: Path) -> str:
    return _ocr(path.read_bytes())


# ---------------------------------------------------------------- dispatcher

FORMAT_BY_SUFFIX = {
    ".pdf": "pdf", ".html": "html", ".htm": "html", ".json": "json",
    ".csv": "csv", ".xlsx": "xlsx", ".xls": "xlsx",
    ".md": "md", ".txt": "txt",
    ".png": "img", ".jpg": "img", ".jpeg": "img", ".webp": "img",
    ".avif": "img",
    ".pbf": "pbf",                       # was missing: 73 files silently lost
}

# A .pbf tile encodes attribute pairs, not prose. 20 words is the right floor
# for an article and the wrong one for a tile holding six municipalities.
MIN_WORDS_BY_FORMAT = {"pbf": 8}
DEFAULT_MIN_WORDS = 20


def load_document(path: Path, doc_id: str, phenomenon: int = 0,
                  mode: str = "digital", ocr_lang: str = "eng+spa",
                  first_page: int = 0, last_page: int | None = None,
                  official_doc_id: str = "") -> Document | None:
    """
    `mode` is supplied per file from triage.json, not guessed here:

        digital -> "digital"  text layer is complete; per-page column
                              detection, XY-cut where needed, no OCR ever.
        sparse  -> Phase B    partial text layer (slides, posters). These now
                              go to GPU OCR rather than the CPU pool, because
                              "some pages have text" used to mean "rasterize
                              the rest with Tesseract, one page at a time".
        scanned -> Phase B    no text layer at all.

    first_page/last_page restrict extraction to a page range so build_index
    can split a very large PDF across several workers.
    """
    file_format = FORMAT_BY_SUFFIX.get(path.suffix.lower())
    if file_format is None:
        return None

    meta: dict = {}
    if file_format == "pdf":
        text = extract_pdf(path, mode=mode, ocr_lang=ocr_lang,
                           first_page=first_page, last_page=last_page)
    elif file_format == "html":
        text = extract_html(path)
    elif file_format == "json":
        text, meta = extract_json(path)
    elif file_format in ("csv", "xlsx"):
        text = extract_tabular(path)
    elif file_format == "img":
        text = extract_image(path)
    elif file_format == "pbf":
        from pbf_extract import extract_pbf
        text = extract_pbf(path)
    else:
        text = clean(path.read_text(encoding="utf-8", errors="ignore"))

    minimum = MIN_WORDS_BY_FORMAT.get(file_format, DEFAULT_MIN_WORDS)
    if len(text.split()) < minimum:      # empty or unreadable document
        return None

    return Document(doc_id=doc_id, source=path.name, file_format=file_format,
                    text=text, phenomenon=phenomenon, file_name=path.name,
                    official_doc_id=official_doc_id or doc_id, extra=meta)


# triage class -> extraction mode.
# NOTE: "sparse" and "scanned" are intercepted by build_index.py before they
# reach here and routed to Phase B. The entries below are the safe fallback
# for when OCR is disabled entirely (--ocr-backend none).
MODE_BY_CLASS = {
    "digital": "digital",
    "scanned": "digital",
    "sparse": "digital",
    "empty": "digital",
    "error": "digital",
}

# Any PDF absent from triage.json lands here. It used to default to "auto",
# which meant an unclassified file could still trigger per-page Tesseract.
DEFAULT_PDF_MODE = "digital"


def walk_corpus(root: Path, phenomenon_by_folder: dict[str, int] | None = None,
                inventory: dict[str, dict] | None = None) -> Iterable[Document]:
    """
    Walk the corpus and yield Documents. (Single-process convenience path;
    build_index.py does not use this.)

    `source` is the RELATIVE PATH, not the file name: 47 basenames repeat
    across 114 files in this corpus, so the bare name silently merges
    unrelated documents.
    """
    files = sorted(p for p in root.rglob("*") if p.is_file())
    for i, path in enumerate(files):
        relative = str(path.relative_to(root))
        record = (inventory or {}).get(relative, {})

        phenomenon = record.get("phenomenon", 0)
        if not phenomenon and phenomenon_by_folder:
            for key, value in phenomenon_by_folder.items():
                if key in relative:
                    phenomenon = value
                    break

        doc_id = record.get("doc_id") or f"DOC-{i:05d}"
        doc = load_document(path, doc_id=doc_id, phenomenon=phenomenon)
        if doc:
            doc.source = relative                  # full relative path
            doc.extra["nombre_archivo"] = path.name
            doc.extra["doc_id_oficial"] = record.get("doc_id", "")
            yield doc