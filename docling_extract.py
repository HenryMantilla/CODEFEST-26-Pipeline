"""
docling_extract.py — Phase C: Docling layout extraction, GPU, decoder-free.

------------------------------------------------------------------------
WHAT DOCLING REPLACES, AND WHAT IT DOES NOT
------------------------------------------------------------------------
    Docling does NOT replace RapidOCR. Those two never touch the same file.
    build_index.py routes every PDF with a usable text layer to Phase A
    (PyMuPDF + XY-cut, or rich_layout) and sends only triage-flagged
    scanned/sparse files to Phase B (OCR). A digital PDF never reaches an OCR
    engine -- design note 1 exists precisely to stop that happening.

    On a digital PDF, Docling replaces the GEOMETRY. Instead of inferring
    reading order from whitespace channels, a trained detector labels each
    region (title, section_header, text, list_item, caption, footnote,
    page_header, table, picture) and orders them. That is what geometry
    cannot do on a four-column spread with pull-quotes and rotated running
    heads.

------------------------------------------------------------------------
DECODER-FREE (spec 4.2)
------------------------------------------------------------------------
    Docling ships several models and only some are usable here:

      LAYOUT (DocLayNet-trained detector)   SAFE. DETR-family object
          detection: a fixed set of object queries decoded in ONE parallel
          pass. There is no p(w_t | w_<t) anywhere in it. "Has a decoder" and
          "is autoregressive" are different claims, and 4.2 is about the
          second. Do not disable this -- it is the entire reason to run
          Docling.

      TABLEFORMER                           NOT SAFE. Im2Seq: a transformer
          decoder emits OTSL structure tokens one at a time, trained on
          next-token cross-entropy.

      CODE / FORMULA ENRICHMENT             NOT SAFE. CodeFormula is a
          vision-language model that generates its output text.

      PICTURE DESCRIPTION                   NOT SAFE. Captioning VLM.

      VLM PIPELINE (SmolDocling)            NOT SAFE. SigLIP encoder plus a
          SmolLM-2 decoder emitting DocTags autoregressively. Not reachable
          from this module; do not add it.

    _assert_decoder_free() enforces this at run time instead of trusting the
    constructor. These flags default differently across Docling releases, and
    a future version could add a generative enrichment that defaults to on. A
    silent upgrade is exactly how a compliance decision gets reversed without
    anyone deciding anything.

Requires: pip install docling      (written against docling 2.x)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Flags that switch on a component which GENERATES its output. Every one of
# these must be False. Add to this list; never remove from it.
GENERATIVE_FLAGS = (
    "do_table_structure",         # TableFormer: autoregressive OTSL decoder
    "do_code_enrichment",         # CodeFormula VLM
    "do_formula_enrichment",      # CodeFormula VLM
    "do_picture_description",     # captioning VLM
)

# Reviewed and allowed. Listed so the audit can tell "known safe" from "new
# flag nobody has looked at".
REVIEWED_SAFE_FLAGS = (
    "do_ocr",                     # EasyOCR / Tesseract / RapidOCR: all CTC
    "do_picture_classification",  # classifier head, emits a label not text
    "do_cell_matching",
)


@dataclass
class TextUnit:
    """
    Mirrors rich_layout.TextUnit so chunking.chunk_text_units() consumes
    either source without knowing which produced it.
    """
    text: str
    kind: str                                  # heading | body | callout | caption
    page: int
    heading_path: list[str] = field(default_factory=list)

    def context_prefix(self) -> str:
        return " > ".join(self.heading_path)


def _assert_decoder_free(options) -> None:
    """Fail loudly rather than quietly extract with a generative model."""
    enabled = [f for f in GENERATIVE_FLAGS if bool(getattr(options, f, False))]
    if enabled:
        raise SystemExit(
            f"docling_extract: generative components are enabled: {enabled}. "
            f"Spec 4.2 prohibits decoder architectures during index "
            f"construction. Refusing to run.")

    known = set(GENERATIVE_FLAGS) | set(REVIEWED_SAFE_FLAGS)
    fields = getattr(options, "model_fields", None) or vars(options)
    unknown = [f for f in fields
               if f.startswith("do_") and f not in known
               and bool(getattr(options, f, False))]
    if unknown:
        print(f"      WARNING: docling exposes do_* flags this module has not "
              f"reviewed, and they are ON: {unknown}. Check whether any is "
              f"autoregressive before trusting this run.")


class DoclingEngine:
    """
    One converter per (do_ocr, ocr_lang) combination. Loading the layout model
    costs seconds and GPU memory, so build_index.py caches instances -- which
    is also why Phase C is single-process: fifteen workers would open fifteen
    CUDA contexts on one device.
    """

    def __init__(self, device: str = "cuda", do_ocr: bool = False,
                 ocr_lang: list[str] | None = None, num_threads: int = 8):
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions)
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions()

        # Set defensively: a flag absent in this Docling version is simply not
        # set, and one added later is caught by the unknown-flag audit rather
        # than slipping through.
        for flag in GENERATIVE_FLAGS:
            if hasattr(options, flag):
                setattr(options, flag, False)

        # OCR only for the scanned subset. Digital PDFs must never rasterize.
        options.do_ocr = bool(do_ocr)
        if do_ocr:
            # Checked ONCE, at construction, not per file. Without this the
            # ImportError surfaces inside convert() and every scanned file
            # reports its own identical failure -- 40 lines of log for one
            # missing package, which is how the nccl problem hid for a whole
            # run.
            try:
                import easyocr  # noqa: F401
            except ImportError:
                raise SystemExit(
                    "docling_extract: do_ocr=True but easyocr is not "
                    "installed.\n"
                    "  Prefer NOT installing it. easyocr is a second OCR "
                    "stack with its own\n  torch pins, and this project "
                    "already has a configured one: RapidOCR with\n  the "
                    "PP-OCRv5 Latin head, which reads Spanish accents "
                    "correctly.\n"
                    "  Scanned files belong on Phase B. build_index.py keeps "
                    "them there unless\n  --docling scanned or --docling all "
                    "is given; use --docling list or\n  --docling complex "
                    "instead.")
            try:
                from docling.datamodel.pipeline_options import EasyOcrOptions
                options.ocr_options = EasyOcrOptions(
                    lang=ocr_lang or ["en", "es"], use_gpu=(device == "cuda"))
            except Exception as exc:
                print(f"      WARNING: EasyOCR options unavailable ({exc}); "
                      f"using Docling's default OCR engine.")

        # No page or picture rasters: they cost time and VRAM and nothing
        # downstream reads them.
        for flag in ("generate_page_images", "generate_picture_images"):
            if hasattr(options, flag):
                setattr(options, flag, False)

        options.accelerator_options = AcceleratorOptions(
            num_threads=num_threads,
            device=(AcceleratorDevice.CUDA if device == "cuda"
                    else AcceleratorDevice.CPU))

        _assert_decoder_free(options)

        self.device = device
        self.do_ocr = bool(do_ocr)
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(
                pipeline_options=options)})


# Docling label -> our unit kind. Anything unlisted is dropped.
#
# page_header / page_footer are dropped even though CORPUS_ANALISIS notes the
# organisers' own fragments KEEP running heads: their pipeline could not
# identify them, ours can, and a running head repeated across 200 pages is 200
# near-identical chunks competing for ten slots. If a rebuild with Docling
# scores worse on nDCG@10, move these two to _KIND as "body" and re-measure --
# that is the one alignment this trades away deliberately.
_KIND = {
    "title": "heading",
    "section_header": "heading",
    "text": "body",
    "paragraph": "body",
    "list_item": "body",
    "caption": "caption",
    "footnote": "caption",
    "code": "body",
    "formula": "body",
    "reference": "body",
}
_DROP = {"page_header", "page_footer", "picture", "chart", "table",
         "document_index", "checkbox_selected", "checkbox_unselected",
         "form", "key_value_region", "empty_value", "marker"}


def _page_of(item) -> int:
    try:
        return int(item.prov[0].page_no)
    except Exception:
        return 0


def document_to_units(document) -> list[TextUnit]:
    """
    Walk the DoclingDocument in reading order, carrying a heading trail.

    iterate_items() yields (item, level), where level is nesting depth. That
    is what lets a section_header pop its siblings off the trail instead of
    accumulating every heading in the document into one 400-character
    `contexto` that dilutes the embedding it was meant to sharpen.
    """
    units: list[TextUnit] = []
    trail: list[tuple[int, str]] = []          # (level, heading text)

    for item, level in document.iterate_items():
        label = getattr(item, "label", "")
        label = str(getattr(label, "value", label) or "")
        text = (getattr(item, "text", "") or "").strip()
        if not text or label in _DROP:
            continue

        kind = _KIND.get(label)
        if kind is None:
            continue

        if kind == "heading":
            trail = [(l, t) for l, t in trail if l < level]
            trail.append((level, text))
            continue                            # headings live in the context

        units.append(TextUnit(text=text, kind=kind, page=_page_of(item),
                              heading_path=[t for _l, t in trail]))
    return units


def _is_useful(unit: TextUnit, min_words: int = 4) -> bool:
    text = unit.text.strip()
    if not text:
        return False
    if unit.kind in ("callout", "caption"):
        return True
    return len(text.split()) >= min_words


def docling_one_file(path: Path, root: Path, doc_id: str, phenomenon: int,
                     engine: DoclingEngine, count_tokens, max_chars: int,
                     overlap_chars: int) -> tuple[list[dict], str]:
    """
    One PDF -> chunk records. Returns ([], note) on failure so Phase C logs
    the file and continues instead of losing a multi-hour run to one bad PDF.

    Two-up spreads are normalised to single logical pages BEFORE conversion.
    Docling's layout model is trained on single-page documents and has no
    reason to treat the fold as a hard boundary, so handing it a spread swaps
    one reading-order guess for another. Geometry first, model second.
    """
    import os

    from chunking import chunk_text_units
    from extraction import guess_title
    from layout import split_spreads

    target, page_map, note_prefix = path, None, ""
    try:
        split = split_spreads(path)
        if split:
            target, page_map = Path(split[0]), split[1]
            note_prefix = f"two-up split {len(page_map)}p; "
    except Exception as exc:
        # Not fatal: convert the original and accept the fold.
        note_prefix = f"spread split failed ({type(exc).__name__}); "

    try:
        result = engine.converter.convert(str(target))
    except Exception as exc:
        return [], (f"{note_prefix}docling failed: "
                    f"{type(exc).__name__}: {str(exc)[:100]}")
    finally:
        if page_map is not None and target != path:
            try:
                os.unlink(target)
            except OSError:
                pass

    document = getattr(result, "document", None)
    if document is None:
        return [], f"{note_prefix}docling returned no document"

    units = [u for u in document_to_units(document) if _is_useful(u)]
    if page_map is not None:
        # Report the page the reader would see, not the index into the
        # temporary split file.
        for unit in units:
            if 1 <= unit.page <= len(page_map):
                unit.page = page_map[unit.page - 1]
    if not units:
        return [], f"{note_prefix}no usable units"

    # FIX: filename-derived title (units already carry their own heading
    # trail via unit.context_prefix(); this only adds the document-level
    # anchor, same reasoning as the rich_layout path in build_index.py).
    doc_title = guess_title("", path.name)
    chunks = chunk_text_units(
        units, doc_id=doc_id, source=str(path.relative_to(root)),
        phenomenon=phenomenon, count_tokens=count_tokens,
        file_name=path.name, official_doc_id=doc_id,
        max_chars=max_chars, overlap_chars=overlap_chars,
        doc_title=doc_title)

    return [c.to_dict() for c in chunks], ""