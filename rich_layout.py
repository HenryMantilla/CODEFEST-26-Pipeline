"""
rich_layout.py — Span-level extraction for design-heavy PDFs.

Magazine-style reports (annual reports, HAI-style spreads) break naive text
extraction in four specific ways. None of them need a vision model:

  1. STAT CALLOUTS. "25" in 40pt and "seed grants" in 9pt are separate spans.
     Extracted naively you get an orphan chunk "25" plus a dangling label.
     Both are retrieval noise. Fixed by pairing the number with its label.

  2. RUNNING HEADS. Vertical text along the page edge ("01 | RESEARCH FOCUS |
     HAI ANNUAL REPORT 2025") gets injected into the middle of body text.
     Fixed by dropping spans whose writing direction is not horizontal.

  3. LOST HEADING CONTEXT. A paragraph about "cooperative and competitive
     games" never contains the words "HAI seed grant" or "cloud credit". The
     heading is a separate block that chunking throws away. Fixed by carrying
     a heading path with every unit.

  4. FLOATING CAPTIONS. Caption text sits in its own tiny block far from any
     paragraph and becomes a sub-minimum chunk that gets merged into whatever
     happens to precede it. Fixed by tagging captions explicitly.

What genuinely cannot be recovered without a VQA model: the semantic content
of photographs, illustrations and decorative vector art. That loss is small,
because in documents like these the retrievable information lives in the
caption, not in the pixels.

Requires: pip install pymupdf
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field

import fitz

from layout import _xy_cut


# ------------------------------------------------------------- data model

@dataclass
class Span:
    text: str
    bbox: tuple[float, float, float, float]
    size: float
    font: str
    bold: bool
    horizontal: bool


@dataclass
class TextUnit:
    """A block of text plus the heading trail that gives it context."""
    text: str
    kind: str                                   # heading | body | callout | caption
    bbox: tuple[float, float, float, float]
    page: int
    heading_path: list[str] = field(default_factory=list)

    def context_prefix(self) -> str:
        return " > ".join(self.heading_path)


# ------------------------------------------------------------- span reading

def read_spans(page) -> list[Span]:
    """All text spans on the page, with font metrics and writing direction."""
    spans: list[Span] = []
    data = page.get_text("dict")

    for block in data.get("blocks", []):
        if block.get("type") != 0:              # 1 = image
            continue
        for line in block.get("lines", []):
            direction = line.get("dir", (1.0, 0.0))
            # dir is the unit vector of the writing direction; (1, 0) is
            # left-to-right. Rotated running heads have dir ~ (0, ±1).
            horizontal = abs(direction[0]) > 0.9
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                spans.append(Span(
                    text=text,
                    bbox=tuple(span["bbox"]),
                    size=round(span.get("size", 0), 1),
                    font=span.get("font", ""),
                    bold=bool(span.get("flags", 0) & 2 ** 4),
                    horizontal=horizontal,
                ))
    return spans


def body_font_size(spans: list[Span]) -> float:
    """
    Dominant body size, weighted by character count. Weighting matters: a
    display page has few huge glyphs and many small ones, so an unweighted
    mode can lock onto a caption size.
    """
    weights: Counter = Counter()
    for span in spans:
        if span.horizontal:
            weights[span.size] += len(span.text)
    if not weights:
        return 10.0
    return weights.most_common(1)[0][0]


# ------------------------------------------------------------- callouts

_NUMERIC = re.compile(r"^[\$€£]?\s*[\d.,]+\s*[%kKmMbB+]?$")


def _is_stat_number(text: str) -> bool:
    """'25', '$1.8', '1,234', '36', '80%' — but not '2021' inside a sentence."""
    return bool(_NUMERIC.match(text)) and any(c.isdigit() for c in text)


def _boxes_touch_horizontally(a, b, tolerance: float, slack: float) -> bool:
    """b sits immediately to the right of a, on roughly the same baseline."""
    a_x0, a_y0, a_x1, a_y1 = a
    b_x0, b_y0, b_x1, b_y1 = b
    gap = b_x0 - a_x1
    vertical_overlap = min(a_y1, b_y1) - max(a_y0, b_y0)
    return -slack <= gap <= tolerance and vertical_overlap > 0


def _box_below(a, b, tolerance: float, slack: float) -> bool:
    """
    b sits just under a, horizontally overlapping it.

    The negative bound must be generous: a small label's ascenders routinely
    intrude into the descender box of a 34pt figure, so the measured gap is
    often negative even though the label is visually below.
    """
    a_x0, a_y0, a_x1, a_y1 = a
    b_x0, b_y0, b_x1, b_y1 = b
    gap = b_y0 - a_y1
    horizontal_overlap = min(a_x1, b_x1) - max(a_x0, b_x0)
    return -slack <= gap <= tolerance and horizontal_overlap > 0 and b_y1 > a_y1


def merge_stat_callouts(spans: list[Span], body_size: float, ratio: float = 1.7
                        ) -> tuple[list[tuple[str, tuple]], set[int]]:
    """
    Pair each oversized numeric span with its adjacent label and emit a
    readable sentence: "25" + "seed grants" -> "25 seed grants."

    Returns (sentences, indices of consumed spans). Consumed spans must be
    removed from the normal text flow so the number is not emitted twice.
    """
    sentences: list[tuple[str, tuple]] = []
    consumed: set[int] = set()
    line_height = body_size * 2.2

    for i, span in enumerate(spans):
        if i in consumed or not span.horizontal:
            continue
        if span.size < body_size * ratio or not _is_stat_number(span.text):
            continue

        label_parts: list[str] = []
        anchor = span.bbox

        # Labels sit to the right of the figure or directly beneath it, and
        # are always set smaller than the figure itself.
        for j, other in enumerate(spans):
            if j == i or j in consumed or not other.horizontal:
                continue
            if other.size >= span.size:
                continue
            if (_boxes_touch_horizontally(anchor, other.bbox, body_size * 1.5, body_size)
                    or _box_below(anchor, other.bbox, line_height, body_size)):
                label_parts.append(other.text)
                consumed.add(j)
                anchor = (min(anchor[0], other.bbox[0]), min(anchor[1], other.bbox[1]),
                          max(anchor[2], other.bbox[2]), max(anchor[3], other.bbox[3]))

        if label_parts:
            label = " ".join(label_parts).strip(" .,:")
            sentence = re.sub(r"\s+", " ", f"{span.text} {label}.")
            # The bbox is kept so the callout can be re-inserted at its real
            # position on the page, and therefore inherit the correct heading.
            sentences.append((sentence, anchor))
            consumed.add(i)
        # A big number with no label nearby is decorative — drop it entirely
        # rather than let it become an orphan chunk.
        else:
            consumed.add(i)

    return sentences, consumed


# ------------------------------------------------------------- page parsing

def parse_page(page, page_number: int, heading_stack: list[tuple[float, str]] | None = None
               ) -> tuple[list[TextUnit], list[tuple[float, str]]]:
    """
    Parse one page into ordered TextUnits carrying their heading path.

    heading_stack is threaded across pages so a section heading on page 8
    still contextualizes paragraphs continuing on page 9.
    """
    spans = read_spans(page)
    if not spans:
        return [], heading_stack or []

    base = body_font_size(spans)

    # 2) Drop rotated running heads before anything else
    spans = [s for s in spans if s.horizontal]
    if not spans:
        return [], heading_stack or []

    # 1) Recombine stat callouts and remove their spans from the flow
    callout_sentences, consumed = merge_stat_callouts(spans, base)
    flow = [s for i, s in enumerate(spans) if i not in consumed]

    # Group surviving spans into visual lines. Grouping by y alone would glue
    # side-by-side columns into one line ("HAI Seed Grants Featured HAI Seed
    # Grants"), so each y-band is then split at large horizontal gaps.
    bands: dict[int, list[Span]] = {}
    for span in flow:
        key = round(span.bbox[1] / max(base * 0.6, 1))
        bands.setdefault(key, []).append(span)

    column_gap = base * 3.0
    blocks: list[tuple[float, float, float, float, str, float, bool]] = []

    for key in sorted(bands):
        row = sorted(bands[key], key=lambda s: s.bbox[0])
        run: list[Span] = []

        def close_run(run: list[Span]) -> None:
            if not run:
                return
            text = " ".join(s.text for s in run)
            blocks.append((
                min(s.bbox[0] for s in run), min(s.bbox[1] for s in run),
                max(s.bbox[2] for s in run), max(s.bbox[3] for s in run),
                text,
                statistics.median([s.size for s in run]),
                any(s.bold for s in run),
            ))

        for span in row:
            if run and span.bbox[0] - run[-1].bbox[2] > column_gap:
                close_run(run)
                run = []
            run.append(span)
        close_run(run)

    # Re-insert merged callouts as ordinary blocks at their real page
    # position. Appending them at the end instead would give a left-column
    # statistic the heading path of the last section on the page.
    callout_indices = set()
    for sentence, bbox in callout_sentences:
        callout_indices.add(len(blocks))
        blocks.append((bbox[0], bbox[1], bbox[2], bbox[3], sentence, base, False))

    # Reading order via the same XY-cut used for plain multi-column pages.
    # Blocks are keyed by index, not text: identical lines (repeated labels,
    # "Featured Grant in ...") would otherwise collide and vanish.
    ordered = _xy_cut([(b[0], b[1], b[2], b[3], f"{i}\x00{b[4]}")
                       for i, b in enumerate(blocks)],
                      min_gutter=page.rect.width * 0.035, min_gap=base * 0.9)

    units: list[TextUnit] = []
    stack: list[tuple[float, str]] = list(heading_stack or [])
    paragraph: list[str] = []
    paragraph_box = None
    pending: dict | None = None      # multi-line heading being assembled

    def flush_heading() -> None:
        """A display heading wraps over several lines; emit it as one unit."""
        nonlocal pending
        if pending is None:
            return
        text = " ".join(pending["lines"])
        size = pending["size"]
        while stack and stack[-1][0] <= size:
            stack.pop()
        stack.append((size, text))
        units.append(TextUnit(text=text, kind="heading", bbox=pending["bbox"],
                              page=page_number,
                              heading_path=[h for _s, h in stack[:-1]]))
        pending = None

    def flush() -> None:
        nonlocal paragraph, paragraph_box
        flush_heading()
        if paragraph:
            units.append(TextUnit(text=" ".join(paragraph), kind="body",
                                  bbox=paragraph_box, page=page_number,
                                  heading_path=[h for _s, h in stack]))
            paragraph, paragraph_box = [], None

    for box in ordered:
        index = int(box[4].split("\x00", 1)[0])
        _x0, _y0, _x1, _y1, text, size, bold = blocks[index]

        if index in callout_indices:
            flush()
            units.append(TextUnit(text=text, kind="callout", bbox=box[:4],
                                  page=page_number,
                                  heading_path=[h for _s, h in stack]))
            continue

        is_heading = (size >= base * 1.12 or (bold and size >= base)) and len(text) < 120
        is_caption = size <= base * 0.86 and not bold

        if is_heading:
            # 3) Maintain the heading path. Consecutive lines of the same size
            # are one wrapped heading, not a stack of nested ones.
            if paragraph:
                flush()
            if (pending is not None and abs(pending["size"] - size) < 0.6
                    and box[1] - pending["bbox"][3] < size * 1.2):
                pending["lines"].append(text)
                pending["bbox"] = (min(pending["bbox"][0], box[0]),
                                   min(pending["bbox"][1], box[1]),
                                   max(pending["bbox"][2], box[2]),
                                   max(pending["bbox"][3], box[3]))
            else:
                flush_heading()
                pending = {"lines": [text], "size": size, "bbox": tuple(box[:4])}
        elif is_caption:
            # 4) Captions stay their own unit and never get merged into body
            flush()
            units.append(TextUnit(text=text, kind="caption", bbox=box[:4],
                                  page=page_number,
                                  heading_path=[h for _s, h in stack]))
        else:
            paragraph.append(text)
            paragraph_box = box[:4] if paragraph_box is None else (
                min(paragraph_box[0], box[0]), min(paragraph_box[1], box[1]),
                max(paragraph_box[2], box[2]), max(paragraph_box[3], box[3]))
    flush()
    _reassign_callout_context(units)
    return units, stack


def _reassign_callout_context(units: list[TextUnit]) -> None:
    """
    Give each callout the heading directly above it in its own column.

    Flow order cannot be trusted for these: a statistic block sits in a
    visual gutter of its own, and whichever cut the XY-cut picks first
    decides whether it lands after its own section heading or after the
    neighbouring column's. Spatial containment is unambiguous, so it wins.
    """
    headings = [u for u in units if u.kind == "heading"]
    if not headings:
        return

    for unit in units:
        if unit.kind != "callout":
            continue
        x0, y0, _x1, _y1 = unit.bbox
        best, best_distance = None, float("inf")

        for heading in headings:
            hx0, _hy0, hx1, hy1 = heading.bbox
            if hy1 > y0:                       # must be above the callout
                continue
            if not (hx0 - 5 <= x0 <= hx1 + 200):   # must share the column
                continue
            distance = y0 - hy1
            if distance < best_distance:
                best, best_distance = heading, distance

        if best is not None:
            unit.heading_path = best.heading_path + [best.text]


def parse_document(path, max_pages: int | None = None,
                   first_page: int = 0,
                   last_page: int | None = None) -> list[TextUnit]:
    """
    Parse a PDF into contextualized TextUnits.

    first_page/last_page restrict parsing to a half-open page range so
    build_index.py can shard a large design-heavy PDF across workers. The
    heading stack is per-shard: a shard that starts mid-section loses the
    heading that opened it. That is why build_index only shards files above
    SHARD_MIN_PAGES, where sections are dense enough that the first heading
    arrives within a page or two.
    """
    doc = fitz.open(path)
    units: list[TextUnit] = []
    stack: list[tuple[float, str]] = []

    total = doc.page_count
    start = max(0, first_page)
    stop = total if last_page is None else min(total, last_page)
    if max_pages:
        stop = min(stop, start + max_pages)

    for i in range(start, stop):
        page_units, stack = parse_page(doc[i], i + 1, stack)
        units.extend(page_units)

    doc.close()
    return units


# ------------------------------------------------------------- junk filter

_JUNK = re.compile(r"^[\s\d.,:;|/\\\-–—•·()\[\]]+$")


def is_useful(unit: TextUnit, min_words: int = 4) -> bool:
    """
    Reject units that cannot support retrieval: page numbers, orphan figures,
    axis labels, decorative fragments. Callouts are exempt because a merged
    callout is short by construction but genuinely informative.
    """
    text = unit.text.strip()
    if not text or _JUNK.match(text):
        return False
    if unit.kind in ("callout", "heading"):
        return True

    words = text.split()
    if len(words) < min_words:
        return False
    digit_ratio = sum(c.isdigit() for c in text) / len(text)
    return digit_ratio <= 0.5