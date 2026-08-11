"""
layout.py — Reading order for multi-column PDFs (CODEFEST AD ASTRA 2026).

Problem: page.get_text("text", sort=True) sorts by (y, x) across the full page
width. On a two-column page that interleaves the columns line by line and
destroys the reading order.

Solution: recursive XY-cut. Look for a vertical whitespace channel (gutter)
separating columns; if one exists, emit the left region in full, then the
right. Otherwise look for a horizontal cut. Elements spanning the gutter
(titles, full-width figures) split the page into bands naturally, with no
ad-hoc rules.

Requires: pip install pymupdf
"""

from __future__ import annotations

import fitz

Block = tuple[float, float, float, float, str]   # x0, y0, x1, y1, text

# Minimum width of the channel between columns, as a fraction of the region
# width. 3.5% separates real columns without mistaking word spacing or table
# cell padding for a gutter.
#
# Exported because the OCR engines in build_index.py call _xy_cut() directly
# on detector output and must use the SAME threshold as the digital path. A
# scanned two-column page and a digital one are the same layout problem; two
# different constants would silently give them two different answers.
GUTTER_FRACTION = 0.035

# Minimum height of an empty horizontal band, in points.
MIN_BAND_GAP = 10.0

# How close to the horizontal midpoint a cut must fall to count as the fold
# of a two-up spread, as a fraction of page width.
FOLD_TOLERANCE = 0.04


# ------------------------------------------------------------- helpers

def _page_blocks(page, min_chars: int = 2) -> list[Block]:
    """Text blocks with their bounding boxes, unsorted."""
    blocks = []
    for x0, y0, x1, y1, text, _no, kind in page.get_text("blocks"):
        if kind != 0:                       # 1 = image block
            continue
        stripped = text.strip()
        if len(stripped) >= min_chars:
            blocks.append((x0, y0, x1, y1, stripped))
    return blocks


def _vertical_cut(blocks: list[Block], min_gutter: float) -> tuple[float | None, float]:
    """
    Find the widest vertical channel that no block crosses and that leaves
    blocks on both sides. Returns the x coordinate of the cut, or None.
    """
    if len(blocks) < 2:
        return None, 0.0

    # Sweep line: the union of the [x0, x1] spans leaves gaps
    spans = sorted((b[0], b[2]) for b in blocks)
    best, best_width = None, 0.0
    current_end = spans[0][1]

    for x0, x1 in spans[1:]:
        gap = x0 - current_end
        if gap > best_width:
            cut = current_end + gap / 2
            has_left = any(b[2] <= cut for b in blocks)
            has_right = any(b[0] >= cut for b in blocks)
            if has_left and has_right:
                best, best_width = cut, gap
        current_end = max(current_end, x1)

    if best is None or best_width < min_gutter:
        return None, 0.0
    return best, best_width


def _horizontal_cut(blocks: list[Block], min_gap: float) -> tuple[float | None, float]:
    """Widest empty horizontal band separating two groups of blocks."""
    if len(blocks) < 2:
        return None, 0.0

    spans = sorted((b[1], b[3]) for b in blocks)
    best, best_height = None, 0.0
    current_end = spans[0][1]

    for y0, y1 in spans[1:]:
        gap = y0 - current_end
        if gap > best_height:
            best, best_height = current_end + gap / 2, gap
        current_end = max(current_end, y1)

    if best is None or best_height < min_gap:
        return None, 0.0
    return best, best_height


def _xy_cut(blocks: list[Block], min_gutter: float, min_gap: float,
            depth: int = 0, max_depth: int = 12) -> list[Block]:
    """
    Recursive ordering: first try to separate columns (vertical cut), then
    bands (horizontal cut). Once no cut applies, sort by (y, x).
    """
    if len(blocks) <= 1 or depth >= max_depth:
        return sorted(blocks, key=lambda b: (round(b[1], 1), b[0]))

    x, x_gap = _vertical_cut(blocks, min_gutter)
    y, y_gap = _horizontal_cut(blocks, min_gap)

    # Pick whichever cut is more decisive, comparing each gap against its own
    # threshold. Trying the vertical cut first would misplace short centered
    # headings: they do not physically cross the gutter, so they get filed
    # inside one column instead of separating the bands above and below.
    #
    # COLUMN_BIAS makes a band split has to be clearly stronger than the
    # gutter to win. Without it, an ordinary paragraph gap in a two-column
    # page beats the gutter and the columns get read across instead of down.
    COLUMN_BIAS = 1.5
    prefer_horizontal = y is not None and (
        x is None or (y_gap / min_gap) > COLUMN_BIAS * (x_gap / min_gutter))

    order = [("h", y), ("v", x)] if prefer_horizontal else [("v", x), ("h", y)]

    for kind, cut in order:
        if cut is None:
            continue
        if kind == "v":     # columns -> entire left region, then the right
            first = [b for b in blocks if b[2] <= cut]
            second = [b for b in blocks if b[0] >= cut]
        else:               # bands -> top, then bottom
            first = [b for b in blocks if b[3] <= cut]
            second = [b for b in blocks if b[1] >= cut]

        if first and second and len(first) + len(second) == len(blocks):
            return (_xy_cut(first, min_gutter, min_gap, depth + 1, max_depth) +
                    _xy_cut(second, min_gutter, min_gap, depth + 1, max_depth))

    return sorted(blocks, key=lambda b: (round(b[1], 1), b[0]))


def _count_columns(blocks: list[Block], min_gutter: float, min_gap: float,
                   depth: int = 0, max_depth: int = 12) -> int:
    """Maximum number of columns found in any band of the region."""
    if len(blocks) <= 1 or depth >= max_depth:
        return 1

    x, x_gap = _vertical_cut(blocks, min_gutter)
    y, y_gap = _horizontal_cut(blocks, min_gap)

    prefer_horizontal = y is not None and (
        x is None or (y_gap / min_gap) > (x_gap / min_gutter))

    # Bands are measured independently and the widest wins; full-width blocks
    # (title, figure) would otherwise hide the columns underneath them.
    if prefer_horizontal:
        top = [b for b in blocks if b[3] <= y]
        bottom = [b for b in blocks if b[1] >= y]
        if top and bottom and len(top) + len(bottom) == len(blocks):
            return max(
                _count_columns(top, min_gutter, min_gap, depth + 1, max_depth),
                _count_columns(bottom, min_gutter, min_gap, depth + 1, max_depth))

    if x is not None:
        left = [b for b in blocks if b[2] <= x]
        right = [b for b in blocks if b[0] >= x]
        if left and right and len(left) + len(right) == len(blocks):
            return (_count_columns(left, min_gutter, min_gap, depth + 1, max_depth) +
                    _count_columns(right, min_gutter, min_gap, depth + 1, max_depth))

    if not prefer_horizontal and y is not None:
        top = [b for b in blocks if b[3] <= y]
        bottom = [b for b in blocks if b[1] >= y]
        if top and bottom and len(top) + len(bottom) == len(blocks):
            return max(
                _count_columns(top, min_gutter, min_gap, depth + 1, max_depth),
                _count_columns(bottom, min_gutter, min_gap, depth + 1, max_depth))
    return 1


# ------------------------------------------------------------- furniture

# A block wider than this fraction of the region, carrying fewer than this
# many characters, is page furniture rather than content.
FURNITURE_WIDTH = 0.60
FURNITURE_CHARS = 50


def _split_furniture(blocks: list[Block], width: float
                     ) -> tuple[list[Block], list[Block]]:
    """
    Separate running heads, footers and merged page numbers from content.

    THIS IS WHAT BREAKS TWO-UP SPREADS. On a 1247x794 page holding two facing
    624-point pages, PyMuPDF merges the two page numbers into ONE block --
    '88\n89' at x[55, 1192] -- because they share a baseline. _vertical_cut
    sweeps the union of the [x0, x1] spans, that single block covers the whole
    width, current_end jumps straight to 1192, and no gap survives. The 193
    point gutter between the two logical pages becomes invisible and the page
    falls back to a horizontal cut, interleaving four columns across two
    pages.

    Measured on ILIA_2023.pdf (160 spread pages): removing one seven-character
    block turns "no vertical cut" into a clean cut at x=624.

    Furniture is not discarded -- it is returned separately and emitted last,
    so nothing is lost, it just stops dictating the geometry. Callers that
    want it gone can drop the second list.
    """
    if len(blocks) < 2:
        return blocks, []

    content, furniture = [], []
    for block in blocks:
        wide = (block[2] - block[0]) > FURNITURE_WIDTH * width
        thin = len(block[4]) < FURNITURE_CHARS
        (furniture if (wide and thin) else content).append(block)

    # If EVERYTHING looks like furniture the heuristic has misfired (a page of
    # short full-width lines). Keep the page intact rather than empty it.
    return (content, furniture) if content else (blocks, [])


def document_is_two_up(doc, sample: int = 24, share: float = 0.30) -> bool:
    """
    Decide ONCE per file whether pages hold two logical pages side by side.

    Per-page detection is unreliable on exactly the pages that need it most: a
    full-page graphic leaves one block, a figure straddling the fold hides the
    gutter, and both make is_two_up() return False on a page that plainly is a
    spread. Measured on ILIA_2023.pdf, per-page detection fired on 60 of 162
    pages while the file is two-up throughout.

    Being a spread is a property of the SCAN, not of the page. Sample, take a
    vote, and apply the answer to every page. The threshold is low (30%)
    because a document with even a quarter of its pages clearly folded is a
    spread whose remaining pages are full-bleed images and section dividers.
    """
    pages = list(doc)
    if not pages:
        return False
    step = max(1, len(pages) // sample)
    sampled = pages[::step][:sample]
    hits = sum(is_two_up(page) for page in sampled)
    return hits >= max(2, int(len(sampled) * share))


def is_two_up(page, tolerance: float = FOLD_TOLERANCE) -> bool:
    """
    True when one physical page holds two logical pages side by side.

    Landscape geometry alone is not enough -- a slide deck is landscape too.
    The test is a clear whitespace channel near the horizontal midpoint once
    furniture is out of the way, which a slide does not have.
    """
    if page.rect.width <= page.rect.height * 1.25:
        return False
    blocks, _furniture = _split_furniture(_page_blocks(page), page.rect.width)
    if len(blocks) < 2:
        return False
    cut, gap = _vertical_cut(blocks, page.rect.width * GUTTER_FRACTION)
    if cut is None:
        return False
    return abs(cut - page.rect.width / 2) <= page.rect.width * tolerance


# ------------------------------------------------------------- public API



def _column_excess(ordered: list[Block], width: float,
                   min_gap_frac: float = 0.04) -> int:
    """
    How often an ordering changes column beyond the minimum a correct reading
    requires. Cluster x-centres into columns, walk the sequence, count the
    changes; a correct order changes column exactly (ncols - 1) times. 0 is
    perfect. Pure geometry, no text, no ground truth.
    """
    if len(ordered) < 3:
        return 0
    centres = sorted((b[0] + b[2]) / 2 for b in ordered)
    gap = width * min_gap_frac
    bounds = [(a + c) / 2 for a, c in zip(centres, centres[1:]) if c - a > gap]
    if not bounds:
        return 0
    sequence = [sum(1 for x in bounds if (b[0] + b[2]) / 2 > x) for b in ordered]
    changes = sum(1 for a, c in zip(sequence, sequence[1:]) if a != c)
    return max(0, changes - len(bounds))


def blocks_are_two_up(blocks: list[Block], width: float, height: float,
                      tolerance: float = FOLD_TOLERANCE) -> bool:
    """
    Two-up detection from a raw block list, in whatever coordinate system the
    blocks use.

    Needed because OCR output has no page object: blocks come back in PIXEL
    coordinates from a rasterized image, and image_text() has no PDF page at
    all. Same test as is_two_up(), no fitz dependency.
    """
    if width <= height * 1.25:
        return False
    content, _furniture = _split_furniture(blocks, width)
    if len(content) < 2:
        return False
    cut, _gap = _vertical_cut(content, width * GUTTER_FRACTION)
    return cut is not None and abs(cut - width / 2) <= width * tolerance


def order_blocks(blocks: list[Block], width: float, height: float | None = None,
                 two_up: bool | None = None, min_gap: float = MIN_BAND_GAP,
                 trust_input_order: bool = False) -> list[Block]:
    """
    Reading order for a raw block list. THE ONE ORDERING PATH.

    Both the digital pipeline and the OCR engines end here, and that is the
    point. The OCR engines used to call _xy_cut() directly, which meant they
    got column detection but NOT furniture stripping and NOT the two-up split.
    A scanned two-column report therefore still had its columns spliced line
    by line -- the exact failure the digital path had been fixed for, silently
    surviving in the 65 files that go through OCR.

    That splice is the worst kind of corruption available here: it produces
    fluent Spanish that nobody wrote, embeds to a plausible vector, sits in the
    index looking healthy, and can never match a ground-truth fragment. It
    costs recall on the affected documents and reports as nothing at all.

    A scanned two-column page and a digital one are the same layout problem.
    They now get the same answer.

    trust_input_order: THE CUT IS CHECKED AGAINST DOING NOTHING.
        Measured across ten multi-column reports, XY-cut scored 10.11 excess
        column alternations per page against 4.70 for PyMuPDF's untouched
        content-stream order. It was more than twice as bad as not reordering
        at all -- because these documents separate columns by 8-12 points and
        GUTTER_FRACTION demands 3.5% of the page width, so no vertical cut
        fires and the recursion falls through to horizontal bands that read
        straight across.

        A reordering step that can be worse than its input must be able to
        decline. When the caller says the input order carries information --
        true for a PDF content stream, FALSE for OCR detector output, which
        is unordered by construction -- both orderings are scored and the
        better one wins. It costs one extra pass over the block list and
        removes the failure mode entirely rather than trading it for a
        threshold that has to be right on every document.
    """
    if len(blocks) <= 1:
        return list(blocks)

    content, furniture = _split_furniture(blocks, width)
    min_gutter = width * GUTTER_FRACTION

    if two_up is None and height is not None:
        two_up = blocks_are_two_up(blocks, width, height)

    if two_up:
        mid = width / 2
        half_gutter = (width / 2) * GUTTER_FRACTION
        left = [b for b in content if (b[0] + b[2]) / 2 < mid]
        right = [b for b in content if (b[0] + b[2]) / 2 >= mid]
        ordered = (_xy_cut(left, half_gutter, min_gap) +
                   _xy_cut(right, half_gutter, min_gap))
    else:
        ordered = _xy_cut(content, min_gutter, min_gap)

    # Applies to the two-up branch as well, not just the plain one: splitting
    # at the fold fixes the fold and does nothing for columns that are too
    # narrow to detect, so a spread can still come out worse than its input.
    if trust_input_order and _column_excess(content, width) < _column_excess(ordered, width):
        ordered = content            # the cut made it worse; keep the original

    return ordered + sorted(furniture, key=lambda b: (round(b[1], 1), b[0]))


def page_text_in_reading_order(page, min_gutter: float | None = None,
                               min_gap: float = MIN_BAND_GAP,
                               two_up: bool | None = None) -> str:
    """
    Page text in true reading order, with or without columns.

    min_gutter: minimum width (in points) of the channel between columns.
    Defaults to 3.5% of the page width, which separates real columns without
    mistaking word spacing or table cells for a gutter.

    two_up: True forces the page to be cut down the middle before ordering.
    Pass document_is_two_up(doc) once per file; None means detect per page.
    """
    if min_gutter is None:
        min_gutter = page.rect.width * GUTTER_FRACTION

    blocks = _page_blocks(page)
    if not blocks:
        return ""
    # TWO-UP SPREADS ARE SPLIT, NOT INFERRED.
    #
    # Removing furniture is enough to expose the gutter on some spread pages
    # but not all: a figure straddling the fold, or an ordinary paragraph gap
    # that beats the gutter under COLUMN_BIAS, still sends _xy_cut down a
    # horizontal cut and interleaves the two logical pages. Measured on
    # ILIA_2023.pdf: furniture removal alone fixed 12 of 35 interleaved pages;
    # splitting at the fold fixed the rest.
    #
    # A physical page holding two logical pages is not a layout ambiguity to
    # be resolved by whitespace statistics -- it is a known fact about the
    # file. Cut at the midpoint and order each half independently.
    #
    # two_up=None falls back to per-page detection. Callers processing a whole
    # file should pass document_is_two_up(doc) instead: it is decided once,
    # from a sample, and applies to the pages where per-page detection fails
    # precisely because they are graphics.
    if two_up is None:
        two_up = is_two_up(page)
    # trust_input_order=True: PyMuPDF returns blocks in content-stream order,
    # which is often already correct on a designed document.
    ordered = order_blocks(blocks, page.rect.width, page.rect.height,
                           two_up=two_up, min_gap=min_gap,
                           trust_input_order=True)
    return "\n\n".join(b[4] for b in ordered)


def detect_columns(page, min_gutter: float | None = None,
                   min_gap: float = MIN_BAND_GAP) -> int:
    """Number of body columns on the page (diagnostics / metadata)."""
    if min_gutter is None:
        min_gutter = page.rect.width * GUTTER_FRACTION
    blocks, _furniture = _split_furniture(_page_blocks(page), page.rect.width)
    if not blocks:
        return 0
    return _count_columns(blocks, min_gutter, min_gap)



def split_spreads(path, out_path=None):
    """
    Rewrite a two-up PDF so each logical page is its own page.

    Returns (new_path, page_map) where page_map[i] is the 1-based ORIGINAL
    page that produced 0-based output page i, or None when the file is not a
    spread.

    WHY THIS EXISTS RATHER THAN TRUSTING A LAYOUT MODEL
        DocLayNet-trained detectors are trained on single-page documents. A
        1247x794 sheet holding two facing 624x794 pages is out of
        distribution: nothing in training says the fold is a hard boundary, so
        the model is free to order a left-page paragraph after a right-page
        one, exactly as XY-cut did. Feeding a spread to a layout model and
        hoping is not a fix, it is a different guess.

        Splitting is not a guess. Two logical pages side by side is a fact
        about the scan, the fold is at the midpoint, and a cropbox expresses
        that exactly. Normalise the geometry first, then let the model do what
        it is actually good at: pull-quotes, sidebars, captions, and reading
        order WITHIN a page.

    MIXED PAGE SIZES ARE THE COMMON CASE, NOT THE EXCEPTION.
        ILIA_2023.pdf is 160 landscape spreads plus 2 portrait pages (cover
        and colophon). Splitting every page would cut those two in half down
        the middle of a column. Only pages that are themselves landscape get
        split; the rest pass through untouched, and page_map records what came
        from where so `pagina` metadata stays honest.

    Implemented with cropboxes rather than by re-drawing: page objects are
    copied, nothing is rasterized or re-encoded, and text extraction on the
    result returns only the half in view.

    Requires: pip install pymupdf
    """
    import os
    import tempfile

    doc = fitz.open(path)
    if not document_is_two_up(doc):
        doc.close()
        return None

    out = fitz.open()
    page_map: list[int] = []

    for page in doc:
        rect = page.rect
        original = page.number + 1

        # Portrait pages inside a spread document are single logical pages.
        if rect.width <= rect.height * 1.25:
            out.insert_pdf(doc, from_page=page.number, to_page=page.number)
            page_map.append(original)
            continue

        mid = (rect.x0 + rect.x1) / 2
        for half in (fitz.Rect(rect.x0, rect.y0, mid, rect.y1),
                     fitz.Rect(mid, rect.y0, rect.x1, rect.y1)):
            out.insert_pdf(doc, from_page=page.number, to_page=page.number)
            out[-1].set_cropbox(half)
            page_map.append(original)

    if out_path is None:
        handle, out_path = tempfile.mkstemp(suffix="_split.pdf")
        os.close(handle)
    out.save(str(out_path))
    out.close()
    doc.close()
    return str(out_path), page_map


def profile_document(path, sample: int = 10) -> dict:
    """
    Quick diagnostic before indexing: how many pages are multi-column, and
    whether the PDF is digital or scanned (near-zero text -> needs full OCR).
    """
    doc = fitz.open(path)
    counts, chars, spreads = {}, 0, 0
    pages = list(doc)[:sample] if sample else list(doc)
    for page in pages:
        n = detect_columns(page)
        counts[n] = counts.get(n, 0) + 1
        chars += len(page.get_text("text"))
        spreads += is_two_up(page)
    n_pages = len(pages)
    doc.close()
    return {
        "columns_per_page": counts,
        "two_up_pages": spreads,
        "is_two_up": spreads > len(pages) / 2 if pages else False,
        "is_multicolumn": max(counts, key=counts.get) > 1 if counts else False,
        "chars_per_page": chars / max(n_pages, 1),
        "likely_scanned": chars / max(n_pages, 1) < 100,
    }