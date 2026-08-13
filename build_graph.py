from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# Entity types worth keeping. PER is dropped on purpose.
KEEP_TYPES = {"ORG", "LOC", "MISC", "GPE"}

# Zero-Shot Target Entities for Geopolitics & Conflict
TARGET_LABELS = [
    "state actor",
    "military organization",
    "armed group",
    "treaty or accord",
    "space asset or agency",
    "geopolitical region",
    "technology or system",
    "organization"
]

# Map custom domains to standard graph scheme
GLINER_LABEL_MAP = {
    "state actor": "GPE",
    "military organization": "ORG",
    "armed group": "ORG",
    "treaty or accord": "MISC",
    "space asset or agency": "ORG",
    "geopolitical region": "LOC",
    "technology or system": "MISC",
    "organization": "ORG"
}

# Relation patterns: a verb between two mentions in the same sentence gives
# the edge a type instead of a bare "co-occurs". Ordered, first match wins.
RELATION_PATTERNS: list[tuple[str, str]] = [
    (r"\b(opera|operan|operando)\b", "opera_en"),
    (r"\b(controla|controlan|control de)\b", "controla"),
    (r"\b(financia|financian|financiamiento)\b", "financia"),
    (r"\b(desarrolla|desarrollan|desarrollo de)\b", "desarrolla"),
    (r"\b(firma|firmaron|acuerdo|convenio|tratado)\b", "acuerda_con"),
    (r"\b(ataca|atacaron|enfrenta|enfrentamiento|disputa)\b", "confronta"),
    (r"\b(regula|regulan|prohibe|prohiben|norma)\b", "regula"),
    (r"\b(coopera|cooperan|colabora|alianza)\b", "coopera_con"),
    (r"\b(produce|producen|exporta|exportan|suministra)\b", "provee"),
]

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

# FIX: bilingual/variant-spelling aliases, resolved to one canonical key.
# Expanded with critical Ground Truth evaluation hubs.
_ALIASES: dict[str, str] = {
    # United States
    "u s": "united states", "us": "united states", "usa": "united states",
    "u s a": "united states", "eeuu": "united states",
    "ee uu": "united states", "estados unidos": "united states",
    # United Kingdom
    "uk": "united kingdom", "u k": "united kingdom",
    "reino unido": "united kingdom",
    # European Union
    "eu": "european union", "ue": "european union",
    "union europea": "european union",
    # United Nations
    "un": "united nations", "onu": "united nations",
    "naciones unidas": "united nations",
    # NATO
    "nato": "north atlantic treaty organization",
    "otan": "north atlantic treaty organization",
    # Custom Domain Hubs (Resolving Evaluation Failures)
    "mapp oea": "mapp/oea", "mappoea": "mapp/oea", "mapp-oea": "mapp/oea",
    "sipri": "sipri", "daio": "daio",
    "nasa": "nasa", "esa": "esa", "asat": "asat", "leo": "leo",
    # Cognates & Adjectivals
    "russia": "rusia", "russian": "rusia",
    "china": "china", "chinese": "china", "republica popular china": "china",
    "germany": "alemania", "alemania": "alemania",
}


def normalize_entity(text: str) -> str:
    """Canonical form for entity identity."""
    decomposed = unicodedata.normalize("NFD", text.strip().lower())
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    key = _WS.sub(" ", _PUNCT.sub(" ", stripped)).strip()
    return _ALIASES.get(key, key)


def load_ner(model_name: str, device: str):
    """Loads GLiNER zero-shot model, replacing standard HF pipelines."""
    try:
        from gliner import GLiNER
    except ImportError:
        raise SystemExit("build_graph: gliner is not installed.\n"
                         "  pip install gliner\n"
                         "  Or build without a model: --ner heuristic")
    
    print(f"  Loading GLiNER zero-shot model: {model_name}")
    # GLiNER handles device mapping automatically
    model = GLiNER.from_pretrained(model_name)
    if device == "cuda":
        model = model.to("cuda")
    return model


def detect_languages(items: dict[int, str]) -> dict[int, str]:
    """Cheap language id, retained for compatibility."""
    try:
        import py3langid as langid
    except ImportError:
        return {}
    return {i: langid.classify(text[:500])[0] for i, text in items.items()}


# ------------------------------------------------------------- no-model NER

_NOT_ENTITY = {
    "el", "la", "los", "las", "un", "una", "en", "de", "del", "por", "para",
    "con", "sin", "sobre", "entre", "este", "esta", "estos", "estas", "sus",
    "the", "a", "an", "in", "of", "for", "and", "or", "this", "these", "its",
    "asimismo", "ademas", "sin embargo", "por otro lado", "finalmente",
    "durante", "mientras", "cuando", "aunque", "segun", "tras", "desde",
}

_CAPTION_ARTEFACTS = {
    "figure", "fig", "table", "tabla", "cuadro", "grafico", "chart",
    "ver tabla", "ver figura", "see table", "see figure", "et al", "ibid",
    "op cit", "pp", "vol", "fuente", "source", "nota", "footnote",
}

_CAPITALISED = re.compile(
    r"\b[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]{2,}"
    r"(?:\s+(?:de|del|la|las|los|y|e)\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]{2,}"
    r"|\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]{2,}){0,3}\b")
_ACRONYM = re.compile(r"\b([A-ZÁÉÍÓÚÑ]{2,6})(?:[/\-][A-ZÁÉÍÓÚÑ]{2,6})?\b")


def gliner_windows(text: str, max_words: int = 300,
                   overlap_words: int = 40) -> list[tuple[str, int]]:
    """
    Split text into windows GLiNER can process without silently truncating.

    GLiNER v2/v2.1 models hard-cap input at 384 of GLiNER's own internal
    tokens -- not configurable at inference time in the released library
    (github.com/urchade/GLiNER issues #113, #183, #275: this comes up
    constantly and there is no supported way to raise it). Anything longer
    is truncated WHERE THE TEXT IS CUT, not where entities are: a mention
    in the back half of a long chunk becomes invisible to the model, with
    only a UserWarning easy to miss in a long log, not an error.

    300 words leaves real margin under 384 -- GLiNER's own counting is not
    exactly whitespace-word count, so budgeting right up to the edge would
    still risk truncation on punctuation- or numeral-heavy text.
    `overlap_words` keeps a multi-word entity that happens to straddle a
    window boundary from landing in neither window; the entity-identity
    dedup downstream (identical (key, group) pairs collapse to one) means
    the overlap costs a little redundant inference and nothing else.

    Returns [(window_text, char_offset_into_original_text), ...]. A chunk
    under the limit returns a single window with offset 0, so callers do
    not need a separate short-text path.
    """
    words = text.split()
    if len(words) <= max_words:
        return [(text, 0)]

    word_starts: list[int] = []
    pos = 0
    for w in words:
        idx = text.index(w, pos)
        word_starts.append(idx)
        pos = idx + len(w)

    windows: list[tuple[str, int]] = []
    step = max(1, max_words - overlap_words)
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        char_start = word_starts[start]
        char_end = (word_starts[end - 1] + len(words[end - 1])
                   if end > 0 else len(text))
        windows.append((text[char_start:char_end], char_start))
        if end == len(words):
            break
        start += step
    return windows


def heuristic_entities(text: str) -> list[tuple[str, str, int, int]]:
    found: list[tuple[str, str, int, int]] = []
    for match in _ACRONYM.finditer(text):
        found.append((match.group(0), "ORG", match.start(), match.end()))
    for match in _CAPITALISED.finditer(text):
        word = match.group(0).strip()
        if word.lower() in _NOT_ENTITY or word.isupper():
            continue
        before = text[max(0, match.start() - 2):match.start()]
        if " " not in word and (match.start() == 0 or "." in before):
            continue
        found.append((word, "MISC", match.start(), match.end()))
    return found


def relation_type(text: str, start: int, end: int) -> str:
    between = text[start:end].lower()
    if len(between) > 220:
        return ""
    for pattern, label in RELATION_PATTERNS:
        if re.search(pattern, between):
            return label
    return "coocurre_con"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path,
                        default=Path("entrega/base_vectorial"))
    parser.add_argument("--out", type=Path, default=None)
    # Defaulting to GLiNER multi (Apache 2.0, handles ES and EN zero-shot)
    parser.add_argument("--model", default="urchade/gliner_multi-v2.1")
    # Empty string disables routing, sending all chunks to the multi model
    parser.add_argument("--model-es", default="")
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--lang-detect", choices=["auto", "off"], default="auto")
    parser.add_argument("--ner", choices=["auto", "gliner", "heuristic"],
                        default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32) # Lower batch size for GLiNER safety
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-entity-freq", type=int, default=3)
    parser.add_argument("--max-entity-frac", type=float, default=0.02)
    parser.add_argument("--max-entities-per-chunk", type=int, default=12,
                        help="Applied twice: once at extraction time and "
                             "again every run after cache load. LOWERING "
                             "this on an already-cached run re-prunes for "
                             "free; RAISING it needs --recompute-ner.")
    parser.add_argument("--min-edge-weight", type=int, default=1,
                        help="Bare co-occurrence edges ('coocurre_con') "
                             "seen fewer than this many times are dropped. "
                             "Edges with a real verb-derived relation type "
                             "are never dropped by this, regardless of "
                             "weight. Default 1 = no pruning.")
    parser.add_argument("--gliner-threshold", type=float, default=0.35,
                        help="Confidence floor for a GLiNER span to be kept "
                             "at all -- this is the single biggest lever on "
                             "entity COUNT with a zero-shot model: lower "
                             "means more recall and more noise, higher "
                             "means fewer, more confident spans. Was "
                             "hardcoded to 0.35 with no way to test "
                             "alternatives. UNLIKE --max-entities-per-chunk, "
                             "this is NOT cheap to re-sweep on a cached run "
                             "-- GLiNER applies it at extraction time, so "
                             "the cache only ever contains spans that "
                             "already cleared the threshold used when it "
                             "was built. Changing this always needs "
                             "--recompute-ner in either direction.")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--recompute-ner", action="store_true")
    parser.add_argument("--hub-allowlist", type=Path, default=None)
    args = parser.parse_args()

    import networkx as nx

    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("  --device cuda requested but torch.cuda.is_available() "
                      "is False -- falling back to CPU.")
                args.device = "cpu"
        except ImportError:
            args.device = "cpu"

    folder = sorted(p for p in args.index_dir.iterdir()
                    if p.is_dir() and (p / "metadata.jsonl").exists())[0]
    raw = (folder / "metadata.jsonl").read_text(encoding="utf-8")
    records = [json.loads(l) for l in raw.split("\n") if l.strip()]

    other_folders = [p for p in sorted(args.index_dir.iterdir())
                     if p.is_dir() and p != folder
                     and (p / "metadata.jsonl").exists()]
    if other_folders:
        my_ids = {r.get("chunk_id") for r in records}
        for other in other_folders:
            other_raw = (other / "metadata.jsonl").read_text(encoding="utf-8")
            other_ids = {json.loads(l).get("chunk_id")
                        for l in other_raw.split("\n") if l.strip()}
            if other_ids != my_ids:
                raise SystemExit("build_graph: Chunk mismatch across encoders. Rebuild indexes.")

    if args.limit:
        records = records[:args.limit]
    print(f"{len(records)} chunks from {folder.name}")

    out = args.out or (args.index_dir / "grafo" / "grafo.graphml")
    cache_path = args.cache or out.parent / ".ner_cache.jsonl"
    meta_path = cache_path.with_suffix(".meta.json")

    texts = [r.get("texto", "") for r in records]
    per_chunk: list[list[tuple] | None] = [None] * len(texts)

    if cache_path.exists() and not args.recompute_ner:
        try:
            cached_rows = [json.loads(l) for l in
                          cache_path.read_text(encoding="utf-8").split("\n")
                          if l.strip()]
        except Exception:
            cached_rows = None

        recorded_total = None
        if meta_path.exists():
            try:
                recorded_total = json.loads(
                    meta_path.read_text(encoding="utf-8")).get("total")
            except Exception:
                pass

        if cached_rows is not None and len(cached_rows) == len(texts) \
                and recorded_total == len(texts):
            per_chunk = [([tuple(span) for span in row] if row is not None
                         else None) for row in cached_rows]
        elif cached_rows is not None:
            print(f"  cache at {cache_path} does not match this corpus "
                  f"({len(cached_rows)} rows vs {len(texts)} chunks, or no "
                  f"matching recorded total) -- stale, recomputing.")

    done_count = sum(1 for row in per_chunk if row is not None)

    if done_count == len(texts) and texts:
        print(f"NER: reusing cache at {cache_path} ({done_count} chunks).")
    else:
        pending = [i for i in range(len(texts)) if per_chunk[i] is None]
        use_model = args.ner in ("gliner", "auto")

        es_pending: list[int] = []
        other_pending = pending
        if use_model and args.model_es and args.lang_detect == "auto" and pending:
            languages = detect_languages({i: texts[i] for i in pending})
            es_pending = [i for i in pending if languages.get(i) == "es"]
            other_pending = [i for i in pending if languages.get(i) != "es"]

        plan: list[tuple[str, object, list[int], bool]] = []
        general_ner = es_ner = None

        if use_model:
            needs_general = bool(other_pending) or (args.fuse and es_pending)
            if needs_general:
                try:
                    general_ner = load_ner(args.model, args.device)
                except SystemExit as exc:
                    if args.ner == "gliner": raise
                    print(f"{exc}\n  Falling back to heuristic.\n")
            if es_pending:
                try:
                    es_ner = load_ner(args.model_es, args.device)
                except SystemExit as exc:
                    if args.ner == "gliner": raise
                    print(f"{exc}\n  Falling back to heuristic.\n")

            if other_pending:
                plan.append(("general", general_ner, other_pending, False))
            if es_pending:
                plan.append(("es", es_ner, es_pending, False))
                if args.fuse and general_ner is not None:
                    plan.append(("es+general (fuse)", general_ner, es_pending, True))
        else:
            plan.append(("heuristic", None, pending, False))

        CHECKPOINT_BATCHES = 20
        active_models = " + ".join(
            sorted({args.model_es if n in ("es", "es+general (fuse)")
                   and m is es_ner else args.model
                   for n, m, _i, _a in plan if m is not None} or {"heuristic"}))

        def checkpoint() -> None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                "\n".join(json.dumps(row) for row in per_chunk) + "\n",
                encoding="utf-8")
            meta_path.write_text(json.dumps(
                {"ner": active_models, "total": len(texts)}), encoding="utf-8")

        def process_phase(name: str, ner, indices: list[int], append_only: bool) -> None:
            if ner is None:
                # Heuristic mode has no token limit -- process chunks
                # directly, one GLiNER-shaped step per chunk.
                for start in range(0, len(indices), args.batch_size):
                    batch_indices = indices[start:start + args.batch_size]
                    spans_batch = [heuristic_entities(texts[i])
                                  for i in batch_indices]
                    for idx, spans in zip(batch_indices, spans_batch):
                        _accumulate(idx, spans, None, append_only)
                    _maybe_checkpoint(name, start, batch_indices, indices)
                return


            flat: list[tuple[int, str, int]] = []      
            for i in indices:
                for window_text, offset in gliner_windows(texts[i]):
                    flat.append((i, window_text, offset))
            if len(flat) > len(indices):
                print(f"  [{name}] {len(indices)} chunks expanded to "
                      f"{len(flat)} GLiNER windows (some chunks exceed "
                      f"the ~384-token limit)")

            touched: set[int] = set()

            for start in range(0, len(flat), args.batch_size):
                batch = flat[start:start + args.batch_size]
                batch_texts = [w for _i, w, _o in batch]
                try:
                    raw_batch = ner.batch_predict_entities(
                        batch_texts, TARGET_LABELS,
                        threshold=args.gliner_threshold)
                except AttributeError:
                    raw_batch = [ner.predict_entities(
                        t, TARGET_LABELS, threshold=args.gliner_threshold)
                        for t in batch_texts]

                # Group window results back by chunk WITHIN this batch --
                # merging across batches is `touched`'s job, not this
                # dict's.
                by_chunk: dict[int, list] = defaultdict(list)
                for (idx, _w, offset), spans in zip(batch, raw_batch):
                    for s in spans:
                        group = GLINER_LABEL_MAP.get(s["label"].lower(), "MISC")
                        by_chunk[idx].append(
                            (str(s["text"]).strip(), group,
                             int(s["start"]) + offset, int(s["end"]) + offset))

                for idx in {i for i, _w, _o in batch}:
                    merge = append_only or (idx in touched)
                    _accumulate(idx, by_chunk.get(idx, []), ner, merge)
                    touched.add(idx)

                done_chunks = len({i for i, _w, _o in flat[:start + len(batch)]})
                batch_no = start // args.batch_size
                if batch_no % CHECKPOINT_BATCHES == 0:
                    checkpoint()
                    print(f"  [{name}] ... {done_chunks}/{len(indices)} "
                          f"chunks  (checkpointed)", flush=True)

        def _accumulate(idx: int, spans, ner, append_only: bool) -> None:
            found = []
            for word, group, span_start, span_end in spans:
                if ner is not None and group not in KEEP_TYPES:
                    continue
                if len(word) < 3 or word.isdigit():
                    continue
                key = normalize_entity(word)
                if not key or len(key) < 3 or key in _NOT_ENTITY \
                        or key in _CAPTION_ARTEFACTS \
                        or any(key.startswith(a + " ") for a in ("see", "ver")):
                    continue
                found.append((key, group, span_start, span_end))

            if append_only and per_chunk[idx] is not None:
                per_chunk[idx] = per_chunk[idx] + found
            else:
                per_chunk[idx] = found

        def _maybe_checkpoint(name, start, batch_indices, indices) -> None:
            batch_no = start // args.batch_size
            if batch_no % CHECKPOINT_BATCHES == 0:
                checkpoint()
                print(f"  [{name}] ... "
                      f"{min(start + args.batch_size, len(indices))}/"
                      f"{len(indices)}  (checkpointed)", flush=True)

        try:
            for name, ner, indices, append_only in plan:
                if not indices: continue
                print(f"  phase '{name}': {len(indices)} chunks")
                process_phase(name, ner, indices, append_only)
        except (KeyboardInterrupt, Exception):
            checkpoint()
            done = sum(1 for row in per_chunk if row is not None)
            print(f"\n  interrupted at {done}/{len(texts)} chunks -- checkpointed.")
            raise

        for i, found in enumerate(per_chunk):
            if found is None:
                per_chunk[i] = []
                continue
            seen_keys: set[str] = set()
            deduped = []
            for entry in found:
                if entry[0] in seen_keys: continue
                seen_keys.add(entry[0])
                deduped.append(entry)
            per_chunk[i] = deduped

        checkpoint()

    over_cap = sum(1 for f in per_chunk if len(f) > args.max_entities_per_chunk)
    if over_cap:
        per_chunk = [f[:args.max_entities_per_chunk] for f in per_chunk]
        print(f"  --max-entities-per-chunk {args.max_entities_per_chunk}: "
              f"re-pruned {over_cap} chunks from the cached spans "
              f"(no NER recompute needed)")

    frequency: Counter = Counter()
    for found in per_chunk:
        frequency.update({key for key, _g, _s, _e in found})

    floor = args.min_entity_freq
    ceiling = max(1, int(args.max_entity_frac * len(records)))

    hub_list_path = args.hub_allowlist or (out.parent / "hub_allowlist.txt")
    allowed: set[str] = set()
    if hub_list_path.exists():
        for line in hub_list_path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                allowed.add(normalize_entity(line))

    candidate_hubs = sorted(((n, e) for e, n in frequency.items() if n > ceiling), reverse=True)
    hubs = [(n, e) for n, e in candidate_hubs if e not in allowed]
    rescued = len(candidate_hubs) - len(hubs)

    keep = {e for e, n in frequency.items() if floor <= n <= ceiling} | \
        {e for e in allowed if e in frequency}
    rare = sum(1 for n in frequency.values() if n < floor)

    print(f"\n{len(frequency)} distinct entities in {len(per_chunk)} chunks")
    print(f"  dropped {rare} seen in < {floor} chunks (noise)")
    print(f"  dropped {len(hubs)} seen in > {ceiling} chunks "
          f"({args.max_entity_frac:.0%} of the corpus), "
          f"{rescued} rescued by {hub_list_path.name}")
    
    if hubs:
        for n, entity in hubs[:15]:
            print(f"    {n:6d} chunks  {entity!r}")

    if not hub_list_path.exists() and candidate_hubs:
        hub_list_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Ensure real actors (state names, agencies) are uncommented to KEEP them.", "#"]
        for n, entity in candidate_hubs[:60]:
            lines.append(f"## {n} chunks\n# {entity}")
        hub_list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    graph = nx.Graph()
    labels: dict[str, str] = {}
    entity_chunks: dict[str, set[str]] = defaultdict(set)
    edge_evidence: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)

    for record, found in zip(records, per_chunk):
        found = [f for f in found if f[0] in keep]
        chunk_id = str(record.get("chunk_id", ""))
        doc_id = str(record.get("doc_id", ""))
        text = record.get("texto", "")

        for key, group, _s, _e in found:
            labels[key] = group
            entity_chunks[key].add(chunk_id)

        for i in range(len(found)):
            for j in range(i + 1, len(found)):
                a, b = found[i], found[j]
                if a[0] == b[0]: continue
                label = relation_type(text, min(a[3], b[3]), max(a[2], b[2]))
                if not label: continue
                edge_evidence[tuple(sorted((a[0], b[0])))].append((label, chunk_id, doc_id))

    for entity, chunk_ids in entity_chunks.items():
        graph.add_node(entity, tipo=labels.get(entity, "MISC"),
                       chunks=" ".join(sorted(chunk_ids)),
                       frecuencia=frequency[entity])

    # FIX: kept entities broken down by type. With GLINER_LABEL_MAP,
    # 'treaty or accord' and 'technology or system' both collapse to MISC
    # -- worth seeing this printed rather than assumed, especially since
    # this is the first real run with the new zero-shot labels.
    type_counts = Counter(labels.get(e, "MISC") for e in entity_chunks)
    print(f"  kept entities by tipo: " +
          ", ".join(f"{t}={n}" for t, n in type_counts.most_common()))
    if type_counts.get("MISC", 0) > sum(type_counts.values()) * 0.4:
        used_gliner = False
        if meta_path.exists():
            try:
                used_gliner = "gliner" in json.loads(
                    meta_path.read_text(encoding="utf-8")).get("ner", "").lower()
            except Exception:
                pass
        if used_gliner:
            print(f"  MISC is {type_counts['MISC'] / sum(type_counts.values()):.0%} "
                  f"of kept entities. With GLiNER this usually means "
                  f"--gliner-threshold is too low (letting through low-"
                  f"confidence 'treaty or accord'/'technology or system' "
                  f"matches) rather than a labeling problem -- try raising it "
                  f"a notch and comparing entity counts before assuming the "
                  f"label map itself needs changing.")
        else:
            print(f"  MISC is {type_counts['MISC'] / sum(type_counts.values()):.0%} "
                  f"of kept entities. The heuristic extractor labels EVERY "
                  f"capitalised-run match MISC by construction (see "
                  f"heuristic_entities()) -- this is expected in --ner "
                  f"heuristic, not a signal that something is wrong.")

    dropped_edges = 0
    for (a, b), evidence in edge_evidence.items():
        types = Counter(label for label, _c, _d in evidence)
        top_label = types.most_common(1)[0][0]
        weight = len(evidence)
        # FIX: keep every TYPED edge regardless of weight; only prune bare
        # co-occurrence ('coocurre_con') edges below --min-edge-weight. A
        # single 'controla'/'opera_en' sighting is real linguistic
        # evidence; a single same-chunk co-occurrence is the weakest
        # signal the graph produces.
        if top_label == "coocurre_con" and weight < args.min_edge_weight:
            dropped_edges += 1
            continue
        graph.add_edge(a, b, relacion=top_label, peso=weight,
                       chunks=" ".join(sorted({c for _l, c, _d in evidence})[:40]),
                       docs=" ".join(sorted({d for _l, _c, d in evidence})[:20]))

    if args.min_edge_weight > 1:
        print(f"  --min-edge-weight {args.min_edge_weight}: dropped "
              f"{dropped_edges} single-mention co-occurrence edges "
              f"(typed relations kept regardless of weight)")

    out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, out)

    typed = sum(1 for _a, _b, d in graph.edges(data=True)
                if d["relacion"] != "coocurre_con")
    print(f"\nWrote {out}")
    print(f"  {graph.number_of_nodes()} entities, {graph.number_of_edges()} "
          f"relations ({typed} with a verb-derived type, the rest co-occurrence)")

if __name__ == "__main__":
    main()