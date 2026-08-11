"""
build_graph.py — Knowledge graph over the indexed chunks (Section 7, bonus).

WHAT THE SPEC ASKS FOR
    G = (E, R, T), T subset of E x R x E. Entities from NER, relations from RE,
    every triple keeping a reference to its doc_id and chunk_id so the textual
    evidence stays traceable (7.2). Delivered as
    entrega/base_vectorial/grafo/grafo.graphml (1.4).

    And, decisively, from the organisers' FAQ:

        "Es bono y para que sea valido lo deben integrar a la recuperacion,
         el solo construirlo no es valido."

    So this file is only half the work. The other half is GraphIndex in
    generador.py, which makes the graph an actual retrieval channel.

DECODER-FREE (4.2, 8.3)
    NER here is token classification -- an encoder with a per-token label
    head, the same architecture family as the retrieval encoders. It assigns
    labels to spans; it generates nothing. Relation extraction is
    co-occurrence plus linguistic patterns, which 7.2 lists explicitly as an
    acceptable method ("heuristicas basadas en patrones linguisticos").

WHY CO-OCCURRENCE RELATIONS AND NOT A TRAINED RE MODEL
    A supervised relation classifier needs a relation schema and labelled
    data for this domain, and neither exists here. Co-occurrence within one
    chunk plus a verb pattern between the two mentions is weaker as knowledge
    but sound as RETRIEVAL evidence, which is what 8.5 actually uses the
    graph for: two entities named in the same passage are exactly the signal
    that makes that passage worth returning when a query names both.

    Being honest about that in informe_tecnico.pdf is better than implying a
    trained RE component that is not there.

Requires: pip install networkx transformers torch

Usage:
    python tools/build_graph.py --index-dir entrega/base_vectorial \
        --out entrega/base_vectorial/grafo/grafo.graphml
    python tools/build_graph.py --index-dir ... --limit 5000   # smoke test
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# Entity types worth keeping. PER is dropped on purpose: this corpus is about
# organisations, places and programmes, and person names are mostly report
# authors -- high frequency, no retrieval value, and they would dominate the
# graph's degree distribution.
KEEP_TYPES = {"ORG", "LOC", "MISC", "GPE"}

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


def normalize_entity(text: str) -> str:
    """
    Canonical form for entity identity.

    Accent- and case-insensitive, punctuation stripped, because the same
    organisation appears as "MAPP/OEA", "Mapp-OEA" and "MAPP OEA" across a
    corpus assembled from a dozen publishers. Without this the graph fills
    with singleton nodes that connect to nothing.
    """
    decomposed = unicodedata.normalize("NFD", text.strip().lower())
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return _WS.sub(" ", _PUNCT.sub(" ", stripped)).strip()


def load_ner(model_name: str, device: str):
    """
    HuggingFace token-classification NER, with the one failure that actually
    happens spelled out.

    XLM-RoBERTa repositories often ship only the slow tokenizer
    (sentencepiece.bpe.model) and no tokenizer.json. transformers then tries
    to CONVERT it, needs the sentencepiece package to read the protobuf,
    and -- if it is missing -- falls back to a tiktoken extractor that parses
    the binary file as text and dies on `Error parsing line b'\x0e'`. The
    traceback names tiktoken, which is not the problem; tiktoken is the
    fallback. The problem is one missing package.
    """
    try:
        from transformers import pipeline
    except ImportError:
        raise SystemExit("build_graph: transformers is not installed.\n"
                         "  pip install transformers sentencepiece\n"
                         "  Or build without a model: --ner heuristic")
    try:
        return pipeline("token-classification", model=model_name,
                        aggregation_strategy="simple",
                        device=0 if device == "cuda" else -1)
    except Exception as exc:
        message = str(exc)
        if "sentencepiece" in message.lower() or "tiktoken" in message.lower() \
                or "parsing line" in message:
            raise SystemExit(
                f"build_graph: {model_name} ships only the slow tokenizer, "
                f"and converting it needs\n  sentencepiece:\n\n"
                f"      pip install sentencepiece\n\n"
                f"  (The traceback blames tiktoken. tiktoken is only the "
                f"fallback that runs\n  once sentencepiece is missing, and it "
                f"chokes reading a binary protobuf.)\n\n"
                f"  Or skip the model entirely:  --ner heuristic")
        raise


# ------------------------------------------------------------- no-model NER

# Function words that start sentences and would otherwise look like entities.
_NOT_ENTITY = {
    "el", "la", "los", "las", "un", "una", "en", "de", "del", "por", "para",
    "con", "sin", "sobre", "entre", "este", "esta", "estos", "estas", "sus",
    "the", "a", "an", "in", "of", "for", "and", "or", "this", "these", "its",
    "asimismo", "ademas", "sin embargo", "por otro lado", "finalmente",
    "durante", "mientras", "cuando", "aunque", "segun", "tras", "desde",
}

# Captioning and citation artefacts. Distinct from _NOT_ENTITY above: these
# are not generic vocabulary that happens to get capitalised, they are a
# PREDICTABLE extraction artefact -- "Figure 1", "Table 2", "See Table"
# -- that the capitalised-run regex cannot tell apart from a real entity by
# pattern alone. Measured on this corpus: 'figure' reached 3855 chunks and
# would have survived any frequency ceiling wide enough to keep 'nasa' and
# 'china', which also sit in the thousands. This is a stoplist for a
# specific failure mode, not a second attempt at the same list.
_CAPTION_ARTEFACTS = {
    "figure", "fig", "table", "tabla", "cuadro", "grafico", "chart",
    "ver tabla", "ver figura", "see table", "see figure", "et al", "ibid",
    "op cit", "pp", "vol", "fuente", "source", "nota", "footnote",
}

# Capitalised runs, and acronyms of 2-6 capitals (ELN, MAPP, OEA, SIPRI).
# The connector must be allowed BETWEEN each pair, not once at the front:
# "Clan del Golfo" and "Norte de Santander" are single entities, and a
# pattern that only permits one leading connector splits them into "Clan del"
# + "Golfo", which then never match anything in a query.
_CAPITALISED = re.compile(
    r"\b[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]{2,}"
    r"(?:\s+(?:de|del|la|las|los|y|e)\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]{2,}"
    r"|\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]{2,}){0,3}\b")
_ACRONYM = re.compile(r"\b([A-ZÁÉÍÓÚÑ]{2,6})(?:[/\-][A-ZÁÉÍÓÚÑ]{2,6})?\b")


def heuristic_entities(text: str) -> list[tuple[str, str, int, int]]:
    """
    Entities without a model: capitalised multi-word runs plus acronyms.

    7.2 recommends NER models but does not mandate one, and a graph that
    exists beats a graph blocked on a tokenizer. This is measurably worse --
    it cannot type entities and it will catch sentence-initial words -- but
    the frequency floor in main() removes most of that noise, and the
    downstream use (8.5) only needs entity IDENTITY, not entity type.

    Use --ner hf when the model loads. Use this when it does not.
    """
    found: list[tuple[str, str, int, int]] = []
    for match in _ACRONYM.finditer(text):
        found.append((match.group(0), "ORG", match.start(), match.end()))
    for match in _CAPITALISED.finditer(text):
        word = match.group(0).strip()
        if word.lower() in _NOT_ENTITY or word.isupper():
            continue
        # A capitalised word at the very start of a sentence is usually just
        # a sentence start, unless it is multi-word.
        before = text[max(0, match.start() - 2):match.start()]
        if " " not in word and (match.start() == 0 or "." in before):
            continue
        found.append((word, "MISC", match.start(), match.end()))
    return found


def relation_type(text: str, start: int, end: int) -> str:
    """Verb pattern in the span between two mentions, or plain co-occurrence."""
    between = text[start:end].lower()
    if len(between) > 220:            # too far apart to be one statement
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
    parser.add_argument("--model", default="Davlan/xlm-roberta-base-ner-hrl",
                        help="multilingual token-classification NER; "
                             "encoder-only, no decoder")
    parser.add_argument("--ner", choices=["auto", "hf", "heuristic"],
                        default="auto",
                        help="auto tries the model and falls back; hf fails "
                             "loudly if it cannot load; heuristic uses no "
                             "model at all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-entity-freq", type=int, default=3,
                        help="entities seen in fewer than this many DISTINCT "
                             "chunks are dropped: OCR noise and one-off "
                             "strings, which add nodes and no edges")
    parser.add_argument("--max-entity-frac", type=float, default=0.02,
                        help="entities seen in more than this FRACTION of "
                             "all chunks are dropped as hubs -- a real named "
                             "entity discriminates between passages; a term "
                             "in 2%% of a 140k-chunk corpus (2800+ chunks) "
                             "is a generic word wearing a capital letter, "
                             "the same principle as removing stopwords in "
                             "the lexical channel.")
    parser.add_argument("--max-entities-per-chunk", type=int, default=12)
    parser.add_argument("--cache", type=Path, default=None,
                        help="raw per-chunk entity spans, cached before "
                             "frequency filtering. Re-running with a "
                             "different --min-entity-freq or "
                             "--max-entity-frac reuses it instead of "
                             "re-running NER over the whole corpus. Default: "
                             "next to --out.")
    parser.add_argument("--recompute-ner", action="store_true",
                        help="ignore an existing cache and re-run NER")
    parser.add_argument("--hub-allowlist", type=Path, default=None,
                        help="entities exempt from --max-entity-frac, one per "
                             "line ('#' comments allowed). Default: "
                             "grafo/hub_allowlist.txt next to --out. "
                             "Auto-created with the current hub list "
                             "commented out on first run -- see the ceiling "
                             "note in main() before editing it blind.")
    args = parser.parse_args()

    import networkx as nx

    folder = sorted(p for p in args.index_dir.iterdir()
                    if p.is_dir() and (p / "metadata.jsonl").exists())[0]
    raw = (folder / "metadata.jsonl").read_text(encoding="utf-8")
    records = [json.loads(l) for l in raw.split("\n") if l.strip()]
    if args.limit:
        records = records[:args.limit]
    print(f"{len(records)} chunks from {folder.name}")

    out = args.out or (args.index_dir / "grafo" / "grafo.graphml")
    cache_path = args.cache or out.parent / ".ner_cache.jsonl"

    # ---- pass 1: entities per chunk. CACHED, because this is the expensive
    # part (a GPU pass over the whole corpus with the model, or a regex pass
    # with the heuristic) and it does not depend on the frequency thresholds
    # being tuned below. Re-running with a different --max-entity-frac should
    # cost seconds, not the twenty-plus minutes NER takes -- the same reason
    # build_index.py caches extraction and Docling output.
    per_chunk: list[list[tuple[str, str, int, int]]] = []
    texts = [r.get("texto", "") for r in records]

    meta_path = cache_path.with_suffix(".meta.json")

    cached = None
    if cache_path.exists() and not args.recompute_ner:
        try:
            cached = [json.loads(l) for l in
                     cache_path.read_text(encoding="utf-8").split("\n") if l.strip()]
        except Exception:
            cached = None
        if cached is not None and len(cached) != len(texts):
            print(f"  cache at {cache_path} has {len(cached)} rows, corpus "
                  f"has {len(texts)} -- stale, recomputing.")
            cached = None

    if cached is not None:
        per_chunk = [[tuple(span) for span in row] for row in cached]
        # WHICH extractor produced this cache is not visible from the cache
        # file itself, and "NER: reusing cache" alone answers a different
        # question than the one that matters later -- whether 'figure'
        # showing up as a kept entity means the heuristic ran (it would
        # invent that from a caption) or the real model ran (it would not,
        # and 'figure' surviving would then mean something else). Recorded
        # once, at cache-write time, so it survives every reuse.
        source = "unknown (cache predates this check)"
        if meta_path.exists():
            try:
                source = json.loads(meta_path.read_text(encoding="utf-8")).get("ner", source)
            except Exception:
                pass
        print(f"NER: reusing cache at {cache_path} ({len(per_chunk)} chunks, "
              f"produced by: {source}).\n     Pass --recompute-ner to force a "
              f"fresh run.")
    else:
        use_model = args.ner in ("hf", "auto")
        ner = None
        if use_model:
            try:
                ner = load_ner(args.model, args.device)
                print(f"NER: {args.model}")
            except SystemExit as exc:
                if args.ner == "hf":
                    raise
                print(f"{exc}\n\n  --ner auto: falling back to the "
                      f"heuristic extractor.\n")
        if ner is None:
            print("NER: heuristic (capitalised runs + acronyms, no model)")

        for start in range(0, len(texts), args.batch_size):
            batch = texts[start:start + args.batch_size]
            if ner is not None:
                spans_batch = [[(str(s.get("word", "")).strip(),
                                str(s.get("entity_group", "")).upper(),
                                int(s["start"]), int(s["end"]))
                               for s in spans] for spans in ner(batch)]
            else:
                spans_batch = [heuristic_entities(t) for t in batch]

            for spans in spans_batch:
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
                per_chunk.append(found[:args.max_entities_per_chunk])
            if (start // args.batch_size) % 20 == 0:
                print(f"  ... {min(start + args.batch_size, len(texts))}/"
                      f"{len(texts)}", flush=True)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            "\n".join(json.dumps(row) for row in per_chunk) + "\n",
            encoding="utf-8")
        meta_path.write_text(json.dumps(
            {"ner": args.model if ner is not None else "heuristic"}), encoding="utf-8")
        print(f"  cached raw entity spans -> {cache_path}")

    # ---- frequency, counted by DISTINCT CHUNK, not by mention. A term
    # repeated ten times in one report and a term appearing once in ten
    # reports are very different evidence; only the second says anything
    # about the corpus.
    frequency: Counter = Counter()
    for found in per_chunk:
        frequency.update({key for key, _g, _s, _e in found})

    floor = args.min_entity_freq
    ceiling = max(1, int(args.max_entity_frac * len(records)))

    # THE CEILING'S ASSUMPTION FAILS ON A TOPICALLY-CONCENTRATED CORPUS, AND
    # THIS ONE IS ONE. "Frequent implies generic" is the same reasoning
    # tokenize() uses for stopwords, and it holds for stopwords because "de",
    # "la", "que" are frequent AND empty in ANY corpus. It does not hold
    # here: measured, the highest-frequency entities were 'united states',
    # 'china', 'nasa', 'esa', 'asat', 'leo', 'russia' -- the actual subject
    # of Fenomeno 2. They are frequent BECAUSE the corpus is about them, not
    # despite it, and a blind ceiling would delete exactly the entities a
    # query is most likely to name.
    #
    # 'figure' at comparable frequency, from the same corpus, is genuine
    # noise -- a caption artefact, now caught upstream by
    # _CAPTION_ARTEFACTS. The two cases are NOT distinguishable by frequency
    # alone, which is why an automatic numeric cutoff cannot be the final
    # word here: it is a proposal, reviewed once by a human, and remembered.
    hub_list_path = args.hub_allowlist or (out.parent / "hub_allowlist.txt")
    allowed: set[str] = set()
    if hub_list_path.exists():
        for line in hub_list_path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                allowed.add(normalize_entity(line))

    candidate_hubs = sorted(((n, e) for e, n in frequency.items() if n > ceiling),
                            reverse=True)
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
        # Frequency and entity name are on SEPARATE lines on purpose. Putting
        # "   350 chunks  nasa" on one line and asking the user to delete the
        # leading '#' looked convenient and was wrong: uncommenting exposed
        # "350 chunks  nasa" as the literal string being parsed, which
        # normalize_entity() then treated as one entity that matches nothing
        # in the graph -- the rescue silently did nothing. A `##` line stays
        # a comment forever; the `#` entity line is the only thing the user
        # ever needs to touch.
        lines = ["# One entity per line. Uncomment the ENTITY line (remove",
                "# its leading '#', leave the '##' frequency line above it",
                "# alone) to KEEP it regardless of --max-entity-frac -- for",
                "# a state actor, an agency, or any term that is frequent",
                "# because the corpus is ABOUT it, not because it is generic",
                "# vocabulary.",
                "#",
                "# Candidates from this run, most frequent first. Nothing",
                "# here is applied until you uncomment it and re-run -- reuses",
                "# the NER cache, costs seconds.",
                "#"]
        for n, entity in candidate_hubs[:60]:
            lines.append(f"## {n} chunks")
            lines.append(f"# {entity}")
        hub_list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n  Wrote {hub_list_path}: review it, uncomment real actors "
              f"like state names\n  and agencies, then re-run (the NER cache "
              f"makes this cost seconds, not the\n  original NER pass).")
    elif rescued:
        print(f"  {rescued} entities kept via {hub_list_path.name}: "
              f"{sorted(allowed & {e for _n, e in candidate_hubs})[:8]}")
    print(f"  kept: {len(keep)}")

    # ---- pass 2: nodes, edges, and the chunk references 7.2 requires
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
                if a[0] == b[0]:
                    continue
                label = relation_type(text, min(a[3], b[3]), max(a[2], b[2]))
                if not label:
                    continue
                edge_evidence[tuple(sorted((a[0], b[0])))].append(
                    (label, chunk_id, doc_id))

    for entity, chunk_ids in entity_chunks.items():
        graph.add_node(entity, tipo=labels.get(entity, "MISC"),
                       # GraphML has no list type: chunk references are stored
                       # space-separated, which is what generador.py parses.
                       chunks=" ".join(sorted(chunk_ids)),
                       frecuencia=frequency[entity])

    for (a, b), evidence in edge_evidence.items():
        types = Counter(label for label, _c, _d in evidence)
        graph.add_edge(a, b,
                       relacion=types.most_common(1)[0][0],
                       peso=len(evidence),
                       chunks=" ".join(sorted({c for _l, c, _d in evidence})[:40]),
                       docs=" ".join(sorted({d for _l, _c, d in evidence})[:20]))

    out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, out)

    typed = sum(1 for _a, _b, d in graph.edges(data=True)
                if d["relacion"] != "coocurre_com" and d["relacion"] != "coocurre_con")
    print(f"\nWrote {out}")
    print(f"  {graph.number_of_nodes()} entities, "
          f"{graph.number_of_edges()} relations ({typed} with a verb-derived "
          f"type, the rest co-occurrence)")
    print(f"  every edge carries the chunk_id and doc_id of its evidence (7.2)")
    print(f"\n  The graph only scores if it is USED. generador.py loads it "
          f"automatically\n  when grafo/grafo.graphml exists; check the "
          f"banner says 'graph: N entities'.")


if __name__ == "__main__":
    main()