"""
generador.py — Retrieval module (CODEFEST AD ASTRA 2026, sections 8-9).

The file name is mandated by section 1.4. The code inside is English.

SELF-CONTAINED BY DESIGN
    This file imports NOTHING from the rest of the repository. Everything it
    needs -- sentence segmentation, the 250-word splitter, fusion, filters --
    is defined here. The only inputs are entrega/base_vectorial/ and the
    query file, both of which ship inside entrega/. Section 1.4 says a
    submission that cannot be reproduced is excluded from evaluation, so the
    script must run from an unpacked entrega/ directory with nothing else
    present.

    Third-party requirements only:
        pip install faiss-cpu sentence-transformers numpy

Usage:
    python generador.py --index-dir base_vectorial \
        --queries consultas_50.jsonl --out resultados.jsonl

    # ablations (each flag isolates one component)
    python generador.py ... --bm25-weight 0
    python generador.py ... --dedupe-threshold 0
    python generador.py ... --phenomenon-boost 0
    python generador.py ... --doc-score cosine

------------------------------------------------------------------------
THE FOUR RANKING CHANNELS
------------------------------------------------------------------------
    dense_1..dense_n  one FAISS index per encoder (8.1, 8.2, 4.4)
    lexical           BM25 over metadata.jsonl, built at run time

    A dense-only pipeline is the wrong shape for this corpus. 74% of the
    documents are English, 100% of the queries are Spanish, and a large part
    of the answer set is named entities -- municipalities, armed groups
    ("ELN", "EMC", "Clan del Golfo"), agency acronyms, treaty names. Those
    are exactly the tokens a dense encoder blurs and exactly the ones BM25
    nails. BM25 is pure term statistics over metadata: no model, no decoder,
    permitted without argument under 8.3.

    The lexical index is built from metadata.jsonl at start-up, restricted to
    the vocabulary of the query set, so it costs one pass over the metadata
    and a few hundred megabytes. Nothing extra ships in entrega/.

------------------------------------------------------------------------
FRAGMENTS AND DOCUMENTS USE DIFFERENT SCORE SPACES
------------------------------------------------------------------------
    Fragments rank on RRF. RRF is rank-based, so it fuses channels whose
    scores are not comparable (cosine ~0.85, BM25 ~14) without any
    calibration. That is what 8.4 describes and it is the right tool for
    picking a top-10.

    Documents do NOT rank on RRF. RRF maps rank r to 1/(60+r): rank 1 scores
    0.0164 and rank 100 scores 0.0063. A document first seen at rank 100
    cannot catch the leader even with the full repetition bonus, so document
    order gets frozen by the first handful of candidates. The measured
    symptom was a doc-pool sweep from 10 to 300 returning byte-identical
    F1@3 at every value.

    Documents therefore rank on min-max normalised CombSUM (8.4, equation 5),
    which preserves the spread between a strong and a weak match. 8.6 also
    says aggregation operates on "las puntuaciones numéricas producidas por
    FAISS" -- scores, not fused ranks. `--doc-score cosine` restores the
    single-encoder behaviour for ablation.

------------------------------------------------------------------------
MULTI-ENCODER FUSION REQUIRES ONE SHARED CHUNK SET
------------------------------------------------------------------------
    Fusion keys on chunk_id, so every index MUST come from the same build.
    Two separate build runs can silently disagree (a file that failed OCR the
    first time and succeeded the second) and then chunk_id N means different
    text in each index. load_stores() warns when the counts differ.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import unicodedata
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# faiss / sentence_transformers are imported lazily inside VectorStore so that
# --help and the schema validator run without a torch install.


# =====================================================================
# defaults -- ONE source of truth, shared with evaluar.py
# =====================================================================
#
# evaluar.py imports these and builds its CLI from add_retrieval_args(), so
# the evaluated pipeline and the shipped pipeline cannot drift apart. They
# already did once: generador.py defaulted --doc-pool to 30 while evaluar.py
# defaulted it to 50, which means every F1@3 number printed for weeks
# described a configuration that was never written to resultados.jsonl.

RRF_K = 60                   # Reciprocal Rank Fusion smoothing constant (8.4)
# Separate, much smaller constant for the rerank blend. RRF_K=60 is tuned for
# fusing whole retrieval runs, where the tail matters; inside a 150-item
# shortlist it is so flat that the cross-encoder cannot lift rank 87 into the
# top 10 even with its top vote. k=10 makes position differences bite while
# still requiring a strong contrary signal to unseat a unanimous rank 1.
RERANK_RRF_K = 10
CANDIDATE_DEPTH = 1000       # results pulled per channel before fusion


@dataclass
class RetrievalConfig:
    depth: int = CANDIDATE_DEPTH
    min_score: float = 0.0

    # near-duplicate suppression on the fragment list
    dedupe_threshold: float = 0.45
    dedupe_window: int = 200

    # metadata post-filter (8.7)
    # FIX: fragments used mode="multiply" at 0.08 -- in RRF space (scores
    # ~0.016) that is worth about five rank positions, which measured out to
    # an 18% off-phenomenon leakage rate in the pooled candidates and near
    # zero difference against the "no phenomenon" ablation (Jaccard 0.98
    # between the two). Switched fragments to the same span-scaled "add"
    # mode documents already use, so one number means the same thing in
    # both places (see apply_phenomenon_boost's docstring on why "multiply"
    # does not port between score spaces). phenomenon_mode is a new flag so
    # the old multiplicative behaviour is still reachable for ablation.
    phenomenon_boost: float = 0.20       # fraction of the RRF pool's score span
    phenomenon_mode: str = "add"         # add | multiply -- add is span-scaled
    phenomenon_boost_doc: float = 0.30   # ADDITIVE, as a FRACTION of the score span

    # lexical channel
    # FIX (measured on pool.xlsx): bm25/graph and the dense encoders got
    # equal weight (1.0 each) in RRF, but BM25 and the graph's typed
    # relations only ever fire on same-language token overlap -- Spanish
    # queries against a 74%-English corpus. A chunk that matches in Spanish
    # gets a vote from every channel; a chunk that only a dense encoder can
    # cross-lingually match gets fewer votes purely because of language, not
    # relevance. Measured result: 79% of retrieved fragments were Spanish
    # against a 74%-English corpus. Down-weighting to 0.5 does not remove
    # the lexical signal (still valuable for acronyms and place names -- see
    # the module docstring) but stops it outvoting cross-lingual dense
    # matches by sheer channel count. Re-sweep if the encoder mix changes.
    bm25_weight: float = 0.5
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    graph_weight: float = 0.5   # 0 disables the graph channel; see bm25_weight
    graph_neighbour: float = 0.4
    rm3_terms: int = 0          # 0 disables pseudo-relevance feedback
    rm3_feedback: int = 10
    rm3_original_weight: float = 0.6
    mmr_lambda: float = 1.0     # 1.0 disables diversification

    # document aggregation (8.6)
    # FIX: default was "cosine", which means documents are ranked by
    # stores[0]'s single best chunk -- and stores[0] is whichever encoder
    # sorts first ALPHABETICALLY unless --doc-encoder is passed (see
    # load_stores()'s own docstring calling this "an arbitrary basis for a
    # scored decision"). Combined with doc_agg="max" (which the docstring
    # below already documents as making --doc-pool a no-op), document
    # ranking was decided by one lucky top-2 chunk from one arbitrarily-
    # chosen encoder, ignoring BM25 and the graph channel entirely at the
    # document level. Measured: q018/q019/q026 each had ~50% of their
    # candidate pool concentrated in a single document.
    #
    # New default "rankdecay" aggregates documents from the SAME fused,
    # phenomenon-boosted, reranked candidate list used for fragments,
    # scoring each document by reciprocal rank of its chunks' POSITIONS in
    # that list with a per-chunk decay (see aggregate_documents_rankdecay).
    # This uses every channel's agreement, not one encoder's raw cosine, and
    # removes the alphabetical-encoder dependency entirely. "combsum" and
    # "cosine" stay available for ablation against the old behaviour.
    doc_score: str = "rankdecay"     # rankdecay | combsum | cosine
    doc_decay: float = 0.85          # per-chunk decay for rankdecay, see below
    doc_pool: int = 30
    doc_agg: str = "max"
    doc_top_m: int = 3
    doc_hit_bonus: float = 0.02
    doc_hit_cap: int = 3

    # optional cross-encoder rerank (see rerank() for the compliance note)
    reranker: str = "BAAI/bge-reranker-v2-m3"
    rerank_depth: int = 150
    rerank_blend: float = 0.5
    rerank_context: bool = True

    weights: list[float] = field(default_factory=list)


def add_retrieval_args(parser: argparse.ArgumentParser) -> None:
    """Every knob that changes ranking. Imported by evaluar.py verbatim."""
    d = RetrievalConfig()
    parser.add_argument("--index-dir", type=Path, default=Path("base_vectorial"),
                        help="Folder CONTAINING the encoder_*/ subfolders, "
                             "each with index.faiss + metadata.jsonl. "
                             "Resolved against the current directory first, "
                             "then against this script's own directory.")
    parser.add_argument("--depth", type=int, default=d.depth)
    parser.add_argument("--min-score", type=float, default=d.min_score)
    parser.add_argument("--dedupe-threshold", type=float, default=d.dedupe_threshold,
                        help="Shingle containment above which a fragment is a "
                             "repeat of one already selected. 0 disables. "
                             "Measured: adjacent windows ~0.31, repeated "
                             "boilerplate ~0.95. Below 0.31 you start "
                             "suppressing neighbours; sweep it.")
    parser.add_argument("--phenomenon-boost", type=float, default=d.phenomenon_boost,
                        help="Bonus for chunks whose `fenomeno` matches the "
                             "query-id range, in the space set by "
                             "--phenomenon-mode.")
    parser.add_argument("--phenomenon-mode", choices=["add", "multiply"],
                        default=d.phenomenon_mode,
                        help="add (default) = span-scaled, same unit as "
                             "--phenomenon-boost-doc. multiply = old "
                             "behaviour, kept for ablation only -- see "
                             "apply_phenomenon_boost().")
    parser.add_argument("--phenomenon-boost-doc", type=float,
                        default=d.phenomenon_boost_doc,
                        help="Bonus in cosine/CombSUM space, expressed as a "
                             "FRACTION OF THE POOL'S SCORE SPAN so that one "
                             "value means the same thing in either space. "
                             "0.30 = 30%% of the range between the best and "
                             "worst candidate.")
    parser.add_argument("--graph-weight", type=float, default=d.graph_weight,
                        help="weight of the knowledge-graph channel in the "
                             "fusion (8.5). 0 disables it. Loaded "
                             "automatically from base_vectorial/grafo/"
                             "grafo.graphml when present.")
    parser.add_argument("--graph-neighbour", type=float, default=d.graph_neighbour,
                        help="discount applied to chunks reached through a "
                             "first-order neighbour rather than the entity "
                             "itself")
    parser.add_argument("--bm25-weight", type=float, default=d.bm25_weight,
                        help="Weight of the lexical channel. 0 disables it "
                             "and skips the index build entirely.")
    parser.add_argument("--rm3-terms", type=int, default=d.rm3_terms,
                        help="RM3 pseudo-relevance feedback: expansion terms "
                             "taken from the lexical channel's own top hits. "
                             "0 disables. No model involved, so 8.3's ban on "
                             "decoder-based query expansion does not apply.")
    parser.add_argument("--rm3-feedback", type=int, default=d.rm3_feedback)
    parser.add_argument("--mmr-lambda", type=float, default=d.mmr_lambda,
                        help="Maximal Marginal Relevance on the fragment "
                             "list. 1.0 = off. Lower trades relevance for "
                             "coverage, which is what nDCG@10 rewards when a "
                             "query has several relevant passages.")
    parser.add_argument("--doc-score", choices=["rankdecay", "combsum", "cosine"],
                        default=d.doc_score,
                        help="Score space documents are aggregated over (8.6). "
                             "rankdecay (default, see RetrievalConfig for why) "
                             "ignores --doc-pool/--doc-agg/--doc-hit-* entirely "
                             "and uses --doc-decay instead; combsum/cosine are "
                             "kept for ablation against the old behaviour and "
                             "use the doc-pool/doc-agg/hit-bonus knobs below.")
    parser.add_argument("--doc-decay", type=float, default=d.doc_decay,
                        help="rankdecay only: weight of a document's i-th "
                             "ranked chunk is doc_decay**i. Lower rewards "
                             "breadth less; 1.0 makes every chunk count "
                             "equally (closer to old doc_agg='sum').")
    parser.add_argument("--doc-agg", choices=["max", "sum", "mean", "rrf"],
                        default=d.doc_agg,
                        help="how a document's chunk scores become one score "
                             "(8.6). max is pure best-chunk and makes "
                             "--doc-pool inert; sum/mean/rrf reward a document "
                             "with several good passages.")
    parser.add_argument("--doc-top-m", type=int, default=d.doc_top_m,
                        help="chunks per document combined by sum/mean/rrf")
    parser.add_argument("--doc-pool", type=int, default=d.doc_pool,
                        help="k_chunk: chunks aggregated into document scores.")
    parser.add_argument("--doc-hit-bonus", type=float, default=d.doc_hit_bonus,
                        help="Per-extra-chunk bonus in the document score. The "
                             "old 0.05 x 5 = 25%% dwarfed the ~11%% spread of "
                             "the cosine scores it multiplied, which made "
                             "document ranking a chunk-count contest.")
    parser.add_argument("--doc-hit-cap", type=int, default=d.doc_hit_cap)
    parser.add_argument("--doc-encoder", default="",
                        help="Substring of the encoder that leads the store "
                             "list, e.g. 'e5'. Default: alphabetical.")
    parser.add_argument("--reranker", default=d.reranker,
                        help="Optional cross-encoder, e.g. "
                             "'BAAI/bge-reranker-v2-m3'. Empty = off. Read the "
                             "compliance note in rerank() before enabling.")
    parser.add_argument("--rerank-depth", type=int, default=d.rerank_depth)
    parser.add_argument("--no-rerank-context", action="store_false",
                        dest="rerank_context", default=d.rerank_context,
                        help="score the cross-encoder on `texto` alone "
                             "instead of `contexto. texto`")
    parser.add_argument("--rerank-blend", type=float, default=d.rerank_blend,
                        help="1.0 = cross-encoder order replaces the "
                             "retrievers'. 0.5 = equal RRF vote. Replacing "
                             "outright demotes chunks every retriever agreed "
                             "on; see rerank().")


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_input(path: Path) -> Path:
    """
    Find an input whether the script is run from inside entrega/ or from the
    repository root.

    The defaults are written for the jury, who unpack entrega/, cd into it
    and run `python generador.py`. Developing from the repo root means the
    same relative path points somewhere else, and the failure mode is a
    SystemExit about a missing index that looks like a broken build. So: try
    the path as given, then the same path next to this script, and say which
    one was used.
    """
    if path.exists():
        return path
    beside = SCRIPT_DIR / path
    if beside.exists():
        return beside
    raise SystemExit(
        f"Not found: {path}\n"
        f"  looked in {path.resolve()}\n"
        f"       and {beside}\n"
        f"  --index-dir must point at the folder CONTAINING the encoder_*/ "
        f"subfolders,\n  e.g. entrega/base_vectorial from the repo root, or "
        f"base_vectorial from inside entrega/.")


def config_from_args(args) -> RetrievalConfig:
    cfg = RetrievalConfig()
    for name in vars(cfg):
        if hasattr(args, name):
            setattr(cfg, name, getattr(args, name))
    return cfg


# =====================================================================
# sentence segmentation  (inlined from chunking.py -- see module docstring)
# =====================================================================
#
# Only split_to_250_words() needs this at query time, and only for chunks
# that exceed the 250-word output cap (9.2.1). Under the shipped chunking
# parameters that never happens, but the rule is mandatory and a chunk set
# built with different parameters must still produce a legal file.

_ABBREV = (r"Sr|Sra|Srta|Dr|Dra|Ing|Lic|Mg|Prof|Ph\.D|EE\.UU|EEUU|etc|vs|cf|"
           r"p\.ej|aprox|núm|No|Nro|Art|Fig|Tab|Cap|Vol|ed|eds|al|Mr|Mrs|Ms|"
           r"St|Jr|Inc|Ltd|Co|U\.S|U\.K|e\.g|i\.e|approx|Ref|Eq")

_PROTECT = [
    (re.compile(r"\b(?:[A-ZÁÉÍÓÚÑ]{1,2}\.){2,}"),
     lambda m: m.group(0).replace(".", "@@")),
    (re.compile(rf"\b({_ABBREV})\.", re.IGNORECASE), r"\1@@"),
    (re.compile(r"\b(\d+)\.(\d)"), r"\1@@\2"),
    (re.compile(r"\b([A-ZÁÉÍÓÚÑ])\.(?=\s*[A-ZÁÉÍÓÚÑ])"), r"\1@@"),
]

_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"'»)\]]*\s+")
_SEGMENTER: object | None | bool = None


def _get_segmenter():
    """pysbd if installed and not disabled; built once per process."""
    global _SEGMENTER
    if _SEGMENTER is None:
        if os.environ.get("CODEFEST_NO_PYSBD"):
            _SEGMENTER = False
        else:
            try:
                import pysbd
                _SEGMENTER = pysbd.Segmenter(language="es", clean=False)
            except Exception:
                _SEGMENTER = False
    return _SEGMENTER or None


def split_sentences(text: str) -> list[str]:
    """
    Multilingual sentence splitter (es/en/pt). The regex fallback masks
    dotted acronyms, abbreviations, decimals and initials before cutting, so
    it does not shatter "EE.UU." or "3.5 millones" into fake sentences.
    """
    segmenter = _get_segmenter()
    if segmenter is not None:
        try:
            out = []
            for paragraph in text.split("\n\n"):
                if paragraph.strip():
                    out += [s.strip() for s in segmenter.segment(paragraph) if s.strip()]
            return out
        except Exception:
            pass
    out = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        masked = paragraph
        for pattern, replacement in _PROTECT:
            masked = pattern.sub(replacement, masked)
        for sentence in _SENTENCE_END.split(masked):
            sentence = sentence.replace("@@", ".").strip()
            if sentence:
                out.append(sentence)
    return out


def split_to_250_words(text: str, limit: int = 250) -> list[str]:
    """
    Split an oversized chunk into sub-fragments of <= limit words, cutting
    only at sentence boundaries (9.2.1). All sub-fragments keep the original
    chunk_id and each takes its own rank.
    """
    if len(text.split()) <= limit:
        return [text]

    parts, current, n_words = [], [], 0
    for sentence in split_sentences(text):
        words = len(sentence.split())
        if current and n_words + words > limit:
            parts.append(" ".join(current))
            current, n_words = [], 0
        current.append(sentence)
        n_words += words
    if current:
        parts.append(" ".join(current))

    # A single sentence over 250 words is pathological; hard-cut it rather
    # than emit an illegal fragment. 9.3.2 discards oversized fragments.
    final = []
    for part in parts:
        words = part.split()
        if len(words) <= limit:
            final.append(part)
        else:
            for i in range(0, len(words), limit):
                final.append(" ".join(words[i:i + limit]))
    return final


# =====================================================================
# text normalisation shared by the lexical index and the deduplicator
# =====================================================================

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    """
    PDF extraction and OCR disagree about accents constantly, and Spanish
    queries are typed both ways. "informacion" and "información" are the same
    token for retrieval purposes.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


# U+0085 (NEL), U+2028 (LINE SEPARATOR) and U+2029 (PARAGRAPH SEPARATOR).
#
# These three are the reason a metadata.jsonl that json.dumps wrote can fail
# to parse back. json.dumps(ensure_ascii=False) escapes control characters
# below 0x20, but not these -- they go into the file raw. Python's
# str.splitlines() then treats all three as line breaks, so any reader built
# on read_text().splitlines() cuts a record in half and json.loads reports an
# unterminated string. Iterating the file object, or splitting on "\n", does
# not have this problem.
#
# They arrive from PDF extraction and from OCR of Latin-1 sources, and they
# are invisible in every editor, so this looks like a corrupt index when
# nothing is wrong with it.
_LINE_SEPARATORS = re.compile("[\u0085\u2028\u2029]")


def sanitize_text(text: str) -> str:
    """Replace the invisible separators with a plain space."""
    return _LINE_SEPARATORS.sub(" ", text)


def read_jsonl(path: Path) -> list[dict]:
    """
    Read JSON Lines safely. Splits on "\n" ONLY -- see _LINE_SEPARATORS.
    9.3 defines the newline as the sole delimiter, so this is also the
    literal reading of the spec.
    """
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").split("\n") if line.strip()]


# Function words in the three corpus languages. Removing them keeps the
# lexical vocabulary (and therefore the postings lists) down to the terms
# that actually discriminate.
_STOPWORDS = set("""
de la que el en y a los del se las por un para con no una su al lo como mas
pero sus le ya o este si porque esta entre cuando muy sin sobre tambien me
hasta hay donde quien desde todo nos durante todos uno les ni contra otros
ese eso ante ellos e esto mi antes algunos qué unos yo otro otras otra él
tanto esa estos mucho quienes nada muchos cual sea poco ella estar haber
estas estaba estamos algunas algo nosotros
son ser sido siendo esta estan estos estas fue fueron era eran han ha hemos
cuales cuál cual como cuando donde cuanto cuantos quienes cuyo cuya cuyos
que qué segun asi tal tales cada mismo misma mismos mismas dentro fuera
puede pueden podria podrian debe deben hace hacen tiene tienen
the of and to in a is for on that with as by at from or an be are this it
was were which has have had not but their its can will more other such
o e do da em para com no na os as um uma dos das ao pelo pela por mais como
mas ou se que nao ser sao foi ate entre sobre
""".split())


def tokenize(text: str) -> list[str]:
    """Accent-free lowercase word tokens, stopwords and 1-char tokens removed."""
    lowered = strip_accents(text.lower())
    words = _WS.sub(" ", _PUNCT.sub(" ", lowered)).split()
    return [w for w in words if len(w) > 1 and w not in _STOPWORDS]


def shingles(text: str, n: int = 8) -> set[tuple]:
    """Accent- and punctuation-insensitive word n-grams, for dedupe only."""
    lowered = strip_accents(text.lower())
    words = _WS.sub(" ", _PUNCT.sub(" ", lowered)).strip().split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


# =====================================================================
# dense channel
# =====================================================================

class VectorStore:
    def __init__(self, folder: Path):
        import faiss
        from sentence_transformers import SentenceTransformer

        config = json.loads((folder / "encoder.json").read_text(encoding="utf-8"))
        self.folder = folder
        self.name = config["model"]
        self.query_prefix = config.get("query_prefix", "")
        # Some encoders ship a custom architecture whose modelling code lives
        # in the HF repo rather than in transformers. build_index.py records
        # the flag in encoder.json so it cannot drift between index time and
        # query time.
        self.model = SentenceTransformer(
            self.name, trust_remote_code=bool(config.get("trust_remote_code")))
        self.index = faiss.read_index(str(folder / "index.faiss"))
        # line i of metadata.jsonl <-> FAISS internal id i (1.4)
        with (folder / "metadata.jsonl").open(encoding="utf-8") as fh:
            self.metadata = [json.loads(line) for line in fh if line.strip()]
        if self.index.ntotal != len(self.metadata):
            raise SystemExit(
                f"{folder}: {self.index.ntotal} vectors vs "
                f"{len(self.metadata)} metadata lines. The index and the "
                f"metadata store are not in the same order; rebuild.")

    def search(self, query: str, k: int) -> list[tuple[dict, float]]:
        # Same encoder, same prefix, same normalisation as indexing time (8.1)
        vector = self.model.encode([self.query_prefix + query],
                                   convert_to_numpy=True,
                                   normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(vector, min(k, self.index.ntotal))
        return [(self.metadata[i], float(s))
                for i, s in zip(ids[0], scores[0]) if i != -1]


def load_stores(index_dir: Path, doc_encoder: str = "",
                doc_score: str = "") -> list[VectorStore]:
    """
    Load every encoder_* folder. The FIRST store leads: its cosine scores are
    what `--doc-score cosine` aggregates over, and its metadata is what the
    lexical index is built from. `doc_encoder` (a substring of the model
    name) moves a chosen encoder to the front, because alphabetical folder
    order is an arbitrary basis for a scored decision.
    """
    folders = sorted(p for p in index_dir.iterdir()
                     if p.is_dir() and (p / "index.faiss").exists())
    if not folders:
        raise SystemExit(f"No encoder_*/index.faiss found under {index_dir}")

    stores = [VectorStore(f) for f in folders]

    if doc_encoder:
        stores.sort(key=lambda s: doc_encoder.lower() not in s.name.lower())
        if doc_encoder.lower() not in stores[0].name.lower():
            print(f"  WARNING: no index matches --doc-encoder {doc_encoder!r}; "
                  f"leading with {stores[0].name}")
        if doc_score in ("combsum", "rankdecay"):
            print(f"  WARNING: --doc-encoder {doc_encoder!r} has NO EFFECT "
                  f"under --doc-score {doc_score}. combsum adds normalised "
                  f"scores from every channel and rankdecay aggregates from "
                  f"the fused fragment ranking directly -- in both cases "
                  f"addition/fusion is commutative, so which encoder leads "
                  f"is irrelevant; stores[0] is only consulted by "
                  f"--doc-score cosine. Results here will be identical to "
                  f"any other --doc-encoder value.")

    sizes = {s.index.ntotal for s in stores}
    if len(sizes) > 1:
        print(f"  WARNING: indexes hold different chunk counts {sorted(sizes)}. "
              "Fusion keys on chunk_id and assumes one shared chunk set. "
              "Rebuild every encoder in a single build_index run.")
    return stores


# =====================================================================
# lexical channel  (BM25 over metadata.jsonl)
# =====================================================================

class LexicalIndex:
    """
    BM25 restricted to the vocabulary of the query set.

    A full inverted index over ~133k chunks is ~20M postings, which in pure
    Python is gigabytes. But the query set is known before retrieval starts:
    50 questions carry a few hundred distinct content words between them.
    Only those terms can ever contribute to a score, so only those get
    postings. The build is one pass over the metadata and a few hundred MB.

    The text indexed is `contexto` + `texto`, matching what build_index.py
    hands the encoder, so the two channels see the same document.
    """

    def __init__(self, metadata, postings, doc_len, k1: float, b: float):
        self.metadata = metadata
        self.postings = postings
        self.doc_len = doc_len
        self.n_docs = len(doc_len)
        self.avgdl = (sum(doc_len) / self.n_docs) if self.n_docs else 1.0
        self.k1, self.b = k1, b
        self.idf = {}
        for term, plist in postings.items():
            df = len(plist) // 2
            self.idf[term] = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)) if df else 0.0

    @classmethod
    def build(cls, metadata: list[dict], queries: list[str],
              k1: float = 1.2, b: float = 0.75) -> "LexicalIndex":
        vocab: set[str] = set()
        for query in queries:
            vocab.update(tokenize(query))

        postings: dict[str, array] = {term: array("i") for term in vocab}
        doc_len = array("i")

        for meta in metadata:
            text = meta.get("texto", "")
            context = meta.get("contexto", "")
            tokens = tokenize(f"{context} {text}" if context else text)
            doc_len.append(len(tokens))
            if not tokens:
                continue
            for term, tf in Counter(t for t in tokens if t in vocab).items():
                postings[term].extend((len(doc_len) - 1, tf))

        return cls(metadata, postings, doc_len, k1, b)

    def search(self, query: str, k: int) -> list[tuple[dict, float]]:
        scores: dict[int, float] = defaultdict(float)
        k1, b, avgdl = self.k1, self.b, self.avgdl

        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for i in range(0, len(plist), 2):
                index, tf = plist[i], plist[i + 1]
                norm = 1.0 - b + b * (self.doc_len[index] / avgdl)
                scores[index] += idf * (tf * (k1 + 1.0)) / (tf + k1 * norm)

        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.metadata[i], float(s)) for i, s in top]



# =====================================================================
# knowledge graph channel  (spec sections 7 and 8.5, bonus)
# =====================================================================

class GraphIndex:
    """
    The knowledge graph as a retrieval channel.

    "Es bono y para que sea valido lo deben integrar a la recuperacion, el
    solo construirlo no es valido." (organisers' FAQ). Building grafo.graphml
    scores nothing on its own; this class is what makes it count.

    Section 8.5, implemented step for step:
      1. identify the entities named in the query;
      2. pull the chunks linked to those entities AND to their first-order
         neighbours in the graph;
      3. score each chunk by the number of relevant relations behind it;
      4. fuse that ranking with the FAISS results as one more index in the
         RRF of 8.4.

    READ WITH THE STANDARD LIBRARY, ON PURPOSE. GraphML is XML, and
    xml.etree parses it in thirty lines. networkx is needed to BUILD the
    graph but shipping it as a query-time dependency would add a package to
    entrega/'s install for no capability -- and 1.4 excludes a submission
    that cannot be reproduced.

    WHY IT ADDS SOMETHING THE ENCODERS DO NOT. A dense encoder places "ELN"
    and "Catatumbo" near each other because they co-occur in training text.
    The graph says they co-occur in THIS corpus, in a specific passage, and
    names the passage. That is a different kind of evidence, and it is
    strongest exactly where embeddings are weakest: rare named entities.
    """

    def __init__(self, entities: dict, edges: list):
        self.entities = entities            # normalised name -> {tipo, chunks}
        self.neighbours: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for a, b, weight in edges:
            self.neighbours[a].append((b, weight))
            self.neighbours[b].append((a, weight))
        # Longest first, so "clan del golfo" is matched before "golfo".
        self.by_length = sorted(entities, key=len, reverse=True)

    @classmethod
    def load(cls, path: Path) -> "GraphIndex | None":
        import xml.etree.ElementTree as ET

        if not path.exists():
            return None
        namespace = {"g": "http://graphml.graphdrawing.org/xmlns"}
        root = ET.parse(path).getroot()

        # GraphML indirects attribute names through <key id=... attr.name=...>
        names = {k.get("id"): k.get("attr.name")
                 for k in root.findall("g:key", namespace)}

        def data(element) -> dict:
            return {names.get(d.get("key"), d.get("key")): (d.text or "")
                    for d in element.findall("g:data", namespace)}

        graph = root.find("g:graph", namespace)
        if graph is None:
            return None

        entities = {}
        for node in graph.findall("g:node", namespace):
            fields = data(node)
            entities[node.get("id")] = {
                "tipo": fields.get("tipo", "MISC"),
                "chunks": fields.get("chunks", "").split()}

        edges = []
        for edge in graph.findall("g:edge", namespace):
            fields = data(edge)
            try:
                weight = int(fields.get("peso", 1))
            except ValueError:
                weight = 1
            edges.append((edge.get("source"), edge.get("target"), weight))

        return cls(entities, edges)

    def link(self, query: str) -> list[str]:
        """
        Entities named in the query.

        Gazetteer matching against the graph's own vocabulary rather than a
        second NER pass: the graph already contains every entity the NER
        found in the corpus, an entity absent from it has no chunks to
        contribute, and this keeps entrega/ free of a transformers
        dependency. Matching is on the same normalisation used at build time.
        """
        text = f" {strip_accents(query.lower())} "
        text = _WS.sub(" ", _PUNCT.sub(" ", text))
        found, consumed = [], []
        for name in self.by_length:
            padded = f" {name} "
            if padded in text and not any(name in c for c in consumed):
                found.append(name)
                consumed.append(name)
        return found

    def search(self, query: str, k: int, metadata_by_chunk: dict,
               neighbour_weight: float = 0.4) -> list[tuple[dict, float]]:
        seeds = self.link(query)
        if not seeds:
            return []

        scores: dict[str, float] = defaultdict(float)
        for seed in seeds:
            # Direct evidence: the entity is named in the chunk.
            for chunk_id in self.entities.get(seed, {}).get("chunks", ()):
                scores[chunk_id] += 1.0
            # First-order neighbours (8.5 step 2), discounted, and weighted by
            # how many times the relation was actually observed -- an edge
            # seen forty times is stronger evidence than one seen once.
            for other, weight in self.neighbours.get(seed, ()):
                bonus = neighbour_weight * min(weight, 10) / 10.0
                for chunk_id in self.entities.get(other, {}).get("chunks", ()):
                    scores[chunk_id] += bonus

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(metadata_by_chunk[c], s) for c, s in ranked
                if c in metadata_by_chunk]


# =====================================================================
# fusion
# =====================================================================

def fuse_rrf(rankings: list[list[tuple[dict, float]]],
             weights: list[float] | None = None) -> list[tuple[dict, float]]:
    """
    Weighted Reciprocal Rank Fusion (8.4). Rank-based, so it needs no score
    calibration between a cosine channel and a BM25 channel.
    """
    weights = weights or [1.0] * len(rankings)
    points: dict[str, float] = defaultdict(float)
    registry: dict[str, dict] = {}

    for ranking, weight in zip(rankings, weights):
        if weight <= 0:
            continue
        for rank, (meta, _score) in enumerate(ranking, start=1):
            key = meta["chunk_id"]
            points[key] += weight / (RRF_K + rank)
            registry[key] = meta

    ordered = sorted(points.items(), key=lambda kv: -kv[1])
    return [(registry[key], score) for key, score in ordered]


def fuse_combsum(rankings: list[list[tuple[dict, float]]],
                 weights: list[float] | None = None) -> list[tuple[dict, float]]:
    """
    Min-max normalised CombSUM (8.4, equation 5).

    Used for the DOCUMENT level only. RRF compresses everything into
    [1/(60+k), 1/61], a range too narrow for 8.6's aggregation to
    discriminate: max pooling over near-identical numbers cannot tell a
    strong document from a mediocre one. CombSUM over normalised scores keeps
    the spread that the aggregation step needs.

    A chunk absent from a channel's candidate list scores 0 in that channel,
    which is what 8.4 prescribes.
    """
    weights = weights or [1.0] * len(rankings)
    totals: dict[str, float] = defaultdict(float)
    registry: dict[str, dict] = {}

    for ranking, weight in zip(rankings, weights):
        if weight <= 0 or not ranking:
            continue
        values = [s for _m, s in ranking]
        low, high = min(values), max(values)
        span = (high - low) or 1.0
        for meta, score in ranking:
            key = meta["chunk_id"]
            totals[key] += weight * (score - low) / span
            registry[key] = meta

    ordered = sorted(totals.items(), key=lambda kv: -kv[1])
    return [(registry[key], score) for key, score in ordered]


# =====================================================================
# post-filters (8.7)
# =====================================================================

def expected_phenomenon(query_id: str) -> int:
    """q001-q016 -> 1, q017-q032 -> 2, q033-q050 -> 3."""
    match = re.search(r"(\d+)", query_id or "")
    if not match:
        return 0
    n = int(match.group(1))
    return 1 if n <= 16 else 2 if n <= 32 else 3


def apply_phenomenon_boost(candidates: list[tuple[dict, float]], phenomenon: int,
                           boost: float, mode: str = "multiply"
                           ) -> list[tuple[dict, float]]:
    """
    Reward chunks whose `fenomeno` matches the query-id range, then re-sort.

    A bonus rather than a filter: the id->phenomenon mapping is inferred from
    the question extract, not published, so if it is wrong somewhere this
    costs a few rank positions instead of deleting the answer from the pool.

    MODE MATTERS, and getting it wrong is invisible. In RRF space scores sit
    around 0.016 and multiplying by 1.08 is worth roughly five rank
    positions -- a nudge. In cosine space the top scores sit around 0.88 and
    the same 8% multiplier adds ~0.07, which reorders the whole head of the
    list. Multiplying is therefore not portable between the two.

    mode="add" IS A FRACTION OF THE POOL'S SCORE SPAN, NOT AN ABSOLUTE.
        This was an absolute number once, and it silently did nothing. The
        reasoning behind 0.03 was "cosine runs 0.78-0.88, so this is 30% of
        the range" -- but that is the range of the TOP of the list, not of
        the 1000-deep pool actually being reordered, whose span is nearer
        1.0. Worse, switching --doc-score to combsum with three channels
        stretches the span to ~3.0, so the same constant became 1% of the
        range and the phenomenon filter stopped existing. The visible
        symptom was Fenomeno 3 documents taking every document slot on
        Fenomeno 1 queries.

        Scaling by the observed span makes one number mean the same thing in
        cosine space, in CombSUM space, and in whatever space comes next.
    """
    if not phenomenon or boost <= 0:
        return candidates

    if mode == "add":
        values = [s for _m, s in candidates]
        span = (max(values) - min(values)) if values else 0.0
        bonus = boost * span
        rescored = [(meta, score + (bonus if meta.get("fenomeno") == phenomenon else 0.0))
                    for meta, score in candidates]
    else:
        factor = 1.0 + boost
        rescored = [(meta, score * (factor if meta.get("fenomeno") == phenomenon else 1.0))
                    for meta, score in candidates]
    return sorted(rescored, key=lambda pair: -pair[1])


def deduplicate(candidates: list[tuple[dict, float]], threshold: float,
                n: int = 8, window: int = 200) -> list[tuple[dict, float]]:
    """
    Drop a candidate whose text substantially repeats one already kept.

    Chunks overlap by ~350 characters by design, so when a passage is
    relevant both the chunk containing it and its neighbour score highly. The
    second adds nothing a grader can reward -- same sentences at a different
    offset -- while occupying one of only ten slots. NDCG@10 discounts by
    log2(i+1), so a wasted slot at rank 2 costs far more than one at rank 9,
    and dropping a duplicate promotes everything below it.

    THRESHOLD CALIBRATION -- THERE ARE TWO REGIMES AND ONE NUMBER.
    Containment is measured against the SHORTER of the two shingle sets.
    Measured on this corpus's chunk geometry:

        adjacent windows (1000 chars, 350 overlap)  containment ~0.31
        repeated boilerplate / re-published text     containment ~0.95

    The old default of 0.6 sits between them, so it caught the boilerplate
    and never once caught the overlapping neighbour the docstring claimed it
    was for. That is not necessarily wrong -- an adjacent window still
    carries 65% new text, and the real grader may award it partial
    relevance -- but it means the knob was never doing the job it was
    documented as doing, and nobody had checked.

    0.45 is the default here: still safely above the 0.31 neighbour case,
    with more margin against lightly-edited republications. Going below 0.31
    turns on neighbour suppression, which is a real behaviour change and
    worth an actual sweep: {0.28, 0.35, 0.45, 0.60} against nDCG@10. Watch
    the metric, not the suppression count -- a big count only proves the
    filter is firing, not that it is helping.

    Do NOT "fix" this by lowering overlap_chars in the chunker: the overlap is
    what aligns fragment boundaries with the organisers' ~708-character step.

    `window` caps the work: this is O(kept^2) in set intersections and the
    fused pool runs to a few thousand entries. The tail is appended untouched
    so the fragment fallback still has a full list to draw on.
    """
    if threshold <= 0:
        return candidates

    head, tail = candidates[:window], candidates[window:]
    kept: list[tuple[dict, float]] = []
    kept_shingles: list[set] = []

    for meta, score in head:
        mine = shingles(meta.get("texto", ""), n)
        if not mine:
            continue
        duplicate = any(
            theirs and len(mine & theirs) / min(len(mine), len(theirs)) >= threshold
            for theirs in kept_shingles)
        if not duplicate:
            kept.append((meta, score))
            kept_shingles.append(mine)

    return kept + tail


def rerank(candidates: list[tuple[dict, float]], query: str, model_name: str,
           depth: int, blend: float = 0.5, use_context: bool = True,
           _cache: dict = {}) -> list[tuple[dict, float]]:
    """
    Optional cross-encoder rerank of the top `depth` fragments.

    COMPLIANCE: ASKED AND ANSWERED. The organisers were asked this directly,
    by three separate teams, and confirmed it in the FAQ:

        "Si esta permitido re ranking con cross-encoders. La restricción
         aplica es para arquitecturas decoders."
        "Un cross encoder si es permitido. La restricción aplica para
         arquitecturas tipo decoder."

    BAAI/bge-reranker-v2-m3 is XLM-RoBERTa-large with a scalar relevance
    head: encoder-only, non-autoregressive, generating nothing. This was ON
    BY DEFAULT only after that ruling; before it, the narrow reading of 8.3
    ("vectores, puntuaciones de similitud y metadata") made it a judgement
    call and the safe default was off. Still name the architecture in
    informe_tecnico.pdf -- a declared reranker reads better than one a judge
    discovers.

    Practically, this is the single largest available gain on nDCG@10: dense
    retrieval that puts the right chunk at rank 15-40 is exactly what a
    cross-encoder repairs.

    BLEND, AND WHY IT IS NOT 1.0.
        Replacing the order outright throws away the retrievers' agreement.
        Measured on this corpus: reranking lifted three ground-truth
        fragments from ranks 87, 43 and 103 into the top 10 -- and pushed a
        fragment that two encoders and BM25 had ALL put at rank 1 down to 31,
        turning the one query scoring nDCG@10 = 1.000 into a zero. Net effect
        on the mean was slightly negative despite three clear wins.

        A cross-encoder is better at fine discrimination inside a shortlist
        and worse at the coarse judgement the retrievers already made
        together. So fuse the two orderings by RRF rather than substituting:
        blend=1.0 is the cross-encoder alone, 0.0 disables it, 0.5 gives each
        an equal vote. Sweep {1.0, 0.7, 0.5, 0.3} -- and on seven queries,
        prefer the value that helps several rather than the one that wins.
    """
    if not model_name or not candidates:
        return candidates

    if model_name not in _cache:
        from sentence_transformers import CrossEncoder
        _cache[model_name] = CrossEncoder(model_name, max_length=512)
    model = _cache[model_name]

    head, tail = candidates[:depth], candidates[depth:]
    # The heading trail goes to the cross-encoder too. The dense channel
    # embeds `contexto + texto`, so scoring the reranker on `texto` alone
    # judges the passage stripped of the one signal that says what it is
    # about: "estas afectaciones se han dado cuando..." reads very differently
    # with "4.5 Paz ambiental" in front of it. Truncation is the encoder's
    # job -- max_length=512 cuts the tail, and the context is worth more per
    # token than the last sentence it displaces.
    pairs = [(query, (f"{meta['contexto']}. {meta.get('texto', '')}"
                      if use_context and meta.get("contexto")
                      else meta.get("texto", "")))
             for meta, _ in head]
    scores = model.predict(pairs, show_progress_bar=False)

    if blend >= 1.0:
        reordered = sorted(zip(head, scores), key=lambda pair: -float(pair[1]))
        return [(meta, float(score)) for (meta, _old), score in reordered] + tail

    # RRF between the retrievers' order (0..n-1, already sorted) and the
    # cross-encoder's order, so a unanimous rank-1 needs a strong contrary
    # signal to be displaced rather than a marginally higher logit.
    by_cross = sorted(range(len(head)), key=lambda i: -float(scores[i]))
    points = {i: (1.0 - blend) / (RERANK_RRF_K + rank)
              for rank, i in enumerate(range(len(head)), 1)}
    for rank, i in enumerate(by_cross, 1):
        points[i] += blend / (RERANK_RRF_K + rank)

    order = sorted(points.items(), key=lambda kv: -kv[1])
    return [(head[i][0], score) for i, score in order] + tail



# =====================================================================
# query expansion and diversification
# =====================================================================

def expand_query_rm3(lexical: "LexicalIndex", query: str,
                     feedback_docs: int = 10, terms: int = 10,
                     original_weight: float = 0.6) -> str:
    """
    RM3 pseudo-relevance feedback: re-run the lexical channel with terms
    borrowed from its own top results.

    LEGAL, AND WORTH CHECKING THAT CAREFULLY. Section 8.3 forbids
    "reformulación o expansión de la consulta mediante un DECODER". RM3 uses
    no model at all -- it counts terms in the documents the first pass
    returned and adds the most discriminative ones back. It is 2001-era term
    statistics, the same family as BM25 itself, which the organisers have
    confirmed twice is permitted.

    WHY IT SHOULD HELP HERE. The queries are one-sentence Spanish
    abstractions ("¿Cómo utilizan los grupos armados ilegales el control
    territorial...?") and the documents are full of the specific vocabulary
    that actually distinguishes them: municipality names, "economías
    ilícitas", "extorsión", "corredores estratégicos". The query cannot match
    what it does not mention. Feedback puts the corpus's own words into the
    query.

    THE RISK IS DRIFT. If the first pass is wrong, expansion amplifies the
    error -- that is why `original_weight` keeps the user's terms dominant
    and why the expansion is repeated, not averaged: repeating a term is how
    a bag-of-words query expresses weight.
    """
    top = lexical.search(query, feedback_docs)
    if not top:
        return query

    counts: Counter = Counter()
    for meta, _score in top:
        text = meta.get("texto", "")
        context = meta.get("contexto", "")
        counts.update(set(tokenize(f"{context} {text}" if context else text)))

    original = set(tokenize(query))
    # Weight by how concentrated a term is in the feedback set relative to
    # the corpus: idf is what stops "gobierno" and "territorio" winning.
    scored = [(counts[t] * lexical.idf.get(t, 0.0) or counts[t], t)
              for t in counts if t not in original]
    if not scored:
        return query
    for term in [t for _w, t in sorted(scored, reverse=True)[:terms]]:
        lexical.idf.setdefault(term, 0.0)

    picked = [t for _w, t in sorted(scored, reverse=True)[:terms]]
    repeats = max(1, int(round((1 - original_weight) / max(original_weight, 0.01) * 2)))
    return query + " " + " ".join(picked * repeats)


def diversify_mmr(candidates: list[tuple[dict, float]], lam: float = 0.7,
                  n: int = 10, window: int = 60) -> list[tuple[dict, float]]:
    """
    Maximal Marginal Relevance over the top candidates.

    WHY THIS AND NOT JUST DEDUPE. deduplicate() is a threshold: a fragment is
    either a repeat or it is not. MMR is continuous -- it trades relevance
    against novelty at every step, so the second slot goes to the best
    fragment that adds something, not merely to one that clears a similarity
    bar. nDCG@10 rewards covering SEVERAL relevant passages, and a query with
    three annotated fragments cannot score well if all ten slots paraphrase
    the first.

    lam=1.0 is the unmodified ranking; lower values buy coverage with
    relevance. Applied only to the head, so the fallback list stays intact.
    """
    if lam >= 1.0 or not candidates:
        return candidates

    head, tail = candidates[:window], candidates[window:]
    shingle_cache = [shingles(m.get("texto", ""), 8) for m, _s in head]
    scores = [s for _m, s in head]
    high = max(scores) or 1.0
    low = min(scores)
    span = (high - low) or 1.0

    chosen: list[int] = []
    remaining = set(range(len(head)))
    while remaining and len(chosen) < n:
        best_index, best_value = None, -1e9
        for i in sorted(remaining):
            relevance = (scores[i] - low) / span
            mine = shingle_cache[i]
            novelty = 0.0
            for j in chosen:
                theirs = shingle_cache[j]
                if mine and theirs:
                    novelty = max(novelty, len(mine & theirs)
                                  / min(len(mine), len(theirs)))
            value = lam * relevance - (1 - lam) * novelty
            if value > best_value:
                best_index, best_value = i, value
        chosen.append(best_index)
        remaining.discard(best_index)

    ordered = [head[i] for i in chosen]
    ordered += [head[i] for i in range(len(head)) if i not in chosen]
    return ordered + tail


# =====================================================================
# output assembly
# =====================================================================

def aggregate_documents(ranking: list[tuple[dict, float]], n: int | None = 3,
                        pool: int = 30, hit_bonus: float = 0.02,
                        hit_cap: int = 3, mode: str = "max",
                        top_m: int = 3) -> list[str]:
    """
    Max pooling plus a bonus for repeated evidence (8.6). Pure arithmetic
    over scores, no generative step.

    MUST be fed a score list with real spread -- cosine or normalised
    CombSUM, never RRF. See the module docstring.

    CALIBRATE hit_bonus AGAINST THE SCORE SPREAD. Cosine similarity on this
    corpus runs about 0.78-0.88, an 11% spread. The previous 0.05 per extra
    chunk, capped at 5, was worth up to 25% -- more than twice the entire
    range of the signal it multiplied. Document ranking was therefore decided
    almost entirely by which document had the most chunks in the pool, which
    systematically favours 1000-page reports over the short document that
    actually answers the question. 0.02 capped at 3 (max 6%) leaves the
    repetition bonus as a tie-breaker, which is what 8.6 describes.

    `pool` bounds how many chunks are aggregated, because 8.6 aggregates the
    k_chunk MOST relevant fragments, not the whole candidate list.

    Deliberately runs on the PRE-dedupe list: at document level, several
    overlapping windows of one passage genuinely are weaker evidence than
    several distinct passages, and the hit bonus is what expresses that.

    n=None returns the FULL ordered list instead of the top n. F1@3 is a
    cliff: a document at rank 4 and a document at rank 400 both score zero,
    so on a seven-query sample almost every configuration change is
    invisible. evaluar.py uses n=None to report where the ground-truth
    document actually landed, which moves continuously and can therefore be
    tuned against.
    """
    per_doc: dict[str, list[float]] = defaultdict(list)
    for meta, score in ranking[:pool]:
        per_doc[meta["doc_id"]].append(score)

    if mode == "max":
        # MAX POOLING MAKES --doc-pool A NO-OP, WHICH IS NOT OBVIOUS.
        # Enlarging the pool can only admit documents whose best chunk is
        # deeper and therefore scores lower, so they can never displace the
        # incumbent top 3. Measured: doc_pool 30 / 60 / 150 / 400 returned
        # byte-identical F1@3. Only the hit bonus can reorder anything, and it
        # is capped at 6%.
        #
        # The consequence is that document ranking is decided by each
        # document's SINGLE best chunk. A report that is broadly relevant --
        # eight good chunks at ranks 40-90 -- loses to one with a single
        # lucky chunk at rank 2. That is the wrong prior for 10.2.2, which
        # asks which DOCUMENTS answer the question.
        base = {d: max(v) for d, v in per_doc.items()}
    elif mode == "sum":
        # Sum of the document's top-m chunks. Rewards breadth: several good
        # passages beat one great one. This is the mode that makes --doc-pool
        # matter, because a document's chunks at ranks 40-90 now contribute.
        base = {d: sum(sorted(v, reverse=True)[:top_m]) for d, v in per_doc.items()}
    elif mode == "mean":
        # Mean of the top-m, padded with zeros when fewer than m exist, so a
        # document with one chunk is not rewarded for having no weak ones.
        base = {d: sum(sorted(v, reverse=True)[:top_m]) / top_m
                for d, v in per_doc.items()}
    else:                                    # rrf
        # Rank-based within the pool: robust when scores are poorly
        # calibrated, and naturally diminishing so one document cannot win on
        # volume alone.
        position = {}
        for rank, (meta, _s) in enumerate(ranking[:pool], 1):
            position.setdefault(meta["doc_id"], []).append(rank)
        base = {d: sum(1.0 / (RRF_K + r) for r in ranks[:top_m])
                for d, ranks in position.items()}

    hits = {d: len(v) for d, v in per_doc.items()}
    aggregated = {d: base[d] * (1.0 + hit_bonus * min(hits[d] - 1, hit_cap))
                  for d in base}
    ordered = [d for d, _ in sorted(aggregated.items(), key=lambda kv: -kv[1])]

    if n is None:                       # full ranking, for diagnostics
        return ordered

    # 9.3.2: exactly 3 documents or the line is discarded. If the pool held
    # fewer than 3 distinct documents, widen it rather than ship a short list.
    if len(ordered) < n:
        for meta, _score in ranking:
            if meta["doc_id"] not in ordered:
                ordered.append(meta["doc_id"])
            if len(ordered) >= n:
                break
    return ordered[:n]


def aggregate_documents_rankdecay(ranking: list[tuple[dict, float]],
                                  n: int | None = 3, pool: int = 30,
                                  decay: float = 0.85,
                                  rrf_k: int = RRF_K) -> list[str]:
    """
    FIX (replaces the cosine/combsum default -- see RetrievalConfig.doc_score).

    Aggregates documents from the FUSED FRAGMENT RANKING directly -- the
    same phenomenon-boosted, reranked candidate list `build_fragments()`
    draws the ten returned fragments from -- instead of a separately
    computed cosine or CombSUM ranking. A document's score is the sum, over
    its chunks in this list, of that chunk's reciprocal rank discounted by
    how many of the document's OWN chunks came before it:

        score(d) = sum_i  1/(rrf_k + rank_i + 1) * decay**i

    where `rank_i` is the chunk's position (0-based) in `ranking` and `i` is
    its position among the document's own chunks, sorted best-first.

    WHY THIS FIXES THE OLD DEFAULT'S THREE PROBLEMS AT ONCE.
      1. No arbitrary encoder. `ranking` is the fused, multi-channel list
         every fragment comes from -- BM25, the graph and every dense
         encoder already voted on it via fuse_rrf(). There is no
         "stores[0]" or --doc-encoder to get wrong.
      2. --doc-pool is no longer a no-op. Because the score is a SUM with
         decay, not a max, a document that is broadly relevant (several
         good chunks at ranks 40-90) can still outscore one with a single
         lucky chunk at rank 2 -- the old max-pooling default could never
         do this regardless of --doc-pool.
      3. One score space, not two. Fragments and documents are now the same
         underlying ranking with two different aggregations, rather than
         two independently-tuned pipelines that can (and did) disagree
         about which documents matter.

    `decay` trades breadth against a single strong hit: 1.0 makes every
    chunk count equally (closest to the old doc_agg="sum"); lower values
    concentrate credit on a document's best 2-3 chunks. 0.85 is a starting
    point, not a measured optimum -- sweep it against F1@3 like any other
    knob here.
    """
    per_doc: dict[str, list[int]] = defaultdict(list)
    for rank, (meta, _score) in enumerate(ranking[:pool]):
        per_doc[meta["doc_id"]].append(rank)

    scores: dict[str, float] = {}
    for doc_id, ranks in per_doc.items():
        scores[doc_id] = sum(
            (1.0 / (rrf_k + rank + 1)) * (decay ** i)
            for i, rank in enumerate(sorted(ranks)))

    ordered = [d for d, _ in sorted(scores.items(), key=lambda kv: -kv[1])]

    if n is None:                       # full ranking, for diagnostics
        return ordered

    # 9.3.2: exactly 3 documents or the line is discarded.
    if len(ordered) < n:
        for meta, _score in ranking:
            if meta["doc_id"] not in ordered:
                ordered.append(meta["doc_id"])
            if len(ordered) >= n:
                break
    return ordered[:n]


def build_fragments(candidates: list[tuple[dict, float]], n: int = 10) -> list[dict]:
    """
    Top-n fragments. 9.2.1: a chunk over 250 words is split into complete
    sub-fragments that keep the original chunk_id and each take their own
    rank.
    """
    fragments: list[dict] = []
    for meta, _score in candidates:
        for piece in split_to_250_words(meta["texto"], 250):
            piece = sanitize_text(piece)
            if not piece.strip():
                continue
            fragments.append({"rank": len(fragments) + 1,
                              "chunk_id": meta["chunk_id"],
                              "doc_id": meta["doc_id"],
                              "text": piece})
            if len(fragments) == n:
                return fragments
    return fragments


# =====================================================================
# the single ranking path
# =====================================================================

@dataclass
class Retrieved:
    candidates: list[tuple[dict, float]]   # fused pool, RRF space. Fallback.
    unique: list[tuple[dict, float]]       # the same, deduped. FRAGMENTS.
    doc_ranking: list[tuple[dict, float]]  # cosine or CombSUM. DOCUMENTS.
    channels: list[tuple[str, list[tuple[dict, float]]]]  # per-channel, for eval


def retrieve(stores: list[VectorStore], query: str, query_id: str = "",
             cfg: RetrievalConfig | None = None,
             lexical: LexicalIndex | None = None,
             graph: "GraphIndex | None" = None) -> Retrieved:
    """
    Query -> Retrieved. THE ONLY ranking path in this project.

    main() calls it and so does evaluar.py. That is not tidiness: the
    post-filters were added to main()'s loop first, and evaluar.py went on
    ranking with a bare `rankings[0]` for a while afterwards. Every number it
    printed described a pipeline that no longer existed. Measuring a
    different system than you ship is worse than not measuring, because it
    looks like data.
    """
    cfg = cfg or RetrievalConfig()

    channels: list[tuple[str, list[tuple[dict, float]]]] = [
        (store.name, store.search(query, cfg.depth)) for store in stores]
    if lexical is not None and cfg.bm25_weight > 0:
        # Expansion applies to the LEXICAL channel only. The dense encoders
        # were trained on natural questions and a query padded with repeated
        # keywords is off-distribution for them; BM25 is a bag of words and
        # has no distribution to leave.
        lexical_query = query
        if cfg.rm3_terms > 0:
            lexical_query = expand_query_rm3(
                lexical, query, cfg.rm3_feedback, cfg.rm3_terms,
                cfg.rm3_original_weight)
        channels.append(("bm25", lexical.search(lexical_query, cfg.depth)))

    if graph is not None and cfg.graph_weight > 0:
        by_chunk = {m["chunk_id"]: m for m in stores[0].metadata}
        hits = graph.search(query, cfg.depth, by_chunk, cfg.graph_neighbour)
        if hits:
            channels.append(("grafo", hits))

    if cfg.min_score > 0:
        # keep the unfiltered ranking if the filter would empty it
        channels = [(name, [(m, s) for m, s in r if s >= cfg.min_score] or r)
                    for name, r in channels]

    weights = [1.0] * len(stores)
    if lexical is not None and cfg.bm25_weight > 0:
        weights.append(cfg.bm25_weight)
    if any(name == "grafo" for name, _r in channels):
        weights.append(cfg.graph_weight)

    rankings = [r for _n, r in channels]
    phenomenon = expected_phenomenon(query_id)

    # ---- fragments: rank space, robust to incomparable score scales (8.4)
    candidates = fuse_rrf(rankings, weights) if len(rankings) > 1 else rankings[0]
    candidates = apply_phenomenon_boost(
        candidates, phenomenon, cfg.phenomenon_boost, mode=cfg.phenomenon_mode)

    if cfg.reranker:
        candidates = rerank(candidates, query, cfg.reranker,
                            cfg.rerank_depth, cfg.rerank_blend,
                            cfg.rerank_context)

    # ---- documents (8.6)
    # FIX: "rankdecay" (default) reuses THIS SAME fused, phenomenon-boosted,
    # reranked list -- every channel already voted on it -- instead of a
    # second, independently-computed cosine/CombSUM ranking that depended on
    # which encoder happened to load first. See aggregate_documents_rankdecay
    # and RetrievalConfig.doc_score for the full reasoning. "combsum"/
    # "cosine" recompute the old separate ranking for ablation.
    if cfg.doc_score == "rankdecay":
        doc_ranking = candidates
    elif cfg.doc_score == "cosine":
        doc_ranking = apply_phenomenon_boost(
            rankings[0], phenomenon, cfg.phenomenon_boost_doc, mode="add")
    else:
        doc_ranking = apply_phenomenon_boost(
            fuse_combsum(rankings, weights), phenomenon,
            cfg.phenomenon_boost_doc, mode="add")

    unique = deduplicate(candidates, cfg.dedupe_threshold,
                         window=cfg.dedupe_window)
    if cfg.mmr_lambda < 1.0:
        unique = diversify_mmr(unique, cfg.mmr_lambda)
    return Retrieved(candidates, unique, doc_ranking, channels)


# =====================================================================
# queries, validation, entry point
# =====================================================================

def read_queries(path: Path) -> list[tuple[str, str]]:
    """Accepts .jsonl ({query_id, query|text}) or .json (dict or list)."""
    if path.suffix == ".jsonl":
        objects = read_jsonl(path)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        objects = ([{"query_id": k, "query": v} for k, v in data.items()]
                   if isinstance(data, dict) else data)

    pairs = []
    for obj in objects:
        query_id = obj.get("query_id") or obj.get("id")
        text = (obj.get("query") or obj.get("text")
                or obj.get("consulta") or obj.get("pregunta"))
        if query_id and text:
            pairs.append((query_id, text))
    return sorted(pairs, key=lambda pair: pair[0])       # q001..q050 (10.3)


def validate(path: Path) -> bool:
    """Strict schema check (9.3.2). Failing here means failing the submission."""
    # split("\n"), not splitlines(): a fragment carrying U+2028 would
    # otherwise be miscounted as two lines and reported as a schema failure
    # that does not exist.
    raw = path.read_text(encoding="utf-8")
    lines = [l for l in raw.split("\n") if l.strip()]
    errors = []

    stray = len(_LINE_SEPARATORS.findall(raw))
    if stray:
        errors.append(f"{stray} raw U+0085/U+2028/U+2029 in the output. A "
                      f"grader using splitlines() will see a malformed file "
                      f"(9.3.2 discards those). Fragment text is not sanitized.")

    if len(lines) != 50:
        errors.append(f"expected 50 lines, found {len(lines)}")

    previous = ""
    for n, line in enumerate(lines, 1):
        obj = json.loads(line)
        query_id = obj.get("query_id", "")
        if not query_id:
            errors.append(f"line {n}: missing query_id")
        if query_id <= previous:
            errors.append(f"line {n}: {query_id} out of order (10.3 wants q001..q050)")
        previous = query_id

        if len(obj.get("documents", [])) != 3:
            errors.append(f"line {n}: documents != 3")
        if len({d["doc_id"] for d in obj.get("documents", [])}) != len(obj.get("documents", [])):
            errors.append(f"line {n}: duplicate doc_id in documents")
        if len(obj.get("fragments", [])) != 10:
            errors.append(f"line {n}: fragments != 10")
        for fragment in obj.get("fragments", []):
            for key in ("rank", "chunk_id", "doc_id", "text"):
                if key not in fragment:
                    errors.append(f"line {n}: fragment missing {key}")
            if len(fragment.get("text", "").split()) > 250:
                errors.append(f"line {n} rank {fragment.get('rank')}: over 250 words")

    print("VALIDATION: OK" if not errors else "VALIDATION:\n  " + "\n  ".join(errors))
    return not errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CODEFEST AD ASTRA 2026 - retrieval and resultados.jsonl")
    add_retrieval_args(parser)
    parser.add_argument("--queries", type=Path, default=Path("consultas_50.jsonl"),
                        help="Ships alongside this script so the jury can "
                             "reproduce resultados.jsonl (1.4).")
    parser.add_argument("--out", type=Path, default=Path("resultados.jsonl"))
    args = parser.parse_args()
    cfg = config_from_args(args)

    index_dir = resolve_input(args.index_dir)
    queries_path = resolve_input(args.queries)
    if args.out == Path("resultados.jsonl") and not Path.cwd().samefile(SCRIPT_DIR):
        args.out = SCRIPT_DIR / args.out      # write next to the index, not the cwd

    stores = load_stores(index_dir, args.doc_encoder, cfg.doc_score)
    queries = read_queries(queries_path)
    print(f"Encoders : {[s.name for s in stores]}")
    print(f"Index    : {index_dir.resolve()}")
    print(f"Queries  : {len(queries)} from {queries_path}")
    print(f"Output   : {args.out.resolve()}")
    print(f"Documents: {cfg.doc_score} space, pool={cfg.doc_pool}, "
          f"hit bonus +{cfg.doc_hit_bonus:.0%} x{cfg.doc_hit_cap}")
    print(f"Fragments: RRF k={RRF_K}, dedupe>={cfg.dedupe_threshold}, "
          f"phenomenon {cfg.phenomenon_mode} {cfg.phenomenon_boost}"
          + (f", reranker={cfg.reranker}" if cfg.reranker else ""))

    lexical = None
    if cfg.bm25_weight > 0:
        print(f"Lexical  : building BM25 over {len(stores[0].metadata)} chunks ...")
        lexical = LexicalIndex.build(stores[0].metadata, [q for _i, q in queries],
                                     cfg.bm25_k1, cfg.bm25_b)
        print(f"           {len(lexical.postings)} query terms, "
              f"avgdl={lexical.avgdl:.0f}")

    graph = None
    if cfg.graph_weight > 0:
        graph = GraphIndex.load(index_dir / "grafo" / "grafo.graphml")
        if graph is not None:
            print(f"Grafo    : {len(graph.entities)} entities, "
                  f"{sum(len(v) for v in graph.neighbours.values()) // 2} "
                  f"relations (bonus, section 8.5)")
        else:
            print(f"Grafo    : none at {index_dir / 'grafo' / 'grafo.graphml'} "
                  f"-- bonus component not built")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dropped_total = 0

    with args.out.open("w", encoding="utf-8") as fh:
        for query_id, text in queries:
            result = retrieve(stores, text, query_id, cfg, lexical, graph)
            dropped_total += len(result.candidates) - len(result.unique)

            fragments = build_fragments(result.unique, 10)
            if len(fragments) < 10:
                # Never ship a short list: 9.3.2 discards the line.
                fragments = build_fragments(result.candidates, 10)

            documents = (
                aggregate_documents_rankdecay(
                    result.doc_ranking, 3, cfg.doc_pool, cfg.doc_decay)
                if cfg.doc_score == "rankdecay" else
                aggregate_documents(
                    result.doc_ranking, 3, cfg.doc_pool,
                    cfg.doc_hit_bonus, cfg.doc_hit_cap,
                    cfg.doc_agg, cfg.doc_top_m))

            fh.write(json.dumps({
                "query_id": query_id,
                "documents": [{"rank": i + 1, "doc_id": d}
                              for i, d in enumerate(documents)],
                "fragments": fragments,
            }, ensure_ascii=False) + "\n")
            print(f"  {query_id} ok")

    print(f"\n  {dropped_total} near-duplicate candidates suppressed in total")
    validate(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()