"""build_kg_v2_fixed.py

A safer replacement for build_kg_v2.py.

Design goals
------------
1. Keep chapter-local canonical KPs only as extraction hints.
2. Use the full-book Global Concept Registry as the identity source.
3. Emit only stable OS_* concept IDs as formal graph nodes.
4. Emit unmapped chapter KPs as NEW_CONCEPT_CANDIDATE records, never as nodes.
5. Exclude Summary / Review / Exercises / Bibliography content by default.
6. Enforce a real maximum chunk length after section-aware chunking.
7. Validate every quote against the source chunk.
8. Use a constrained relation ontology with explicit direction rules.
9. Do not create SAME_TYPE_AS or SIBLING_IN_SECTION graph noise.
10. Optionally verify each extracted relation in a second LLM pass.
11. Aggregate multiple evidence items instead of keeping only the first edge.
12. Produce alignment, quality, and checkpoint files for auditability.

Expected project modules (same as the original pipeline)
---------------------------------------------------------
- config.settings
- utils.gemini_client.init_gemini / generate_with_rotation
- kg_builder.kp_loader_v2.load_kp_v2
- kg_builder.section_chunker.chunk_pdf_by_sections

Example
-------
python build_kg_v2_fixed.py \
  --pdf BOOK/002_Chapter_2_Operating-System_Structures.pdf \
  --kp data/knowledge_points_v2/CH02/CH02_canonical_kps.json \
  --global-registry BOOK/OSC10E_CH02_CH21_Merged_Global_Concept_Registry.json \
  --output outputs/kg_v2/ch02.json \
  --chunk-size 2000 --chunk-overlap 300 --max-chunk-chars 2400 \
  --workers 3
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from config import settings
from utils.gemini_client import init_gemini, generate_with_rotation
from kg_builder.kp_loader_v2 import load_kp_v2, KP, KPRegistry
from kg_builder.section_chunker import chunk_pdf_by_sections
from kg_builder.graph_schema import (
    FORMAL_RELATION_TYPES,
    SYMMETRIC_RELATION_TYPES,
    VERIFIED_RELATION_STATUSES,
)


# v3.2 is the import contract consumed by ingest_global_registry_kg.py.
# A build may call itself successful only when every formal node, mention and
# relation satisfies that contract.  Lower-confidence relations remain in the
# separate review_only_relations collection and are never promoted here.
PIPELINE_VERSION = "kp-first-global-v3.6-import-contract-preflight-v1"
# Changing this value invalidates resumable extraction checkpoints.  It is
# deliberately part of the fingerprint because it changes which model output
# can become formal graph evidence.
# Direct-prerequisite construction happens after chunk extraction and only
# consumes already validated formal evidence.  Keep the extraction fingerprint
# stable so an existing chapter checkpoint can be reused; this avoids paying
# to repeat unrelated evidence/relation calls when only this derived layer is
# added.
EXTRACTION_POLICY_VERSION = "candidate-scope-anchored-evidence-and-reviewed-relations-v4"
MIN_EVIDENCE_CONFIDENCE = {
    "definition": 0.75,
    "example": 0.70,
    "usage": 0.70,
    "mention": 0.75,
}
MIN_DEFINITION_OVERLAP = 0.18
ALLOWED_EVIDENCE_TYPES = {"definition", "example", "usage", "mention"}
ALLOWED_RELATIONS = set(FORMAL_RELATION_TYPES)
SYMMETRIC_RELATIONS = set(SYMMETRIC_RELATION_TYPES)
DIRECTIONAL_RELATIONS = ALLOWED_RELATIONS - SYMMETRIC_RELATIONS

# These gates intentionally favour precision over relation count.  A relation
# that is only plausible is preserved in review_only_relations, never injected
# into the formal graph or a teaching path.
_CAUSAL_CUE_RE = re.compile(
    r"\b(?:cause(?:s|d|ing)?|lead(?:s|ing)?\s+to|result(?:s|ed|ing)?\s+in|"
    r"therefore|thus|consequently|because\s+of)\b",
    re.I,
)
_MANAGEMENT_CUE_RE = re.compile(r"\b(?:manage(?:s|d|ment)?|administer(?:s|ed|ing)?)\b", re.I)
_PROVISION_CUE_RE = re.compile(r"\b(?:provide(?:s|d|ing)?|offer(?:s|ed|ing)?|return(?:s|ed|ing)?)\b", re.I)
_ENABLEMENT_CUE_RE = re.compile(r"\b(?:enable(?:s|d|ing)?|allow(?:s|ed|ing)?|permit(?:s|ted|ting)?)\b", re.I)
_STATE_TRANSITION_CUE_RE = re.compile(
    r"\b(?:transition(?:s|ed|ing)?|switch(?:es|ed|ing)?|move(?:s|d|ing)?|"
    r"change(?:s|d|ing)?\s+(?:from|to))\b",
    re.I,
)
_REDUCTION_CUE_RE = re.compile(r"\b(?:reduc(?:e|es|ed|ing)|decreas(?:e|es|ed|ing)|lower(?:s|ed|ing)?)\b", re.I)
_INCREASE_CUE_RE = re.compile(r"\b(?:increas(?:e|es|ed|ing)|rais(?:e|es|ed|ing)|higher)\b", re.I)

RELATION_SAFETY_CONTRACT = """# Final relation safety contract (overrides earlier examples)
- Do NOT output PREREQUISITE_OF. Textual order, execution order, and a condition
  for an operation are not pedagogical prerequisites.
- HAS_SUBTYPE is allowed only when the source is explicitly a category and the
  target is explicitly a member/type of that category. Never use it for an
  analogy, an implementation, a file-like representation, or a shared name.
- CAUSES requires the source quote itself to state a causal result; an aim,
  motivation, resource, object, or co-occurrence is not a causal effect.
- Use the specific relations REPRESENTS, CONTAINS, CREATES, TRANSITIONS_TO,
  SAVES_TO, LOADS_FROM, RECLAIMS, WAITS_ON, SCHEDULES, REDUCES, or INCREASES when they fit. Do not
  use MANAGES or CAUSES as a catch-all.
- If a relation is not directly and unambiguously supported by the source
  chunk, omit it. Precision is more important than the number of edges.
- A numbered figure, table, or worked case is source evidence.  Use a precise
  relation such as TRANSITIONS_TO, REPRESENTS, or CONTAINS only when the
  source text directly supports it; never invent a node merely because a
  figure probably exists.
- Optional schema_gap_candidates may identify a major teachable concept or
  a figure/table/API reference that is absent from the supplied registry.
  Each item must have proposed_name, category, quote, and rationale. Use
  category=concept only for a teachable concept.  Use figure/table/diagram
  for an asset and api/library/interface/system call for a technical-name
  review item.  They are review items only and never become formal nodes.
"""

RELATION_DECISION_CONTRACT = """# Relation decision rules for the verification pass
- Judge direction from the quoted sentence, not from the order of the two IDs.
- A TRANSITIONS_TO edge is valid only if the quote explicitly describes a
  transition between the two named state Concepts.  Do not infer a state edge
  merely because a variable is called "new state".
- A communication link may enable *interprocess communication*; it does not
  automatically enable a Process node.  Drop or retype such a mismatch.
- A system call that returns or obtains a status does not CAUSE the status.
  Use PROVIDES only when the sentence actually says it returns/provides that
  value; otherwise DROP the edge.
- If either endpoint is only implied, or the relation could be reversed,
  choose DROP rather than guessing.  The formal graph prioritises precision.
- Use REDUCES or INCREASES only when the quoted text explicitly expresses that
  direction of effect.  Do not use the generic CAUSES relation for a decrease
  or increase.
"""

# Headings that should not independently create graph evidence.
EXCLUDED_HEADING_RE = re.compile(
    r"(?im)^\s*(?:\d+(?:\.\d+)*\s+)?(?:"
    r"chapter\s+summary|summary|practice\s+exercises?|exercises?|"
    r"review\s+questions?|bibliographical\s+notes?|bibliography|"
    r"further\s+reading"
    r")\s*$"
)


def section_sort_key(value: str) -> Tuple[int, ...]:
    """Sort numbered textbook sections naturally (3.8.2 before 3.10)."""
    try:
        return tuple(int(part) for part in str(value).split("."))
    except (TypeError, ValueError):
        return (9999,)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CleanChunk:
    chunk_id: str
    text: str
    pages: List[int]
    sections: List[str]
    primary_section: str
    page_start: int
    page_end: int
    parent_chunk_id: str
    excluded_tail_chars: int = 0


@dataclass(frozen=True)
class Resolution:
    concept_id: Optional[str]
    method: str
    score: float
    reason: str = ""


class GlobalRegistry:
    """Read-only adapter for the cumulative full-book registry."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if raw.get("registry_type") != "CUMULATIVE_GLOBAL_CONCEPT_REGISTRY":
            raise ValueError(
                f"--global-registry must be a cumulative registry; got "
                f"{raw.get('registry_type')!r}"
            )
        concepts = raw.get("concepts") or []
        if not concepts:
            raise ValueError("Global registry contains no concepts")

        self.raw = raw
        self.by_id: Dict[str, Dict[str, Any]] = {}
        self.name_index: Dict[str, List[str]] = defaultdict(list)
        self.chapter_index: Dict[str, Set[str]] = defaultdict(set)
        self.section_index: Dict[str, Set[str]] = defaultdict(set)

        for concept in concepts:
            cid = str(concept.get("concept_id", "")).strip()
            if not cid or not cid.startswith("OS_"):
                raise ValueError(f"Invalid global concept_id: {cid!r}")
            if cid in self.by_id:
                raise ValueError(f"Duplicate global concept_id: {cid}")
            if concept.get("status") != "APPROVED":
                # Formal extraction only uses approved nodes.
                continue
            self.by_id[cid] = concept

            names = [concept.get("canonical_name", ""), *(concept.get("aliases") or [])]
            for name in names:
                for key in name_keys(name):
                    if cid not in self.name_index[key]:
                        self.name_index[key].append(cid)

            owner = str(concept.get("canonical_owner_chapter", "")).strip()
            if owner:
                self.chapter_index[owner].add(cid)
            for section in concept.get("evidence_sections") or []:
                section = str(section).strip()
                if section:
                    self.section_index[section].add(cid)
                    chapter = section.split(".", 1)[0]
                    if chapter.isdigit():
                        self.chapter_index[chapter].add(cid)

        if not self.by_id:
            raise ValueError("Global registry contains no APPROVED concepts")

    def get(self, concept_id: str) -> Dict[str, Any]:
        return self.by_id[concept_id]

    def all_concepts(self) -> Iterable[Dict[str, Any]]:
        return self.by_id.values()

    def concepts_for_section(self, section: str) -> Set[str]:
        result: Set[str] = set()
        section = str(section).strip()
        if not section:
            return result
        # Exact section first.
        result.update(self.section_index.get(section, set()))
        # Parent/child section compatibility: 2.3 <-> 2.3.1.
        for sec, ids in self.section_index.items():
            if sec.startswith(section + ".") or section.startswith(sec + "."):
                result.update(ids)
        return result

    def concepts_for_chapter(self, chapter_number: str) -> Set[str]:
        return set(self.chapter_index.get(str(chapter_number), set()))

    def lexical_hits(self, text: str, allowed_ids: Optional[Set[str]] = None) -> Set[str]:
        norm_text = normalize_search_text(text)
        hits: Set[str] = set()
        for cid, concept in self.by_id.items():
            if allowed_ids is not None and cid not in allowed_ids:
                continue
            names = [concept.get("canonical_name", ""), *(concept.get("aliases") or [])]
            for name in names:
                key = normalize_name(name)
                if len(key) < 3:
                    continue
                if phrase_in_normalized_text(key, norm_text):
                    hits.add(cid)
                    break
        return hits


# ---------------------------------------------------------------------------
# Normalization and deterministic validation
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_text(text: str) -> str:
    """Whitespace/case normalization used for source-quote validation."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00ad", "")  # soft hyphen
    # Join only PDF line-wrap hyphens (``inter-\nprocess``), not normal
    # hyphenated compounds.  This makes quote validation robust without
    # weakening its source-grounding requirement.
    text = re.sub(r"(?<=[A-Za-z])[-\u2010-\u2015]\s*\n\s*(?=[A-Za-z])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def normalize_search_text(text: str) -> str:
    """Name-search normalization; treats punctuation and hyphens as separators."""
    value = unicodedata.normalize("NFKC", text or "").lower()
    value = value.replace("\u00ad", "")
    value = re.sub(r"(?<=[a-z])[-\u2010-\u2015]\s*\n\s*(?=[a-z])", "", value)
    value = value.replace("&", " and ")
    value = re.sub(r"[\u2010-\u2015_/]+", " ", value)
    value = re.sub(r"[^a-z0-9+#.]+", " ", value)
    return " ".join(value.split())


def normalize_name(name: str, *, singularize: bool = False) -> str:
    value = unicodedata.normalize("NFKC", name or "").lower()
    value = re.sub(r"(?<=[a-z])[-\u2010-\u2015]\s*\n\s*(?=[a-z])", "", value)
    value = value.replace("&", " and ")
    value = re.sub(r"[\u2010-\u2015_/]+", " ", value)
    value = re.sub(r"[^a-z0-9+#.]+", " ", value)
    tokens = value.split()
    if singularize:
        tokens = [_singularize_token(t) for t in tokens]
    return " ".join(tokens)


def _singularize_token(token: str) -> str:
    # Conservative normalization; avoids changing words such as process, class, bus.
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith(
        ("ss", "us", "is", "ics")
    ):
        return token[:-1]
    return token


def name_keys(name: str) -> Set[str]:
    return {
        key
        for key in (normalize_name(name), normalize_name(name, singularize=True))
        if key
    }


def phrase_in_normalized_text(phrase: str, normalized_text: str) -> bool:
    if not phrase:
        return False
    # Names contain only normalized word-like characters, so explicit boundaries work.
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", normalized_text) is not None


def quote_is_in_chunk(quote: str, chunk_text: str) -> bool:
    q = normalize_text(quote)
    t = normalize_text(chunk_text)
    if not q:
        return False
    if q in t:
        return True
    # PDF line wrapping may vary around hyphens; this fallback still requires the
    # same alphanumeric sequence and is used only for quote validation.
    q_compact = re.sub(r"[-\s]+", "", q)
    t_compact = re.sub(r"[-\s]+", "", t)
    return len(q_compact) >= 12 and q_compact in t_compact


def clamp_confidence(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if math.isnan(number) or math.isinf(number):
        number = default
    return max(0.0, min(1.0, number))


def token_jaccard(a: str, b: str) -> float:
    sa, sb = set(normalize_name(a, singularize=True).split()), set(
        normalize_name(b, singularize=True).split()
    )
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def definition_overlap(a: str, b: str) -> float:
    stop = {
        "the", "a", "an", "of", "to", "and", "or", "that", "in", "on", "for",
        "with", "by", "from", "is", "are", "as", "which", "through", "system",
        "operating", "service", "mechanism", "process", "program",
    }
    ta = {t for t in normalize_name(a).split() if t not in stop and len(t) > 2}
    tb = {t for t in normalize_name(b).split() if t not in stop and len(t) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def chapter_number_from_registry(registry: KPRegistry) -> str:
    match = re.search(r"(\d+)", str(registry.chapter))
    if not match:
        raise ValueError(f"Cannot infer chapter number from {registry.chapter!r}")
    return str(int(match.group(1)))


def section_compatible(local_section: str, global_concept: Mapping[str, Any]) -> bool:
    local_section = str(local_section or "").strip()
    if not local_section:
        return False
    for section in global_concept.get("evidence_sections") or []:
        section = str(section)
        if (
            section == local_section
            or section.startswith(local_section + ".")
            or local_section.startswith(section + ".")
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Local-KP -> global-concept alignment
# ---------------------------------------------------------------------------
def load_mapping_overrides(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "mappings" in raw:
        raw = raw["mappings"]
    if not isinstance(raw, dict):
        raise ValueError("Mapping override file must be a JSON object")
    return {str(k): str(v) for k, v in raw.items()}


def resolve_local_kp(
    kp: KP,
    global_registry: GlobalRegistry,
    chapter_number: str,
    overrides: Mapping[str, str],
) -> Resolution:
    # Explicit overrides take precedence. Keys may be local ID or local canonical name.
    override = overrides.get(kp.id) or overrides.get(kp.name)
    if override:
        if override not in global_registry.by_id:
            return Resolution(None, "INVALID_OVERRIDE", 0.0, f"Unknown global ID {override}")
        return Resolution(override, "OVERRIDE", 1.0)

    # Exact canonical/alias matching, including conservative singular normalization.
    exact_hits: Set[str] = set()
    for local_name in [kp.name, *kp.aliases]:
        for key in name_keys(local_name):
            exact_hits.update(global_registry.name_index.get(key, []))

    if len(exact_hits) == 1:
        return Resolution(next(iter(exact_hits)), "EXACT_NAME_OR_ALIAS", 1.0)
    if len(exact_hits) > 1:
        ranked = sorted(
            exact_hits,
            key=lambda cid: (
                section_compatible(kp.primary_section, global_registry.get(cid)),
                definition_overlap(kp.definition, global_registry.get(cid).get("short_definition", "")),
            ),
            reverse=True,
        )
        top = ranked[0]
        top_score = (
            0.7
            + (0.2 if section_compatible(kp.primary_section, global_registry.get(top)) else 0.0)
            + 0.1 * definition_overlap(
                kp.definition, global_registry.get(top).get("short_definition", "")
            )
        )
        # Accept only if the best candidate is clearly superior.
        if len(ranked) == 1:
            return Resolution(top, "EXACT_DISAMBIGUATED", top_score)
        second = ranked[1]
        first_rank = (
            int(section_compatible(kp.primary_section, global_registry.get(top))),
            definition_overlap(kp.definition, global_registry.get(top).get("short_definition", "")),
        )
        second_rank = (
            int(section_compatible(kp.primary_section, global_registry.get(second))),
            definition_overlap(kp.definition, global_registry.get(second).get("short_definition", "")),
        )
        if first_rank[0] > second_rank[0] or first_rank[1] - second_rank[1] >= 0.25:
            return Resolution(top, "EXACT_DISAMBIGUATED", top_score)
        return Resolution(None, "AMBIGUOUS_EXACT", 0.0, ", ".join(sorted(exact_hits)))

    # High-precision fuzzy matching is restricted to concepts relevant to this chapter.
    candidate_ids = global_registry.concepts_for_chapter(chapter_number)
    best: Optional[Tuple[float, float, float, str]] = None
    local_names = [kp.name, *kp.aliases]
    for cid in candidate_ids:
        concept = global_registry.get(cid)
        global_names = [concept.get("canonical_name", ""), *(concept.get("aliases") or [])]
        best_name_score = 0.0
        best_jaccard = 0.0
        for left in local_names:
            for right in global_names:
                nl = normalize_name(left, singularize=True)
                nr = normalize_name(right, singularize=True)
                if not nl or not nr:
                    continue
                score = difflib.SequenceMatcher(None, nl, nr).ratio()
                jac = token_jaccard(left, right)
                if (score, jac) > (best_name_score, best_jaccard):
                    best_name_score, best_jaccard = score, jac
        sec_bonus = 0.03 if section_compatible(kp.primary_section, concept) else 0.0
        def_bonus = 0.04 * definition_overlap(kp.definition, concept.get("short_definition", ""))
        total = min(1.0, best_name_score + sec_bonus + def_bonus)
        candidate = (total, best_jaccard, best_name_score, cid)
        if best is None or candidate > best:
            best = candidate

    if best:
        total, jac, raw_name_score, cid = best
        # Deliberately strict: false merges are more damaging than candidates.
        if (raw_name_score >= 0.965 and jac >= 0.75) or (
            raw_name_score >= 0.93 and jac >= 0.85 and total >= 0.96
        ):
            return Resolution(cid, "HIGH_CONFIDENCE_FUZZY", round(total, 4))

    return Resolution(None, "UNMAPPED", 0.0)


def build_alignment(
    local_registry: KPRegistry,
    global_registry: GlobalRegistry,
    overrides: Mapping[str, str],
) -> Dict[str, Any]:
    chapter_number = chapter_number_from_registry(local_registry)
    local_to_global: Dict[str, str] = {}
    global_to_local: Dict[str, List[str]] = defaultdict(list)
    records: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []

    for kp in local_registry.all_kps():
        resolution = resolve_local_kp(kp, global_registry, chapter_number, overrides)
        record = {
            "local_kp_id": kp.id,
            "local_name": kp.name,
            "primary_section": kp.primary_section,
            "global_concept_id": resolution.concept_id,
            "method": resolution.method,
            "score": resolution.score,
            "reason": resolution.reason,
        }
        records.append(record)
        if resolution.concept_id:
            local_to_global[kp.id] = resolution.concept_id
            global_to_local[resolution.concept_id].append(kp.id)
        else:
            candidates.append(
                {
                    "candidate_type": "NEW_CONCEPT_CANDIDATE",
                    "local_kp_id": kp.id,
                    "name": kp.name,
                    "aliases": list(kp.aliases),
                    "concept_type": kp.concept_type,
                    "definition": kp.definition,
                    "primary_section": kp.primary_section,
                    "source_pdf_pages": list(kp.source_pdf_pages),
                    "status": "PENDING_REVIEW",
                    "reason": resolution.method,
                    "details": resolution.reason,
                }
            )

    method_counts = Counter(r["method"] for r in records)
    return {
        "chapter_number": chapter_number,
        "local_to_global": local_to_global,
        "global_to_local": dict(global_to_local),
        "records": records,
        "new_concept_candidates": candidates,
        "summary": {
            "local_kps": len(records),
            "mapped_local_kps": len(local_to_global),
            "unique_global_concepts": len(global_to_local),
            "unmapped_local_kps": len(candidates),
            "by_method": dict(method_counts),
        },
    }


# ---------------------------------------------------------------------------
# Chunk cleaning and real size enforcement
# ---------------------------------------------------------------------------
def _clean_chunk_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=[A-Za-z])[-\u2010-\u2015]\s*\n\s*(?=[A-Za-z])", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _trim_leading_fragment(text: str) -> Tuple[str, bool]:
    """Remove a high-confidence overlap fragment without rewriting prose.

    A normal sentence may validly start with a lower-case code identifier, so
    this is deliberately limited to a leading punctuation mark or one of the
    common connective fragments created by an overlap (``is, either ...``).
    If no later sentence begins, the chunk is kept and reported for review.
    """
    value = text.lstrip()
    damaged = bool(
        re.match(r"^[,.;:)]", value)
        or re.match(r"^(?:is|are|was|were|and|or|but|which|that)\b", value, re.I)
    )
    if not damaged:
        return value, False
    match = re.search(r"(?:[.!?][\"')\]]?\s+|\n\s*\n)([A-Z])", value)
    if not match:
        return value, True
    return value[match.start(1):].lstrip(), True


def _remove_embedded_visual_labels(text: str) -> Tuple[str, int]:
    """Remove diagram-label runs that PyMuPDF interleaves with prose.

    A figure's ``From:``, ``To:``, and ``Port:`` labels are neither a sentence
    nor a usable relationship claim.  The pattern requires a following figure
    caption and is bounded, so ordinary prose containing one of these words is
    retained.  It also repairs the common ``applicaFigure ... tion`` artefact.
    """
    removed = 0
    label_run = re.compile(
        r"(?ims)(?:^|\n)\s*(?:from:|to:|port:).{0,2200}?(?=\b(?:figure|table|diagram)\s+\d+\.\d+\b)"
    )
    def strip_run(match: re.Match[str]) -> str:
        nonlocal removed
        removed += len(match.group(0))
        return "\n"
    value = label_run.sub(strip_run, text)
    split_caption = re.compile(
        r"(?is)([A-Za-z]{3,})(?:figure|table|diagram)\s+\d+\.\d+[^\n]{0,220}(?:\n\s*)+([a-z]{2,})"
    )
    def join_word(match: re.Match[str]) -> str:
        nonlocal removed
        removed += len(match.group(0)) - len(match.group(1)) - len(match.group(2))
        return match.group(1) + match.group(2)
    value = split_caption.sub(join_word, value)
    # Some diagrams do not retain a caption in reading order.  When several
    # strongly diagram-shaped labels appear together, preserve only the prose
    # before the first marker.  The next overlapping source Chunk retains the
    # following prose, while the unparseable visual labels cannot contaminate
    # a retrieval embedding.
    markers = list(re.finditer(
        r"(?i)\b(?:from:|to:|port:|user\s+calls\s+kernel|kernel\s+(?:sends|receives)|matchmaker)(?![a-z])",
        value,
    ))
    if len(markers) >= 2:
        cut = markers[0].start()
        removed += len(value) - cut
        value = value[:cut].rstrip()
    return value, removed


def _chunk_contamination_flags(text: str) -> List[str]:
    """Identify source-layout contamination before it reaches embedding/LLM."""
    flags: List[str] = []
    if re.search(r"(?m)^\s*[A-Z][A-Z ]{6,}\s*$", text):
        flags.append("sidebar_or_running_head")
    if re.search(r"\b(?:PIPES\s+IN\s+PRACTICE|IN\s+PRACTICE)\b", text, re.I):
        flags.append("sidebar_contamination")
    if re.search(r"\b(?:DWORD|SIZE\s+\d+|char\s*\*|\w+\s*\[\s*\d+\s*\])\b", text):
        flags.append("code_fragment_contamination")
    if "sidebar_contamination" in flags and "code_fragment_contamination" in flags:
        flags.append("cross_section_or_sidebar_mixed_content")
    return flags


def _truncate_excluded_tail(text: str, include_summary: bool) -> Tuple[str, int]:
    if include_summary:
        return text, 0
    # A chapter page can contain a running "3.9 Summary" header at the top
    # and the real Summary title later on the same page.  Keeping the first
    # match truncates valid body text (notably the Android AIDL example).
    matches = list(EXCLUDED_HEADING_RE.finditer(text))
    if not matches:
        return text, 0
    match = matches[-1]
    kept = text[: match.start()].rstrip()
    return kept, len(text) - len(kept)


def likely_incomplete_sentence(text: str) -> bool:
    """Detect only high-confidence truncations, not normal code/caption tails."""
    tail = " ".join((text or "").split())[-180:]
    if not tail:
        return False
    if re.search(r"(?:[.!?;:]|[)}\]])[\"']?$", tail):
        return False
    return bool(
        re.search(
            r"\b(?:a|an|the|and|or|but|to|of|in|on|for|with|from|that|which|whose|when|where)$",
            tail,
            re.I,
        )
    )


def _choose_split_point(text: str, target: int, minimum: int) -> int:
    if len(text) <= target:
        return len(text)
    # Prefer paragraph, sentence, then whitespace boundaries.
    search_start = max(minimum, int(target * 0.65))
    window = text[search_start:target + 1]
    candidates = [
        window.rfind("\n\n"),
        max(window.rfind(". "), window.rfind("? "), window.rfind("! ")),
        window.rfind("; "),
        window.rfind(" "),
    ]
    best = max(candidates)
    if best >= 0:
        return search_start + best + (2 if window[best:best + 2] == "\n\n" else 1)
    # Never cut a word merely to meet a soft character target.  A rare long
    # token can make a chunk slightly longer, but corrupted tokens are worse
    # for embeddings, evidence quotes, and later retrieval.
    if target < len(text) and text[target - 1].isalnum() and text[target].isalnum():
        match = re.search(r"\s+", text[target:])
        return len(text) if match is None else target + match.start()
    return target


def _forward_to_token_boundary(text: str, start: int) -> int:
    """Move an overlap start right when it would begin inside a word."""
    start = max(0, min(start, len(text)))
    if start == 0 or start >= len(text):
        return start
    if not (text[start - 1].isalnum() and text[start].isalnum()):
        return start
    match = re.search(r"\s+", text[start:])
    return len(text) if match is None else start + match.end()


def _split_chunk_text(
    text: str,
    max_chars: int,
    overlap: int,
    min_chars: int,
) -> List[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("chunk overlap must be >= 0 and < max chunk size")
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    start = 0
    while start < len(text):
        remaining = text[start:]
        if len(remaining) <= max_chars:
            tail = remaining.strip()
            if tail:
                if len(tail) < min_chars and parts:
                    merged = (parts[-1] + "\n\n" + tail).strip()
                    if len(merged) <= max_chars:
                        parts[-1] = merged
                    else:
                        parts.append(tail)
                else:
                    parts.append(tail)
            break
        split = _choose_split_point(remaining, max_chars, min_chars)
        piece = remaining[:split].strip()
        if not piece:
            split = _choose_split_point(remaining, max_chars, 1)
            piece = remaining[:split].strip()
        parts.append(piece)
        next_start = _forward_to_token_boundary(
            text, start + split - overlap
        )
        if next_start <= start:
            next_start = start + split
        start = next_start
    return parts


def postprocess_chunks(
    raw_chunks: Sequence[Any],
    *,
    max_chunk_chars: int,
    overlap: int,
    min_chars: int,
    include_summary: bool,
) -> Tuple[List[CleanChunk], Dict[str, Any]]:
    cleaned: List[CleanChunk] = []
    excluded_chunks = 0
    excluded_tail_chars = 0
    excluded_visual_label_chunks: List[Dict[str, Any]] = []
    trimmed_leading_fragments: List[Dict[str, Any]] = []
    removed_embedded_visual_label_chars = 0
    excluded_contaminated_chunks: List[Dict[str, Any]] = []
    sequence = 1

    for raw in raw_chunks:
        parent_id = str(getattr(raw, "chunk_id", f"raw{sequence:04d}"))
        text = _clean_chunk_text(str(getattr(raw, "text", "")))
        text, trimmed_fragment = _trim_leading_fragment(text)
        text, visual_chars = _remove_embedded_visual_labels(text)
        removed_embedded_visual_label_chars += visual_chars
        contamination_flags = _chunk_contamination_flags(text)
        if "cross_section_or_sidebar_mixed_content" in contamination_flags:
            excluded_chunks += 1
            excluded_contaminated_chunks.append({
                "parent_chunk_id": parent_id,
                "pages": list(getattr(raw, "pages", []) or []),
                "primary_section": str(getattr(raw, "primary_section", "") or ""),
                "flags": contamination_flags,
                "text_preview": text[:500],
            })
            continue
        if trimmed_fragment:
            trimmed_leading_fragments.append({
                "parent_chunk_id": parent_id,
                "text_preview": text[:180],
            })
        # `section_chunker` has already made a geometry-aware end-matter cut
        # before a page is allowed to join the next page.  Reapplying a
        # text-only Summary regex here is unsafe: in a cross-page Chunk, the
        # next page can begin with a running ``3.9 Summary`` head while its
        # remaining body text still contains AIDL or another final example.
        # Do not make a second lossy cut after section-aware chunking.
        removed = 0
        excluded_tail_chars += removed
        if len(text) < min_chars:
            excluded_chunks += 1
            continue
        # A PDF figure's individual labels are not teaching prose.  They are
        # especially harmful when extraction order interleaves them with a
        # caption (for example an RPC diagram).  Retain the audit record but
        # never embed or extract graph facts from such a fragment.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        short_lines = sum(1 for line in lines if len(line.split()) <= 5)
        prose_sentences = len(re.findall(r"[.!?](?:\s|$)", text))
        is_visual_label_fragment = (
            bool(re.search(r"\b(?:figure|table|diagram)\s+\d+\.\d+\b", text, re.I))
            and len(lines) >= 4
            and short_lines / max(1, len(lines)) >= 0.70
            and prose_sentences <= 1
        )
        if is_visual_label_fragment:
            excluded_chunks += 1
            excluded_visual_label_chunks.append({
                "parent_chunk_id": parent_id,
                "pages": list(getattr(raw, "pages", []) or []),
                "primary_section": str(getattr(raw, "primary_section", "") or ""),
                "reason": "figure_or_table_label_fragment_without_retrieval_prose",
                "text_preview": text[:300],
            })
            continue
        parts = _split_chunk_text(text, max_chunk_chars, overlap, min_chars)
        for part in parts:
            if len(part) < min_chars and cleaned:
                # Small tails are not independently useful and tend to create weak evidence.
                excluded_chunks += 1
                continue
            pages = list(getattr(raw, "pages", []) or [])
            page_start = int(getattr(raw, "page_start", pages[0] if pages else 0) or 0)
            page_end = int(getattr(raw, "page_end", pages[-1] if pages else page_start) or page_start)
            sections = [str(s) for s in (getattr(raw, "sections", []) or [])]
            primary_section = str(getattr(raw, "primary_section", "") or "")
            cleaned.append(
                CleanChunk(
                    chunk_id=f"c{sequence:04d}",
                    text=part,
                    pages=pages,
                    sections=sections,
                    primary_section=primary_section,
                    page_start=page_start,
                    page_end=page_end,
                    parent_chunk_id=parent_id,
                    excluded_tail_chars=removed,
                )
            )
            sequence += 1

    lengths = [len(c.text) for c in cleaned]
    report = {
        "raw_chunk_count": len(raw_chunks),
        "final_chunk_count": len(cleaned),
        "dropped_or_too_small_chunks": excluded_chunks,
        "excluded_tail_chars": excluded_tail_chars,
        "excluded_visual_label_chunks": excluded_visual_label_chunks,
        "trimmed_leading_fragments": trimmed_leading_fragments,
        "removed_embedded_visual_label_chars": removed_embedded_visual_label_chars,
        "excluded_contaminated_chunks": excluded_contaminated_chunks,
        "actual_length": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": round(statistics.mean(lengths), 2) if lengths else 0,
            "median": round(statistics.median(lengths), 2) if lengths else 0,
        },
    }
    return cleaned, report


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------
_GLOBAL_CANDIDATE_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "that", "in", "on",
    "for", "with", "by", "from", "is", "are", "as", "which", "through",
    "system", "operating", "service", "mechanism", "concept", "computer",
    "program", "process", "memory", "data", "information", "using", "used",
    "use", "provides", "provide", "support", "supports", "within", "between",
}


def _global_content_match_score(chunk_text: str, concept: Mapping[str, Any]) -> Tuple[float, str]:
    """Rank one approved registry Concept from its own text, never its chapter.

    This deliberately does *not* use owner chapter, evidence section, or the
    chapter-local bridge.  Exact canonical/alias phrases receive the largest
    score.  A smaller fallback is allowed only when at least two distinctive
    definition terms occur in the Chunk; the LLM still has to return a
    quote that passes the normal evidence validator before a formal node is
    emitted.
    """
    normalized_chunk = normalize_search_text(chunk_text)
    chunk_terms = {
        token for token in normalize_name(chunk_text).split()
        if len(token) >= 3 and token not in _GLOBAL_CANDIDATE_STOPWORDS
    }
    best_name_score = 0.0
    for name in [concept.get("canonical_name", ""), *(concept.get("aliases") or [])]:
        key = normalize_name(str(name))
        if len(key) < 3:
            continue
        if phrase_in_normalized_text(key, normalized_chunk):
            # Prefer a longer exact phrase but never use section provenance as
            # a hidden tie-breaker.
            best_name_score = max(best_name_score, 1000.0 + 25.0 * len(key.split()))
    if best_name_score:
        return best_name_score, "all_global_exact_name_or_alias"

    definition_terms = {
        token for token in normalize_name(str(concept.get("short_definition") or "")).split()
        if len(token) >= 4 and token not in _GLOBAL_CANDIDATE_STOPWORDS
    }
    shared = chunk_terms & definition_terms
    if len(shared) >= 2:
        # This is candidate generation only, not evidence.  Requiring two
        # distinctive terms prevents a generic word such as "memory" from
        # selecting an unrelated full-book Concept.
        coverage = len(shared) / max(1, len(definition_terms))
        return 200.0 + 20.0 * len(shared) + 100.0 * coverage, "all_global_definition_terms"
    return 0.0, ""


def candidate_global_concepts_for_chunk(
    chunk: CleanChunk,
    local_registry: KPRegistry,
    global_registry: GlobalRegistry,
    alignment: Mapping[str, Any],
    max_kps: int,
    candidate_scope: str = "chapter_scoped",
) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = defaultdict(float)
    sources: Dict[str, Set[str]] = defaultdict(set)
    local_to_global: Mapping[str, str] = alignment["local_to_global"]
    chapter_number = alignment["chapter_number"]

    def promote(concept_id: str, score: float, source: str) -> None:
        if concept_id not in global_registry.by_id:
            return
        scores[concept_id] = max(scores[concept_id], score)
        sources[concept_id].add(source)

    if candidate_scope == "all_global_chunk_match":
        # All APPROVED Concepts are considered for every Chunk.  The chapter
        # bridge remains available for page/section provenance only; it cannot
        # make a Concept eligible or ineligible here.
        for cid, concept in global_registry.by_id.items():
            score, source = _global_content_match_score(chunk.text, concept)
            if score:
                promote(cid, score, source)
    else:
        # 1. Chapter-local KP metadata mapped to the global registry.
        for section in chunk.sections or [chunk.primary_section]:
            for kp in local_registry.kps_in_section(section, include_cross_ref=True):
                cid = local_to_global.get(kp.id)
                if cid:
                    promote(cid, 100.0, "chapter_local_kp")
                    if kp.primary_section == chunk.primary_section:
                        scores[cid] += 20.0
                        sources[cid].add("primary_section_kp")

        # 2. Full-book registry section evidence.
        for section in chunk.sections or [chunk.primary_section]:
            for cid in global_registry.concepts_for_section(section):
                promote(cid, 80.0, "registry_evidence_section")

        # 3. Lexical hits are limited to this chapter's owned concepts.  A global
        # lexical scan used to promote accidental word matches such as "high
        # memory" -> ZONE_HIGHMEM and "stream I/O" -> STREAMS.  Legitimate
        # cross-chapter reuse is supplied by explicit evidence_sections above.
        chapter_ids = global_registry.concepts_for_chapter(chapter_number)
        for cid in global_registry.lexical_hits(chunk.text, chapter_ids):
            promote(cid, 120.0, "chapter_lexical_name_or_alias")

    # Do not add a broader concept merely because a child was selected.  That
    # previously exposed broad background terms to the model even when the
    # Chunk contained no evidence for them.  A broader Concept remains
    # eligible only when it is explicitly declared for this section or named
    # in the source text by one of the rules above.

    def rank(item: Tuple[str, float]) -> Tuple[float, int, str]:
        cid, score = item
        concept = global_registry.get(cid)
        section_bonus = 0 if candidate_scope == "all_global_chunk_match" else int(section_compatible(chunk.primary_section, concept))
        return (score, section_bonus, concept.get("canonical_name", ""))

    selected = sorted(scores.items(), key=rank, reverse=True)[:max_kps]
    output: List[Dict[str, Any]] = []
    for cid, score in selected:
        concept = dict(global_registry.get(cid))
        # Audit metadata is retained in each checkpoint but is not interpreted
        # as part of the stable global Concept identity.
        concept["_candidate_score"] = round(score, 3)
        concept["_candidate_sources"] = sorted(sources[cid])
        output.append(concept)
    return output


def format_global_concepts(concepts: Sequence[Mapping[str, Any]], max_def_chars: int = 180) -> str:
    lines: List[str] = []
    for concept in concepts:
        aliases = concept.get("aliases") or []
        alias_text = f"; aliases={', '.join(aliases[:6])}" if aliases else ""
        lines.append(
            f"- {concept['concept_id']} | {concept.get('canonical_name', '')}{alias_text}"
        )
        lines.append(
            f"  definition: {str(concept.get('short_definition', ''))[:max_def_chars]}"
        )
        lines.append(
            f"  domain={concept.get('domain_id', '')}; broader={concept.get('broader_concept_id', '')}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM extraction and relation verification
# ---------------------------------------------------------------------------
EVIDENCE_SYSTEM = r"""你是作業系統教材知識圖譜的證據抽取器。你只能使用候選清單中的正式 global concept_id。

任務一：Evidence
- 每個 evidence 必須引用教材片段中的逐字原文。
- evidence_type 只能是 definition、example、usage、mention。
- definition：原文明確定義概念。
- example：原文明確提供例子。
- usage：原文描述用途、行為、機制或操作方式。
- mention：只提到名稱，沒有足夠教學內容。
- confidence 為 0 到 1。
- 每個 relation 的兩端都必須各自出現在 evidence 陣列中。

任務二：Typed relation
只有原文直接支持時才能輸出。方向規則必須嚴格遵守：
- HAS_SUBTYPE：父概念 → 子類型。例：User Interface → Command-Line Interface。
- PART_OF：部分／元件 → 整體。例：Loader → Run-Time Environment（僅在原文確實說它是組成部分時）。
- PREREQUISITE_OF：先備概念 → 後續概念。只有原文明確呈現先備依賴才可使用。
- IMPLEMENTS：具體實作 → 抽象介面或模型。
- USES：使用者／機制 → 被使用的工具、機制或資源。
- INVOKES：呼叫者 → 被呼叫的介面、程序或系統呼叫。
- PROVIDES：提供者 → 被提供的服務或能力。
- MANAGES：管理者 → 被管理的資源或物件。
- ENABLES：使能機制 → 被促成的能力或結果。
- CAUSES：原因 → 結果。
- EXAMPLE_OF：具體例子 → 一般類別。
- CONTRASTED_WITH：兩概念被原文明確比較；此關係對稱。
- MAPS_TO：來源表示／名稱／位址 → 目標表示／名稱／位址。

禁止：
- 不得新增候選外的概念。
- 不得把同義詞建立成關係。
- 不得因為同節、同 concept type 就建立關係。
- 不得將「A 與 B 同時被提到」視為 PART_OF、USES 或 HAS_SUBTYPE。
- 不確定方向或類型時，不輸出 relation。
- quote 必須能在教材片段中逐字找到。

只輸出 JSON：
{
  "evidence": [
    {
      "concept_id": "OS_...",
      "quote": "exact source quote",
      "evidence_type": "definition|example|usage|mention",
      "matches_definition": true,
      "confidence": 0.95
    }
  ],
  "relations": [
    {
      "source_concept_id": "OS_...",
      "target_concept_id": "OS_...",
      "relation": "HAS_SUBTYPE|PART_OF|IMPLEMENTS|USES|INVOKES|PROVIDES|MANAGES|ENABLES|CAUSES|EXAMPLE_OF|CONTRASTED_WITH|MAPS_TO|REPRESENTS|CONTAINS|CREATES|TRANSITIONS_TO|SAVES_TO|LOADS_FROM|RECLAIMS|WAITS_ON|SCHEDULES",
      "quote": "exact source quote",
      "confidence": 0.90
    }
  ],
  "schema_gap_candidates": [
    {
      "proposed_name": "only if a major source-grounded concept is absent",
      "category": "concept|figure|case",
      "quote": "exact source quote",
      "rationale": "why the existing registry cannot represent it"
    }
  ]
}
"""

EVIDENCE_USER_TEMPLATE = r"""# Source chunk
chunk_id={chunk_id}
section={section}
pages={page_start}-{page_end}

<SOURCE_CHUNK>
{chunk_text}
</SOURCE_CHUNK>

# Approved candidate concepts
{concept_list}

Formal evidence rules:
- Use only the candidate concept_id values listed above.
- Quote the exact sentence or passage from SOURCE_CHUNK. Do not paraphrase.
- Create evidence only when the quote explicitly names the Concept or one of
  its aliases. A definition may instead be used only when the quote itself
  states the registered definition.
- A relation must name at least one of its two endpoints in its quoted text;
  both endpoints must also have their own evidence in this same Chunk.
- If the source is merely related background, omit it instead of guessing.

Extract only source-grounded evidence and relations. Return JSON only.
"""

RELATION_VERIFY_SYSTEM = r"""你是教材知識圖譜關係覆核器。針對每一條候選關係，只能做以下決定：
- ACCEPT：原文直接支持，且方向與類型正確。
- REVERSE：原文支持但方向相反。
- RETYPE：原文支持兩概念有關，但 relation 類型錯誤；new_relation 必須來自允許清單。
- DROP：原文不足、只共同出現、方向無法判斷、或關係屬於推測。

關係方向：HAS_SUBTYPE 父→子；PART_OF 部分→整體；PREREQUISITE_OF 先備→後續；IMPLEMENTS 實作→抽象；INVOKES 呼叫者→被呼叫者；PROVIDES 提供者→服務；MANAGES 管理者→資源；CAUSES 原因→結果；EXAMPLE_OF 例子→類別。

只輸出 JSON：
{"decisions":[{"index":0,"action":"ACCEPT|REVERSE|RETYPE|DROP","new_relation":"...或空字串","reason":"簡短理由"}]}
"""


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    match = re.search(r"```(?:json)?\s*(\{.+\})\s*```", text, re.S)
    if match:
        text = match.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def call_json_model(model: Any, prompt: str, retries: int) -> Tuple[Optional[Dict[str, Any]], str]:
    last_error = ""
    for attempt in range(retries + 1):
        try:
            raw = generate_with_rotation(model, prompt)
        except Exception as exc:  # API/client failure
            last_error = f"model_error: {exc}"
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
                continue
            return None, last_error
        parsed = extract_json_object(str(raw))
        if parsed is not None:
            return parsed, ""
        last_error = "invalid_json"
        if attempt < retries:
            prompt = (
                prompt
                + "\n\nPrevious response was invalid JSON. Return exactly one valid JSON object and no markdown."
            )
    return None, last_error


def concept_terms(concept: Mapping[str, Any]) -> List[str]:
    return [concept.get("canonical_name", ""), *(concept.get("aliases") or [])]


def concept_lexically_present(concept: Mapping[str, Any], chunk_text: str) -> bool:
    normalized = normalize_search_text(chunk_text)
    return any(
        phrase_in_normalized_text(normalize_name(term), normalized)
        for term in concept_terms(concept)
        if normalize_name(term)
    )


_ANCHOR_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "of",
    "on", "or", "the", "to", "with", "system", "operating", "process",
    "program", "computer", "service", "mechanism", "concept",
}


def distinctive_name_tokens_in_quote(concept: Mapping[str, Any], quote: str) -> float:
    """Return coverage of a distinctive registered name phrase in a quote.

    Exact name matching is ideal, but PDF source text often states a definition
    such as ``Text section—the executable code`` while the canonical name is
    ``Process Text Section``.  The shared distinctive tokens ``text`` and
    ``section`` are a reproducible anchor; a generic one-word overlap such as
    ``process`` never is.
    """
    quote_tokens = set(normalize_name(quote, singularize=True).split())
    best = 0.0
    for term in concept_terms(concept):
        tokens = {
            token for token in normalize_name(str(term), singularize=True).split()
            if len(token) > 2 and token not in _ANCHOR_STOPWORDS
        }
        if len(tokens) < 2:
            continue
        overlap = len(tokens & quote_tokens) / len(tokens)
        if len(tokens & quote_tokens) >= 2:
            best = max(best, overlap)
    return best


def evidence_anchor(
    concept: Mapping[str, Any],
    quote: str,
    evidence_type: str,
) -> Tuple[Optional[str], float]:
    """Return an independently checkable anchor for formal evidence.

    A model may only connect a Concept to a Chunk when the exact quoted text
    names the Concept (or a registered alias), or when a claimed definition has
    measurable overlap with the registry definition.  This prevents a generic
    sentence in a broad Chunk from becoming evidence for a merely plausible
    candidate Concept.
    """
    if concept_lexically_present(concept, quote):
        return "name_or_alias_in_quote", 1.0
    distinctive_coverage = distinctive_name_tokens_in_quote(concept, quote)
    if distinctive_coverage >= 0.67:
        return "distinctive_name_tokens_in_quote", distinctive_coverage
    overlap = definition_overlap(
        quote, str(concept.get("short_definition") or "")
    )
    if evidence_type == "definition" and overlap >= MIN_DEFINITION_OVERLAP:
        return "definition_overlap", overlap
    return None, overlap


def validate_schema_gap_candidates(
    parsed: Mapping[str, Any],
    chunk: CleanChunk,
    global_registry: GlobalRegistry,
) -> Dict[str, List[Dict[str, Any]]]:
    """Separate genuine concept gaps from assets/APIs and registry duplicates.

    The extractor may notice a Figure label or a platform/API name.  Those are
    useful audit artefacts, but not evidence that the global concept schema is
    incomplete.  A name already represented by a canonical name or alias is
    also not a new concept candidate.
    """
    output: Dict[str, List[Dict[str, Any]]] = {
        "schema_gap_candidates": [],
        "asset_candidates": [],
        "api_candidates": [],
        "rejected_schema_gap_candidates": [],
    }
    def near_duplicate_ids(candidate_name: str) -> List[str]:
        """Conservative global de-duplication beyond exact canonical/alias text.

        ``Text Section`` is already represented by ``Process Text Section``;
        ``Background Process`` is represented by ``Android Background Process``.
        Both are safe matches because every candidate token occurs in a longer
        approved concept name.  A one-word candidate is deliberately not
        merged this way, as that would collapse unrelated broad concepts.
        """
        candidate_tokens = set(normalize_name(candidate_name, singularize=True).split())
        candidate_tokens -= {"the", "a", "an", "of", "and", "for", "to"}
        if len(candidate_tokens) < 2:
            return []
        matches: List[str] = []
        for cid, concept in global_registry.by_id.items():
            for term in concept_terms(concept):
                term_tokens = set(normalize_name(str(term), singularize=True).split())
                if candidate_tokens <= term_tokens:
                    matches.append(cid)
                    break
        return sorted(set(matches))

    seen: Set[Tuple[str, str]] = set()
    for item in parsed.get("schema_gap_candidates") or []:
        if not isinstance(item, Mapping):
            continue
        name = " ".join(str(item.get("proposed_name", "")).split())
        quote = str(item.get("quote", "")).strip()
        rationale = " ".join(str(item.get("rationale", "")).split())[:600]
        category = " ".join(str(item.get("category", "")).split())[:80]
        if len(name) < 3 or len(name) > 160 or not quote_is_in_chunk(quote, chunk.text):
            continue
        key = (normalize_name(name), normalize_text(quote))
        if key in seen:
            continue
        seen.add(key)
        record = {
            "proposed_name": name,
            "quote": quote[:700],
            "rationale": rationale,
            "category": category or "concept_or_case",
            "review_status": "PENDING_SCHEMA_REVIEW",
        }
        existing_ids = sorted({
            cid
            for name_key in name_keys(name)
            for cid in global_registry.name_index.get(name_key, [])
        })
        if existing_ids:
            output["rejected_schema_gap_candidates"].append({
                **record,
                "review_status": "REJECTED_ALREADY_IN_GLOBAL_REGISTRY",
                "matched_global_concept_ids": existing_ids,
            })
            continue
        near_ids = near_duplicate_ids(name)
        if near_ids:
            output["rejected_schema_gap_candidates"].append({
                **record,
                "review_status": "REJECTED_NEAR_DUPLICATE_GLOBAL_CONCEPT",
                "matched_global_concept_ids": near_ids,
                "match_method": "candidate_name_token_subset_of_canonical_or_alias",
            })
            continue

        normalized_category = normalize_name(category)
        normalized_name = normalize_name(name)
        if (
            normalized_category in {"figure", "table", "diagram", "case", "asset", "illustration"}
            or re.match(r"^(?:figure|table|diagram)\s+\d", normalized_name)
        ):
            output["asset_candidates"].append({**record, "review_status": "PENDING_ASSET_REVIEW"})
        elif normalized_category in {"api", "library", "interface", "system call", "command"}:
            output["api_candidates"].append({**record, "review_status": "PENDING_API_REVIEW"})
        elif normalized_category in {"", "concept", "teachable concept", "concept or case"}:
            output["schema_gap_candidates"].append(record)
        else:
            output["rejected_schema_gap_candidates"].append({
                **record,
                "review_status": "REJECTED_UNSUPPORTED_GAP_CATEGORY",
            })
    return output


def validate_extraction(
    parsed: Mapping[str, Any],
    chunk: CleanChunk,
    candidates: Sequence[Mapping[str, Any]],
    global_registry: GlobalRegistry,
    min_relation_confidence: float,
) -> Dict[str, Any]:
    candidate_ids = {str(c["concept_id"]) for c in candidates}
    evidence: List[Dict[str, Any]] = []
    rejected_evidence: List[Dict[str, Any]] = []
    seen_evidence: Set[Tuple[str, str, str]] = set()

    for item in parsed.get("evidence") or []:
        cid = str(item.get("concept_id", ""))
        quote = str(item.get("quote", "")).strip()
        evidence_type = str(item.get("evidence_type", "mention")).lower()
        confidence = clamp_confidence(item.get("confidence"), 0.5)
        concept = global_registry.get(cid) if cid in global_registry.by_id else {}
        matches_definition = bool(item.get("matches_definition", False))
        anchor, definition_score = evidence_anchor(concept, quote, evidence_type)
        reason = ""
        if cid not in candidate_ids or cid not in global_registry.by_id:
            reason = "unknown_or_non_candidate_concept"
        elif evidence_type not in ALLOWED_EVIDENCE_TYPES:
            reason = "invalid_evidence_type"
        elif not quote_is_in_chunk(quote, chunk.text):
            reason = "quote_not_found_in_chunk"
        elif not quote:
            reason = "empty_quote"
        elif confidence < MIN_EVIDENCE_CONFIDENCE[evidence_type]:
            reason = "below_evidence_confidence_threshold"
        elif not anchor:
            reason = "quote_not_anchored_to_concept_name_alias_or_definition"
        elif evidence_type == "definition" and not matches_definition:
            reason = "definition_not_confirmed_by_model"
        if reason:
            rejected_evidence.append({**dict(item), "reason": reason})
            continue
        key = (cid, normalize_text(quote), evidence_type)
        if key in seen_evidence:
            continue
        seen_evidence.add(key)
        evidence.append(
            {
                "concept_id": cid,
                "quote": quote[:700],
                "evidence_type": evidence_type,
                "matches_definition": matches_definition,
                "confidence": confidence,
                "anchor_type": anchor,
                "definition_overlap": round(definition_score, 4),
            }
        )

    supported_ids = {e["concept_id"] for e in evidence}
    relations: List[Dict[str, Any]] = []
    rejected_relations: List[Dict[str, Any]] = []
    seen_relations: Set[Tuple[str, str, str, str]] = set()

    for item in parsed.get("relations") or []:
        source = str(item.get("source_concept_id", ""))
        target = str(item.get("target_concept_id", ""))
        relation = str(item.get("relation", "")).upper()
        quote = str(item.get("quote", "")).strip()
        confidence = clamp_confidence(item.get("confidence"), 0.0)
        source_concept = global_registry.get(source) if source in global_registry.by_id else {}
        target_concept = global_registry.get(target) if target in global_registry.by_id else {}
        quote_has_endpoint_anchor = (
            concept_lexically_present(source_concept, quote)
            or concept_lexically_present(target_concept, quote)
        )
        reason = ""
        if source not in candidate_ids or target not in candidate_ids:
            reason = "unknown_or_non_candidate_endpoint"
        elif source == target:
            reason = "self_loop"
        elif relation not in ALLOWED_RELATIONS:
            reason = "invalid_relation_type"
        elif confidence < min_relation_confidence:
            reason = "below_confidence_threshold"
        elif not quote_is_in_chunk(quote, chunk.text):
            reason = "quote_not_found_in_chunk"
        elif source not in supported_ids or target not in supported_ids:
            reason = "relation_endpoint_without_same_chunk_evidence"
        elif not quote_has_endpoint_anchor:
            reason = "relation_quote_lacks_named_endpoint_anchor"
        if reason:
            rejected_relations.append({**dict(item), "reason": reason})
            continue

        # Deterministic direction repair when the registry explicitly declares broader.
        if relation == "HAS_SUBTYPE":
            source_broader = global_registry.get(source).get("broader_concept_id")
            target_broader = global_registry.get(target).get("broader_concept_id")
            if source_broader == target:
                source, target = target, source
            elif target_broader == source:
                pass
            # Otherwise the verifier must decide; do not guess here.

        if relation in SYMMETRIC_RELATIONS and source > target:
            source, target = target, source

        key = (source, target, relation, normalize_text(quote))
        if key in seen_relations:
            continue
        seen_relations.add(key)
        relations.append(
            {
                "source_concept_id": source,
                "target_concept_id": target,
                "relation": relation,
                "quote": quote[:700],
                "confidence": confidence,
                "verification_status": "PENDING",
            }
        )

    return {
        "evidence": evidence,
        "relations": relations,
        "rejected_evidence": rejected_evidence,
        "rejected_relations": rejected_relations,
        **validate_schema_gap_candidates(parsed, chunk, global_registry),
    }


def verify_relations(
    model: Any,
    chunk: CleanChunk,
    relations: Sequence[Mapping[str, Any]],
    global_registry: GlobalRegistry,
    retries: int,
    failure_policy: str,
) -> Tuple[List[Dict[str, Any]], str]:
    if not relations:
        return [], ""
    payload = []
    for index, rel in enumerate(relations):
        source = global_registry.get(str(rel["source_concept_id"]))
        target = global_registry.get(str(rel["target_concept_id"]))
        payload.append(
            {
                "index": index,
                "source": {
                    "concept_id": source["concept_id"],
                    "name": source.get("canonical_name"),
                    "definition": source.get("short_definition"),
                },
                "target": {
                    "concept_id": target["concept_id"],
                    "name": target.get("canonical_name"),
                    "definition": target.get("short_definition"),
                },
                "relation": rel["relation"],
                "quote": rel["quote"],
            }
        )
    prompt = (
        RELATION_VERIFY_SYSTEM
        + "\n\n"
        + RELATION_SAFETY_CONTRACT
        + "\n\n"
        + RELATION_DECISION_CONTRACT
        + "\n\n# Authoritative allowed relation types\n"
        + ", ".join(sorted(ALLOWED_RELATIONS))
        + "\n\n# Source chunk\n"
        + chunk.text
        + "\n\n# Candidate relations\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    parsed, error = call_json_model(model, prompt, retries)
    if parsed is None:
        if failure_policy == "keep":
            return [{**dict(r), "verification_status": "UNVERIFIED_API_FAILURE"} for r in relations], error
        return [], error

    decisions = {
        int(d.get("index")): d
        for d in (parsed.get("decisions") or [])
        if isinstance(d, dict) and str(d.get("index", "")).isdigit()
    }
    verified: List[Dict[str, Any]] = []
    for index, relation in enumerate(relations):
        decision = decisions.get(index)
        if not decision:
            if failure_policy == "keep":
                verified.append({**dict(relation), "verification_status": "UNVERIFIED_NO_DECISION"})
            continue
        action = str(decision.get("action", "DROP")).upper()
        reason = str(decision.get("reason", ""))[:300]
        current = dict(relation)
        if action == "ACCEPT":
            current["verification_status"] = "VERIFIED"
            current["verification_action"] = "ACCEPT"
            current["verification_reason"] = reason
            verified.append(current)
        elif action == "REVERSE":
            current["original_source_concept_id"] = current["source_concept_id"]
            current["original_target_concept_id"] = current["target_concept_id"]
            current["source_concept_id"], current["target_concept_id"] = (
                current["target_concept_id"],
                current["source_concept_id"],
            )
            if current["relation"] in SYMMETRIC_RELATIONS and current["source_concept_id"] > current["target_concept_id"]:
                current["source_concept_id"], current["target_concept_id"] = (
                    current["target_concept_id"], current["source_concept_id"]
                )
            current["verification_status"] = "VERIFIED_REVERSED"
            current["verification_action"] = "REVERSE"
            current["verification_reason"] = reason
            verified.append(current)
        elif action == "RETYPE":
            new_relation = str(decision.get("new_relation", "")).upper()
            if new_relation in ALLOWED_RELATIONS:
                current["original_relation"] = current["relation"]
                current["relation"] = new_relation
                if new_relation in SYMMETRIC_RELATIONS and current["source_concept_id"] > current["target_concept_id"]:
                    current["source_concept_id"], current["target_concept_id"] = (
                        current["target_concept_id"], current["source_concept_id"]
                    )
                current["verification_status"] = "VERIFIED_RETYPED"
                current["verification_action"] = "RETYPE"
                current["verification_reason"] = reason
                verified.append(current)
        # DROP is intentionally omitted.
    return verified, ""


def extract_chunk(
    model: Any,
    chunk: CleanChunk,
    local_registry: KPRegistry,
    global_registry: GlobalRegistry,
    alignment: Mapping[str, Any],
    *,
    max_candidates: int,
    retries: int,
    min_relation_confidence: float,
    relation_verification: bool,
    verification_failure_policy: str,
    candidate_scope: str,
) -> Dict[str, Any]:
    candidates = candidate_global_concepts_for_chunk(
        chunk, local_registry, global_registry, alignment, max_candidates, candidate_scope
    )
    candidate_ids = [c["concept_id"] for c in candidates]
    fingerprint = extraction_fingerprint(chunk, candidate_ids)
    candidate_details = [
        {
            "concept_id": str(candidate["concept_id"]),
            "score": float(candidate.get("_candidate_score") or 0),
            "sources": list(candidate.get("_candidate_sources") or []),
        }
        for candidate in candidates
    ]
    if not candidates:
        return {
            "chunk_id": chunk.chunk_id,
            "fingerprint": fingerprint,
            "candidate_ids": [],
            "candidate_details": [],
            "evidence": [],
            "relations": [],
            "rejected_evidence": [],
            "rejected_relations": [],
            "schema_gap_candidates": [],
            "asset_candidates": [],
            "api_candidates": [],
            "rejected_schema_gap_candidates": [],
            "errors": [],
        }

    prompt = EVIDENCE_SYSTEM + "\n\n" + RELATION_SAFETY_CONTRACT + "\n\n" + EVIDENCE_USER_TEMPLATE.format(
        chunk_id=chunk.chunk_id,
        section=chunk.primary_section,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        chunk_text=chunk.text,
        concept_list=format_global_concepts(candidates),
    )
    parsed, error = call_json_model(model, prompt, retries)
    errors: List[str] = []
    if parsed is None:
        errors.append(error or "extraction_failed")
        return {
            "chunk_id": chunk.chunk_id,
            "fingerprint": fingerprint,
            "candidate_ids": candidate_ids,
            "candidate_details": candidate_details,
            "evidence": [],
            "relations": [],
            "rejected_evidence": [],
            "rejected_relations": [],
            "schema_gap_candidates": [],
            "asset_candidates": [],
            "api_candidates": [],
            "rejected_schema_gap_candidates": [],
            "errors": errors,
        }

    validated = validate_extraction(
        parsed, chunk, candidates, global_registry, min_relation_confidence
    )
    relations = validated["relations"]
    if relation_verification and relations:
        relations, verify_error = verify_relations(
            model,
            chunk,
            relations,
            global_registry,
            retries,
            verification_failure_policy,
        )
        if verify_error:
            errors.append(f"relation_verification: {verify_error}")
    else:
        relations = [
            {**dict(r), "verification_status": "NOT_REQUESTED"} for r in relations
        ]

    return {
        "chunk_id": chunk.chunk_id,
        "fingerprint": fingerprint,
        "candidate_ids": candidate_ids,
        "candidate_details": candidate_details,
        "evidence": validated["evidence"],
        "relations": relations,
        "rejected_evidence": validated["rejected_evidence"],
        "rejected_relations": validated["rejected_relations"],
        "schema_gap_candidates": validated["schema_gap_candidates"],
        "asset_candidates": validated["asset_candidates"],
        "api_candidates": validated["api_candidates"],
        "rejected_schema_gap_candidates": validated["rejected_schema_gap_candidates"],
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Checkpoints and aggregation
# ---------------------------------------------------------------------------
def expected_fingerprint(
    chunk: CleanChunk,
    local_registry: KPRegistry,
    global_registry: GlobalRegistry,
    alignment: Mapping[str, Any],
    max_candidates: int,
    candidate_scope: str,
) -> str:
    candidates = candidate_global_concepts_for_chunk(
        chunk, local_registry, global_registry, alignment, max_candidates, candidate_scope
    )
    # Must exactly mirror the fingerprint written by `extract_chunk`.  Changes
    # to EXTRACTION_POLICY_VERSION intentionally force re-extraction; purely
    # aggregation-only changes still reuse the existing checkpoint.
    return extraction_fingerprint(
        chunk, [c["concept_id"] for c in candidates]
    )


def extraction_fingerprint(
    chunk: CleanChunk,
    candidate_ids: Sequence[str],
    candidate_scope: str = "chapter_scoped",
) -> str:
    return hashlib.sha256(
        (
            EXTRACTION_POLICY_VERSION
            + "\n"
            + candidate_scope
            + "\n"
            + chunk.text
            + "\n"
            + "\n".join(str(concept_id) for concept_id in candidate_ids)
        ).encode("utf-8")
    ).hexdigest()


def load_checkpoint(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("chunk_id"):
            records[str(record["chunk_id"])] = record
    return records


def append_checkpoint(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def aggregate_relations(relations: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for relation in relations:
        source = str(relation["source_concept_id"])
        target = str(relation["target_concept_id"])
        rel_type = str(relation["relation"])
        if rel_type in SYMMETRIC_RELATIONS and source > target:
            source, target = target, source
        key = (source, target, rel_type)
        entry = grouped.setdefault(
            key,
            {
                "source_concept_id": source,
                "target_concept_id": target,
                "relation": rel_type,
                "evidence": [],
                "source_chunks": set(),
                "confidences": [],
                "verification_statuses": set(),
            },
        )
        quote = str(relation.get("quote", ""))
        chunk_id = str(relation.get("chunk_id", ""))
        evidence_key = (chunk_id, normalize_text(quote))
        existing_keys = {
            (str(e.get("chunk_id", "")), normalize_text(str(e.get("quote", ""))))
            for e in entry["evidence"]
        }
        if evidence_key not in existing_keys:
            entry["evidence"].append(
                {
                    "chunk_id": chunk_id,
                    "quote": quote,
                    "confidence": clamp_confidence(relation.get("confidence")),
                    "verification_status": relation.get("verification_status", ""),
                    "verification_reason": relation.get("verification_reason", ""),
                }
            )
        if chunk_id:
            entry["source_chunks"].add(chunk_id)
        entry["confidences"].append(clamp_confidence(relation.get("confidence")))
        entry["verification_statuses"].add(str(relation.get("verification_status", "")))

    output: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        entry = grouped[key]
        confidences = entry.pop("confidences")
        entry["source_chunks"] = sorted(entry["source_chunks"])
        entry["verification_statuses"] = sorted(entry["verification_statuses"])
        entry["confidence_max"] = round(max(confidences), 4) if confidences else 0.0
        entry["confidence_mean"] = (
            round(statistics.mean(confidences), 4) if confidences else 0.0
        )
        entry["evidence_count"] = len(entry["evidence"])
        output.append(entry)
    return output


def recover_explicit_name_evidence(
    chunks: Sequence[CleanChunk],
    mapped_concept_ids: Set[str],
    evidence_by_concept: Mapping[str, Sequence[Mapping[str, Any]]],
    global_registry: GlobalRegistry,
) -> List[Dict[str, Any]]:
    """Recover only deterministic, quote-verifiable evidence missed by the LLM.

    This is intentionally not semantic inference: a missing Concept is added
    only when a canonical name or registered alias occurs in a retained source
    Chunk and the selected sentence can pass the same evidence-anchor check.
    It raises evidence coverage without allowing an uncovered registry item to
    become a formal graph node merely because it looks relevant.
    """
    recovered: List[Dict[str, Any]] = []
    missing_ids = sorted(mapped_concept_ids - set(evidence_by_concept))
    for concept_id in missing_ids:
        concept = global_registry.get(concept_id)
        terms = [term for term in concept_terms(concept) if len(normalize_name(str(term))) >= 4]
        for chunk in chunks:
            if not concept_lexically_present(concept, chunk.text):
                continue
            sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n\s*\n", chunk.text) if part.strip()]
            quote = next(
                (part for part in sentences if concept_lexically_present(concept, part)),
                "",
            )
            if not quote:
                continue
            anchor, overlap = evidence_anchor(concept, quote, "mention")
            if not anchor:
                continue
            recovered.append({
                "concept_id": concept_id,
                "quote": quote[:700],
                "evidence_type": "mention",
                "matches_definition": False,
                "confidence": 0.80,
                "anchor_type": anchor,
                "definition_overlap": round(overlap, 4),
                "evidence_origin": "deterministic_explicit_name_recovery",
                "chunk_id": chunk.chunk_id,
                "pages": chunk.pages,
                "primary_section": chunk.primary_section,
                "quote_validated": True,
            })
            break
    return recovered


def recover_missing_evidence_with_targeted_pass(
    model: Any,
    chunks: Sequence[CleanChunk],
    missing_concept_ids: Set[str],
    local_registry: KPRegistry,
    global_registry: GlobalRegistry,
    alignment: Mapping[str, Any],
    max_candidates: int,
    retries: int,
    min_relation_confidence: float,
    max_calls: int,
) -> List[Dict[str, Any]]:
    """A bounded second pass for missing Concept--Chunk evidence only.

    Unlike the main extraction pass this prompt cannot emit relations or schema
    gaps.  Every returned quote is validated by the normal evidence validator,
    so the coverage target never permits a guessed node into the formal graph.
    """
    remaining = set(missing_concept_ids)
    recovered: List[Dict[str, Any]] = []
    calls = 0
    for chunk in chunks:
        if not remaining or calls >= max_calls:
            break
        candidates = candidate_global_concepts_for_chunk(
            chunk, local_registry, global_registry, alignment, max_candidates
        )
        focus = [item for item in candidates if str(item["concept_id"]) in remaining][:6]
        if not focus:
            continue
        prompt = """Return exactly one JSON object. You are performing an evidence-only recovery pass.
For each supplied Concept, return an evidence item only when the supplied textbook Chunk explicitly supports it.
Quote the exact shortest source sentence or phrase. Do not infer from surrounding knowledge.
Do not output relations, new concepts, or items without an exact quote.
JSON schema: {\"evidence\":[{\"concept_id\":\"...\",\"quote\":\"...\",\"evidence_type\":\"definition|example|usage|mention\",\"matches_definition\":true|false,\"confidence\":0.0}]}

Chunk:\n""" + chunk.text + "\n\nConcepts:\n" + format_global_concepts(focus)
        parsed, _ = call_json_model(model, prompt, retries)
        calls += 1
        if not parsed:
            continue
        validated = validate_extraction(
            {**parsed, "relations": [], "schema_gap_candidates": []},
            chunk, focus, global_registry, min_relation_confidence,
        )
        for item in validated["evidence"]:
            cid = str(item["concept_id"])
            if cid not in remaining:
                continue
            recovered.append({
                **item,
                "chunk_id": chunk.chunk_id,
                "pages": chunk.pages,
                "primary_section": chunk.primary_section,
                "quote_validated": True,
                "evidence_origin": "targeted_missing_concept_recovery",
            })
            remaining.remove(cid)
    return recovered


def semantic_relation_gate(
    relations: Sequence[Mapping[str, Any]],
    global_registry: GlobalRegistry,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate safe formal edges from evidence-backed but ambiguous edges.

    The review queue is deliberately retained in the JSON so a reviewer can
    inspect rejected semantics.  Only the first return value is eligible for
    Neo4j import and GraphRAG traversal.
    """
    formal: List[Dict[str, Any]] = []
    review: List[Dict[str, Any]] = []

    def move_to_review(relation: Mapping[str, Any], reason: str) -> None:
        review.append({
            **dict(relation),
            "review_status": "REVIEW_ONLY",
            "review_reason": reason,
        })

    def is_explicit_state(concept_id: str) -> bool:
        concept = global_registry.get(concept_id)
        labels = [concept.get("canonical_name", ""), *(concept.get("aliases") or [])]
        normalized = [normalize_name(str(label)) for label in labels]
        return any(
            re.search(r"\b(?:process|thread|task|execution)\s+state\b", label)
            or label in {"new", "ready", "running", "waiting", "blocked", "terminated"}
            for label in normalized
        )

    def is_explicit_registry_subtype(parent_id: str, child_id: str) -> bool:
        """Require both hierarchy metadata and an explicit subtype statement.

        Registry hierarchy metadata is useful but can itself be curated
        imperfectly (for example Program was once placed beneath Process).
        It may suppress a contrast only when the child definition actually
        says it is an implementation/type/form of the parent.
        """
        child = global_registry.get(child_id)
        if str(child.get("broader_concept_id") or "") != parent_id:
            return False
        parent_terms = [normalize_name(term) for term in concept_terms(global_registry.get(parent_id))]
        definition = normalize_search_text(str(child.get("short_definition") or ""))
        return any(
            term and re.search(
                rf"\b(?:implementation|type|subtype|kind|form)\s+of\s+(?:a\s+|an\s+)?{re.escape(term)}\b",
                definition,
            )
            for term in parent_terms
        )

    for relation in relations:
        current = dict(relation)
        source = str(current["source_concept_id"])
        target = str(current["target_concept_id"])
        rel_type = str(current["relation"])
        evidence_items = current.get("evidence") or []
        quote = "\n".join(
            str(item.get("quote", "")) for item in evidence_items if item.get("quote")
        )
        verification_statuses = {
            str(status).upper()
            for status in (current.get("verification_statuses") or [])
            if str(status).strip()
        }
        # Before aggregation a single verified status is present; afterwards
        # aggregate_relations stores the evidence statuses as a list.  Accept
        # either representation, but never promote an unverified relation.
        single_status = str(current.get("verification_status") or "").upper()
        if single_status:
            verification_statuses.add(single_status)
        verified_statuses = verification_statuses & VERIFIED_RELATION_STATUSES
        if not verified_statuses:
            move_to_review(current, "relation_not_semantically_verified")
            continue
        current["verification_status"] = sorted(verified_statuses)[0]

        if rel_type not in ALLOWED_RELATIONS:
            # In particular, PREREQUISITE_OF belongs to a separately curated
            # pedagogical layer and cannot leak in from legacy checkpoints.
            move_to_review(current, "relation_type_requires_reviewed_layer")
            continue
        if rel_type == "CAUSES" and _REDUCTION_CUE_RE.search(quote):
            # Preserve the direction expressed by the textbook rather than
            # flattening "reduces" into a generic causal edge.
            current["relation"] = "REDUCES"
            rel_type = "REDUCES"
        elif rel_type == "CAUSES" and _INCREASE_CUE_RE.search(quote):
            current["relation"] = "INCREASES"
            rel_type = "INCREASES"
        if rel_type == "HAS_SUBTYPE":
            # A hierarchy edge must agree with the independently curated
            # broader_concept_id.  Model-only subtype claims are review-only.
            if global_registry.get(target).get("broader_concept_id") != source:
                move_to_review(current, "subtype_not_confirmed_by_global_hierarchy")
                continue
        elif rel_type == "CAUSES" and not _CAUSAL_CUE_RE.search(quote):
            move_to_review(current, "cause_quote_lacks_explicit_causal_language")
            continue
        elif rel_type == "REDUCES" and not _REDUCTION_CUE_RE.search(quote):
            move_to_review(current, "reduces_quote_lacks_explicit_decrease_language")
            continue
        elif rel_type == "INCREASES" and not _INCREASE_CUE_RE.search(quote):
            move_to_review(current, "increases_quote_lacks_explicit_increase_language")
            continue
        elif rel_type == "MANAGES" and not _MANAGEMENT_CUE_RE.search(quote):
            move_to_review(current, "manages_used_without_explicit_management_evidence")
            continue
        elif rel_type == "PROVIDES" and not _PROVISION_CUE_RE.search(quote):
            move_to_review(current, "provides_used_without_explicit_provision_evidence")
            continue
        elif rel_type == "ENABLES" and not _ENABLEMENT_CUE_RE.search(quote):
            move_to_review(current, "enables_used_without_explicit_enablement_evidence")
            continue
        elif rel_type == "RELATED_TO" and (
            float(current.get("confidence_max", 0.0) or 0.0) < 0.90
            or not quote
        ):
            move_to_review(current, "weak_related_to_is_not_a_formal_teaching_edge")
            continue
        elif rel_type == "TRANSITIONS_TO" and (
            not is_explicit_state(source)
            or not is_explicit_state(target)
            or not _STATE_TRANSITION_CUE_RE.search(quote)
            or not concept_lexically_present(global_registry.get(source), quote)
            or not concept_lexically_present(global_registry.get(target), quote)
        ):
            move_to_review(
                current,
                "state_transition_requires_two_explicit_state_nodes_and_transition_evidence",
            )
            continue
        elif rel_type == "CONTRASTED_WITH":
            if is_explicit_registry_subtype(source, target) or is_explicit_registry_subtype(target, source):
                move_to_review(current, "contrast_between_category_and_its_subtype")
                continue
        formal.append(current)

    # Specific predicates carry the same teaching meaning as a generic USES
    # edge, but without the ambiguity and extra traversal branch.  Keep the
    # specific edge and retain the suppressed generic edge in the audit queue.
    specific_types = {"INVOKES", "SCHEDULES", "CREATES", "LOADS_FROM"}
    specific_pairs = {
        (str(item["source_concept_id"]), str(item["target_concept_id"]))
        for item in formal if str(item["relation"]) in specific_types
    }
    if specific_pairs:
        retained: List[Dict[str, Any]] = []
        for item in formal:
            pair = (str(item["source_concept_id"]), str(item["target_concept_id"]))
            if str(item["relation"]) == "USES" and pair in specific_pairs:
                move_to_review(item, "generic_uses_shadowed_by_specific_relation")
            else:
                retained.append(item)
        formal = retained

    # No directional relation may appear in both directions automatically.
    relation_keys = {
        (str(item["source_concept_id"]), str(item["target_concept_id"]), str(item["relation"]))
        for item in formal
    }
    reciprocal_keys = {
        key for key in relation_keys
        if key[2] in DIRECTIONAL_RELATIONS and (key[1], key[0], key[2]) in relation_keys
    }
    if reciprocal_keys:
        kept: List[Dict[str, Any]] = []
        for item in formal:
            key = (str(item["source_concept_id"]), str(item["target_concept_id"]), str(item["relation"]))
            if key in reciprocal_keys:
                move_to_review(item, "reciprocal_directional_relation")
            else:
                kept.append(item)
        formal = kept

    deduped_review: List[Dict[str, Any]] = []
    seen_review: Set[Tuple[str, str, str, str]] = set()
    for item in review:
        key = (
            str(item["source_concept_id"]),
            str(item["target_concept_id"]),
            str(item["relation"]),
            str(item.get("review_reason", "")),
        )
        if key not in seen_review:
            seen_review.add(key)
            deduped_review.append(item)
    return formal, deduped_review


def aggregate_schema_gap_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge repeated missing-concept reports while retaining every source."""
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in candidates:
        name = " ".join(str(item.get("proposed_name", "")).split())
        category = " ".join(str(item.get("category", "concept")).split()).lower()
        if not name:
            continue
        key = (normalize_name(name), category)
        entry = grouped.setdefault(
            key,
            {
                "proposed_name": name,
                "category": category or "concept",
                "review_status": str(item.get("review_status") or "PENDING_SCHEMA_REVIEW"),
                "evidence": [],
                "source_chunks": set(),
            },
        )
        quote = str(item.get("quote", "")).strip()
        evidence_key = (str(item.get("chunk_id", "")), normalize_text(quote))
        existing = {
            (str(row.get("chunk_id", "")), normalize_text(str(row.get("quote", ""))))
            for row in entry["evidence"]
        }
        if evidence_key not in existing:
            entry["evidence"].append(
                {
                    "chunk_id": str(item.get("chunk_id", "")),
                    "pages": list(item.get("pages") or []),
                    "primary_section": str(item.get("primary_section", "")),
                    "quote": quote,
                    "rationale": str(item.get("rationale", "")),
                }
            )
        if item.get("chunk_id"):
            entry["source_chunks"].add(str(item["chunk_id"]))

    output: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        entry = grouped[key]
        entry["source_chunks"] = sorted(entry["source_chunks"])
        entry["evidence_count"] = len(entry["evidence"])
        output.append(entry)
    return output


def consolidate_schema_gap_queues(
    schema_candidates: Sequence[Mapping[str, Any]],
    asset_candidates: Sequence[Mapping[str, Any]],
    api_candidates: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply one final category/name decision across all Chunk-level queues.

    A model may call ``ps command`` a concept in one Chunk and an API/command
    in another.  The final artefact must contain one consistent item, not both.
    The same normalisation merges harmless suffix variants such as ``systemd``
    and ``systemd process`` for human review.
    """
    rows: List[Dict[str, Any]] = []
    for queue, label in ((schema_candidates, "schema"), (asset_candidates, "asset"), (api_candidates, "api")):
        rows.extend([{**dict(item), "_queue": label} for item in queue])

    def family(name: str) -> str:
        tokens = normalize_name(name, singularize=True).split()
        if tokens and tokens[-1] in {"process", "command", "service", "daemon"}:
            tokens = tokens[:-1]
        return " ".join(tokens)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[family(str(row.get("proposed_name") or ""))].append(row)
    schema_out: List[Dict[str, Any]] = []
    asset_out: List[Dict[str, Any]] = []
    api_out: List[Dict[str, Any]] = []
    for key, group in grouped.items():
        queues = {str(row["_queue"]) for row in group}
        target = "api" if "api" in queues else "asset" if "asset" in queues else "schema"
        for row in group:
            row.pop("_queue", None)
            row["consolidation_key"] = key
            row["consolidated_category"] = target
            (api_out if target == "api" else asset_out if target == "asset" else schema_out).append(row)
    return schema_out, asset_out, api_out


def load_curated_relation_overrides(
    path: Optional[str],
    formal_concept_ids: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load reviewed diagram relations without treating them as model output."""
    if not path:
        return [], []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = raw.get("relations", raw if isinstance(raw, list) else [])
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for item in rows:
        source = str(item.get("source_concept_id") or "")
        target = str(item.get("target_concept_id") or "")
        relation = str(item.get("relation") or "").upper()
        if source not in formal_concept_ids or target not in formal_concept_ids or relation not in ALLOWED_RELATIONS:
            rejected.append({**dict(item), "review_reason": "curated_relation_invalid_or_endpoint_has_no_formal_evidence"})
            continue
        accepted.append({
            **dict(item),
            "relation": relation,
            "verification_status": "CURATED_HUMAN_REVIEWED",
            "source_type": "curated_figure_relation",
            "confidence_max": 1.0,
            "confidence_mean": 1.0,
            "evidence_count": 1,
            "source_chunks": list(item.get("source_chunks") or []),
        })
    return accepted, rejected


def _has_directed_path(
    edges: Sequence[Mapping[str, Any]],
    start: str,
    goal: str,
    skip_key: Optional[Tuple[str, str]] = None,
) -> bool:
    """Return whether ``goal`` is reachable from ``start`` in a small DAG.

    This deliberately uses a local, dependency-free traversal instead of
    treating an LLM assertion as acyclic.  It is shared by the prerequisite
    cycle and directness checks below.
    """
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    for edge in edges:
        source = str(edge.get("source_concept_id") or "")
        target = str(edge.get("target_concept_id") or "")
        if not source or not target or (source, target) == skip_key:
            continue
        adjacency[source].add(target)

    pending = [start]
    visited: Set[str] = set()
    while pending:
        current = pending.pop()
        if current == goal:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(sorted(adjacency.get(current, set()) - visited))
    return False


def _prerequisite_candidate_pairs(
    formal_concept_ids: Set[str],
    relations: Sequence[Mapping[str, Any]],
    evidence_by_concept: Mapping[str, Sequence[Mapping[str, Any]]],
    candidates_per_concept: int,
) -> List[Dict[str, Any]]:
    """Create a bounded, auditable candidate set for pedagogical review.

    Co-occurrence and textbook relations are only *candidate generators*.
    They never become a prerequisite on their own.  This keeps the later LLM
    review focused while avoiding the invalid rule that textual order equals a
    learning dependency.
    """
    pair_scores: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def register(a: str, b: str, score: float, origin: str) -> None:
        if a == b or a not in formal_concept_ids or b not in formal_concept_ids:
            return
        key = tuple(sorted((a, b)))
        row = pair_scores.setdefault(
            key,
            {
                "concept_a_id": key[0],
                "concept_b_id": key[1],
                "score": 0.0,
                "origins": set(),
            },
        )
        row["score"] += score
        row["origins"].add(origin)

    # Existing formal textbook edges are strong evidence that the pair belongs
    # to the same explanatory neighbourhood, but their relation direction is
    # intentionally not reused as a prerequisite direction.
    for relation in relations:
        source = str(relation.get("source_concept_id") or "")
        target = str(relation.get("target_concept_id") or "")
        rel_type = str(relation.get("relation") or "").upper()
        register(source, target, 3.0, f"formal:{rel_type}")

    # Concepts evidenced in the same Chunk are possible local teaching pairs.
    # Limit each Chunk before forming pairs so a broad overview Chunk cannot
    # create a quadratic number of speculative prerequisite candidates.
    by_chunk: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for concept_id, evidence_items in evidence_by_concept.items():
        if concept_id not in formal_concept_ids:
            continue
        confidence_by_chunk: Dict[str, float] = {}
        for evidence in evidence_items:
            chunk_id = str(evidence.get("chunk_id") or "")
            if chunk_id:
                confidence_by_chunk[chunk_id] = max(
                    confidence_by_chunk.get(chunk_id, 0.0),
                    clamp_confidence(evidence.get("confidence")),
                )
        for chunk_id, confidence in confidence_by_chunk.items():
            by_chunk[chunk_id].append((concept_id, confidence))

    for chunk_id, rows in by_chunk.items():
        selected = sorted(rows, key=lambda item: (-item[1], item[0]))[:10]
        for index, (concept_a, _) in enumerate(selected):
            for concept_b, _ in selected[index + 1:]:
                register(concept_a, concept_b, 1.0, f"shared_chunk:{chunk_id}")

    # Rank a neighbour list per endpoint, then take the union.  The candidate
    # direction remains undecided until the pedagogical verifier evaluates it.
    neighbours: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    for row in pair_scores.values():
        a, b = str(row["concept_a_id"]), str(row["concept_b_id"])
        neighbours[a].append((float(row["score"]), b))
        neighbours[b].append((float(row["score"]), a))

    selected_keys: Set[Tuple[str, str]] = set()
    for concept_id, rows in neighbours.items():
        for _, neighbour in sorted(rows, key=lambda item: (-item[0], item[1]))[:candidates_per_concept]:
            selected_keys.add(tuple(sorted((concept_id, neighbour))))

    output: List[Dict[str, Any]] = []
    for key in sorted(selected_keys):
        row = pair_scores[key]
        output.append(
            {
                "concept_a_id": row["concept_a_id"],
                "concept_b_id": row["concept_b_id"],
                "candidate_score": round(float(row["score"]), 4),
                "candidate_origins": sorted(row["origins"]),
            }
        )
    return output


def _prerequisite_evidence_context(
    concept_id: str,
    evidence_by_concept: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[Dict[str, str]]:
    """Return a compact set of already validated Concept--Chunk evidence."""
    unique: Dict[Tuple[str, str], Dict[str, str]] = {}
    for item in evidence_by_concept.get(concept_id, []):
        chunk_id = str(item.get("chunk_id") or "")
        quote = " ".join(str(item.get("quote") or "").split())
        if not chunk_id or not quote:
            continue
        unique.setdefault((chunk_id, normalize_text(quote)), {
            "concept_id": concept_id,
            "chunk_id": chunk_id,
            "quote": quote[:900],
        })
    return list(unique.values())[:3]


def _prerequisite_pair_context(
    concept_a_id: str,
    concept_b_id: str,
    relations: Sequence[Mapping[str, Any]],
    evidence_by_concept: Mapping[str, Sequence[Mapping[str, Any]]],
    chunks_by_id: Mapping[str, CleanChunk],
) -> List[Dict[str, str]]:
    """Expose compact source context that can justify a relationship itself.

    Separate evidence for A and B proves that the nodes exist.  It does not
    prove that A should be taught before B.  This context therefore contains
    only a formal relation quote or a Chunk that explicitly mentions both
    concepts, allowing the verifier to ground the *dependency* claim.
    """
    contexts: List[Dict[str, str]] = []
    pair = {concept_a_id, concept_b_id}
    seen: Set[Tuple[str, str]] = set()
    for relation in relations:
        endpoints = {
            str(relation.get("source_concept_id") or ""),
            str(relation.get("target_concept_id") or ""),
        }
        if endpoints != pair:
            continue
        relation_type = str(relation.get("relation") or "")
        for evidence in relation.get("evidence") or []:
            chunk_id = str(evidence.get("chunk_id") or "")
            quote = " ".join(str(evidence.get("quote") or "").split())
            if chunk_id and quote and (chunk_id, normalize_text(quote)) not in seen:
                seen.add((chunk_id, normalize_text(quote)))
                contexts.append({
                    "chunk_id": chunk_id,
                    "quote": quote[:1100],
                    "origin": f"formal_relation:{relation_type}",
                })

    chunks_a = {
        str(item.get("chunk_id") or "")
        for item in evidence_by_concept.get(concept_a_id, [])
        if str(item.get("chunk_id") or "")
    }
    chunks_b = {
        str(item.get("chunk_id") or "")
        for item in evidence_by_concept.get(concept_b_id, [])
        if str(item.get("chunk_id") or "")
    }
    for chunk_id in sorted(chunks_a & chunks_b):
        chunk = chunks_by_id.get(chunk_id)
        if not chunk:
            continue
        contexts.append({
            "chunk_id": chunk_id,
            "quote": chunk.text[:1800],
            "origin": "shared_concept_evidence_chunk",
        })
        if len(contexts) >= 3:
            break
    return contexts[:3]


def build_verified_pedagogical_prerequisites(
    model: Any,
    formal_concept_ids: Set[str],
    relations: Sequence[Mapping[str, Any]],
    evidence_by_concept: Mapping[str, Sequence[Mapping[str, Any]]],
    chunks_by_id: Mapping[str, CleanChunk],
    global_registry: GlobalRegistry,
    candidates_per_concept: int,
    batch_size: int,
    min_confidence: float,
    max_per_dependent: int,
    retries: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """Build a separate, verified, one-hop pedagogical prerequisite layer.

    A relationship is accepted only when the verifier identifies one endpoint
    as a *direct* prerequisite for the other, returns source-grounded evidence
    for both concepts, exceeds the configured confidence, and survives cycle
    and transitive-edge checks.  Consequently this layer is safe for a
    distance-1 learning traversal but remains distinct from textbook semantic
    relations such as USES or PART_OF.
    """
    candidates = _prerequisite_candidate_pairs(
        formal_concept_ids,
        relations,
        evidence_by_concept,
        candidates_per_concept,
    )
    review: List[Dict[str, Any]] = []
    proposed: List[Dict[str, Any]] = []

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        payload = []
        for index, candidate in enumerate(batch):
            concept_a = global_registry.get(candidate["concept_a_id"])
            concept_b = global_registry.get(candidate["concept_b_id"])
            payload.append({
                "index": index,
                "concept_a": {
                    "concept_id": candidate["concept_a_id"],
                    "name": concept_a.get("canonical_name", ""),
                    "definition": concept_a.get("short_definition", ""),
                    "evidence": _prerequisite_evidence_context(candidate["concept_a_id"], evidence_by_concept),
                },
                "concept_b": {
                    "concept_id": candidate["concept_b_id"],
                    "name": concept_b.get("canonical_name", ""),
                    "definition": concept_b.get("short_definition", ""),
                    "evidence": _prerequisite_evidence_context(candidate["concept_b_id"], evidence_by_concept),
                },
                "candidate_origins": candidate["candidate_origins"],
                "relationship_context": _prerequisite_pair_context(
                    candidate["concept_a_id"],
                    candidate["concept_b_id"],
                    relations,
                    evidence_by_concept,
                    chunks_by_id,
                ),
            })

        prompt = """You are building a strictly one-hop pedagogical prerequisite layer for a textbook knowledge graph.

For each candidate pair, decide whether one concept is a DIRECT prerequisite for learning the other.  A direct prerequisite means teaching it first is normally necessary to understand or apply the dependent concept at this textbook level.  It is NOT enough that concepts merely co-occur, are sequential in prose, are both part of a larger topic, or are broadly related.  Do not create a relation for a generic background topic, a subtype/category pair, or a possible extension unless it is a genuinely direct learning dependency.

A cause/effect statement, an implementation relation, a comparison, or a taxonomy statement alone is NOT a direct prerequisite.  Reject pairs such as a category and one of its subtypes, a mechanism and a merely contrasting mechanism, or a performance metric and a parameter it affects.  If the relationship context does not contain a precise source statement that names both concepts and supports teaching one before the other, return NOT_DIRECT.

Only choose a direction between the supplied two IDs.  Cite the supplied, exact textbook evidence for BOTH endpoints.  Do not invent quotes or Chunk IDs.  Use NOT_DIRECT if unsure.

Return exactly one JSON object:
{"decisions":[{"index":0,"decision":"DIRECT_PREREQUISITE|NOT_DIRECT","prerequisite_concept_id":"... or empty","dependent_concept_id":"... or empty","confidence":0.0,"reason":"short pedagogical reason","relationship_evidence":{"chunk_id":"...","quote":"exact supplied quote naming both concepts"},"evidence":[{"concept_id":"...","chunk_id":"...","quote":"exact supplied quote"}]}]}

Candidate pairs:
""" + json.dumps(payload, ensure_ascii=False)
        parsed, error = call_json_model(model, prompt, retries)
        if not parsed:
            for candidate in batch:
                review.append({
                    **candidate,
                    "relation": "PREREQUISITE_OF",
                    "review_status": "REVIEW_ONLY",
                    "review_reason": "pedagogical_verifier_failure",
                    "error": error,
                })
            continue

        decisions_by_index: Dict[int, Mapping[str, Any]] = {}
        for decision in parsed.get("decisions") or []:
            if isinstance(decision, Mapping):
                try:
                    decisions_by_index[int(decision.get("index"))] = decision
                except (TypeError, ValueError):
                    pass

        for index, candidate in enumerate(batch):
            decision = decisions_by_index.get(index)
            if not decision or str(decision.get("decision") or "").upper() != "DIRECT_PREREQUISITE":
                review.append({
                    **candidate,
                    "relation": "PREREQUISITE_OF",
                    "review_status": "REVIEW_ONLY",
                    "review_reason": "not_a_direct_pedagogical_prerequisite",
                })
                continue

            taxonomy_or_non_prereq_origin = {
                "formal:HAS_SUBTYPE",
                "formal:EXAMPLE_OF",
                "formal:CONTRASTED_WITH",
            } & set(candidate["candidate_origins"])
            if taxonomy_or_non_prereq_origin:
                review.append({
                    **candidate,
                    "relation": "PREREQUISITE_OF",
                    "review_status": "REVIEW_ONLY",
                    "review_reason": "taxonomy_example_or_contrast_is_not_a_direct_prerequisite",
                })
                continue
            causal_only_origins = {
                item for item in candidate["candidate_origins"]
                if item.startswith("formal:")
            }
            if causal_only_origins and causal_only_origins <= {
                "formal:CAUSES", "formal:INCREASES", "formal:REDUCES",
            }:
                review.append({
                    **candidate,
                    "relation": "PREREQUISITE_OF",
                    "review_status": "REVIEW_ONLY",
                    "review_reason": "causal_or_quantitative_association_is_not_a_direct_prerequisite",
                })
                continue

            prerequisite = str(decision.get("prerequisite_concept_id") or "")
            dependent = str(decision.get("dependent_concept_id") or "")
            allowed_pair = {candidate["concept_a_id"], candidate["concept_b_id"]}
            confidence = clamp_confidence(decision.get("confidence"))
            if prerequisite == dependent or {prerequisite, dependent} != allowed_pair:
                review.append({
                    **candidate,
                    "relation": "PREREQUISITE_OF",
                    "review_status": "REVIEW_ONLY",
                    "review_reason": "verifier_returned_invalid_endpoint_or_direction",
                })
                continue
            if confidence < min_confidence:
                review.append({
                    **candidate,
                    "source_concept_id": prerequisite,
                    "target_concept_id": dependent,
                    "relation": "PREREQUISITE_OF",
                    "confidence": confidence,
                    "review_status": "REVIEW_ONLY",
                    "review_reason": "pedagogical_confidence_below_threshold",
                })
                continue

            relationship_evidence = decision.get("relationship_evidence") or {}
            relationship_chunk_id = str(relationship_evidence.get("chunk_id") or "")
            relationship_quote = " ".join(
                str(relationship_evidence.get("quote") or "").split()
            )
            if (
                relationship_chunk_id not in chunks_by_id
                or not relationship_quote
                or not quote_is_in_chunk(
                    relationship_quote, chunks_by_id[relationship_chunk_id].text
                )
                or not concept_lexically_present(
                    global_registry.get(prerequisite), relationship_quote
                )
                or not concept_lexically_present(
                    global_registry.get(dependent), relationship_quote
                )
            ):
                review.append({
                    **candidate,
                    "source_concept_id": prerequisite,
                    "target_concept_id": dependent,
                    "relation": "PREREQUISITE_OF",
                    "confidence": confidence,
                    "review_status": "REVIEW_ONLY",
                    "review_reason": "missing_validated_relationship_evidence_naming_both_endpoints",
                })
                continue

            evidence: List[Dict[str, str]] = []
            evidence_concepts: Set[str] = set()
            for item in decision.get("evidence") or []:
                if not isinstance(item, Mapping):
                    continue
                concept_id = str(item.get("concept_id") or "")
                chunk_id = str(item.get("chunk_id") or "")
                quote = " ".join(str(item.get("quote") or "").split())
                if concept_id not in allowed_pair or not quote or chunk_id not in chunks_by_id:
                    continue
                if not quote_is_in_chunk(quote, chunks_by_id[chunk_id].text):
                    continue
                anchor, _ = evidence_anchor(global_registry.get(concept_id), quote, "mention")
                if not anchor:
                    continue
                evidence.append({"concept_id": concept_id, "chunk_id": chunk_id, "quote": quote[:900]})
                evidence_concepts.add(concept_id)
            if not {prerequisite, dependent} <= evidence_concepts:
                review.append({
                    **candidate,
                    "source_concept_id": prerequisite,
                    "target_concept_id": dependent,
                    "relation": "PREREQUISITE_OF",
                    "confidence": confidence,
                    "review_status": "REVIEW_ONLY",
                    "review_reason": "missing_validated_evidence_for_both_prerequisite_endpoints",
                })
                continue

            proposed.append({
                "source_concept_id": prerequisite,
                "target_concept_id": dependent,
                "relation": "PREREQUISITE_OF",
                "confidence": round(confidence, 4),
                "verification_status": "PEDAGOGICALLY_VERIFIED",
                "verification_reason": " ".join(str(decision.get("reason") or "").split())[:700],
                "selection_method": "post_extraction_direct_pedagogical_prerequisite_verifier",
                "candidate_origins": candidate["candidate_origins"],
                "relationship_evidence": {
                    "chunk_id": relationship_chunk_id,
                    "quote": relationship_quote[:1100],
                },
                "source_chunks": sorted({item["chunk_id"] for item in evidence}),
                "evidence": evidence,
            })

    # Preserve the strongest source-grounded assertion for a directed pair.
    best_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in proposed:
        key = (str(item["source_concept_id"]), str(item["target_concept_id"]))
        previous = best_by_key.get(key)
        if previous is None or float(item["confidence"]) > float(previous["confidence"]):
            best_by_key[key] = item

    accepted: List[Dict[str, Any]] = []
    for item in sorted(
        best_by_key.values(),
        key=lambda row: (-float(row["confidence"]), str(row["source_concept_id"]), str(row["target_concept_id"])),
    ):
        source = str(item["source_concept_id"])
        target = str(item["target_concept_id"])
        if _has_directed_path(accepted, target, source):
            review.append({
                **item,
                "review_status": "REVIEW_ONLY",
                "review_reason": "would_create_pedagogical_prerequisite_cycle",
            })
            continue
        accepted.append(item)

    # A learner should receive a small, interpretable first-layer set rather
    # than every plausible foundation.  Keep the strongest direct relations
    # for each dependent Concept; retain the rest for audit instead of losing
    # the fact that they were considered.
    capped: List[Dict[str, Any]] = []
    kept_per_dependent: Counter = Counter()
    for item in sorted(
        accepted,
        key=lambda row: (-float(row["confidence"]), str(row["target_concept_id"]), str(row["source_concept_id"])),
    ):
        dependent = str(item["target_concept_id"])
        if kept_per_dependent[dependent] >= max_per_dependent:
            review.append({
                **item,
                "review_status": "REVIEW_ONLY",
                "review_reason": "exceeds_max_direct_prerequisites_per_dependent",
            })
            continue
        kept_per_dependent[dependent] += 1
        capped.append(item)

    # A direct-prerequisite layer must not keep A->C when A->B->C is already
    # retained.  Such an edge would cause the tutoring system to skip a level.
    direct: List[Dict[str, Any]] = []
    for item in capped:
        source = str(item["source_concept_id"])
        target = str(item["target_concept_id"])
        if _has_directed_path(capped, source, target, skip_key=(source, target)):
            review.append({
                **item,
                "review_status": "REVIEW_ONLY",
                "review_reason": "transitive_prerequisite_not_direct",
            })
        else:
            direct.append(item)

    stats = {
        "candidate_pairs": len(candidates),
        "proposed_direct_prerequisites": len(proposed),
        "accepted_direct_prerequisites": len(direct),
        "review_only_candidates": len(review),
    }
    return direct, review, stats


def validate_pedagogical_prerequisites(
    prerequisites: Sequence[Mapping[str, Any]],
    concepts: Sequence[Mapping[str, Any]],
    chunks: Sequence[CleanChunk],
    min_confidence: float,
) -> Dict[str, Any]:
    """Validate the dedicated one-hop learning layer before Neo4j import."""
    concept_set = {str(item.get("concept_id") or "") for item in concepts}
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    errors: List[str] = []
    seen: Set[Tuple[str, str]] = set()
    for item in prerequisites:
        source = str(item.get("source_concept_id") or "")
        target = str(item.get("target_concept_id") or "")
        key = (source, target)
        if source not in concept_set or target not in concept_set:
            errors.append("dangling prerequisite endpoint")
        if not source or source == target:
            errors.append("self prerequisite")
        if key in seen:
            errors.append("duplicate prerequisite")
        seen.add(key)
        if str(item.get("relation") or "") != "PREREQUISITE_OF":
            errors.append("invalid prerequisite relation type")
        if clamp_confidence(item.get("confidence")) < min_confidence:
            errors.append("prerequisite confidence below threshold")
        relationship_evidence = item.get("relationship_evidence") or {}
        relationship_chunk_id = str(relationship_evidence.get("chunk_id") or "")
        relationship_quote = str(relationship_evidence.get("quote") or "")
        if (
            relationship_chunk_id not in chunk_by_id
            or not quote_is_in_chunk(
                relationship_quote, chunk_by_id[relationship_chunk_id].text
            )
        ):
            errors.append("invalid prerequisite relationship evidence quote")
        evidence_concepts: Set[str] = set()
        for evidence in item.get("evidence") or []:
            chunk_id = str(evidence.get("chunk_id") or "")
            quote = str(evidence.get("quote") or "")
            concept_id = str(evidence.get("concept_id") or "")
            if chunk_id not in chunk_by_id or not quote_is_in_chunk(quote, chunk_by_id[chunk_id].text):
                errors.append("invalid prerequisite evidence quote")
            if concept_id in {source, target}:
                evidence_concepts.add(concept_id)
        if not {source, target} <= evidence_concepts:
            errors.append("prerequisite lacks evidence for both endpoints")

    for item in prerequisites:
        source = str(item.get("source_concept_id") or "")
        target = str(item.get("target_concept_id") or "")
        if _has_directed_path(prerequisites, target, source):
            errors.append("cyclic prerequisite layer")
            break
    return {
        "validation_status": "FAILED" if errors else "PASSED",
        "total": len(prerequisites),
        "errors": list(dict.fromkeys(errors)),
    }


def validate_final_graph(
    concepts: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    chunks: Sequence[CleanChunk],
    max_chunk_chars: int,
    mentions: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    concept_ids = [str(c["concept_id"]) for c in concepts]
    concept_set = set(concept_ids)
    duplicate_concept_ids = len(concept_ids) - len(concept_set)
    chunk_set = {str(chunk.chunk_id) for chunk in chunks}
    invalid_endpoints = []
    dangling_mentions = []
    self_loops = []
    invalid_relation_types = []
    duplicate_relations = 0
    relation_keys: Set[Tuple[str, str, str]] = set()

    # A MENTIONS edge is part of the formal Neo4j contract.  It must never
    # refer to an audit-only cross-chapter concept that was intentionally not
    # promoted to a formal Concept node in this chapter.
    for mention in mentions:
        concept_id = str(mention.get("concept_id", ""))
        chunk_id = str(mention.get("chunk_id", ""))
        if concept_id not in concept_set or chunk_id not in chunk_set:
            dangling_mentions.append((concept_id, chunk_id))

    for rel in relations:
        source = str(rel.get("source_concept_id", ""))
        target = str(rel.get("target_concept_id", ""))
        rel_type = str(rel.get("relation", ""))
        if source not in concept_set or target not in concept_set:
            invalid_endpoints.append((source, target, rel_type))
        if source == target:
            self_loops.append((source, rel_type))
        if rel_type not in ALLOWED_RELATIONS:
            invalid_relation_types.append(rel_type)
        key = (source, target, rel_type)
        if key in relation_keys:
            duplicate_relations += 1
        relation_keys.add(key)

    reciprocal_directional = sorted(
        {
            key
            for key in relation_keys
            if key[2] in DIRECTIONAL_RELATIONS and (key[1], key[0], key[2]) in relation_keys
        }
    )

    oversized = [c.chunk_id for c in chunks if len(c.text) > max_chunk_chars]
    # A normal fixed-size Chunk may end between sentences; that alone is not
    # corruption.  It is an error only if a known lossy end-matter truncation
    # produced the dangling ending.  The current section-aware chunker never
    # makes such a second text-only truncation.
    incomplete_sentence_chunks = [
        c.chunk_id
        for c in chunks
        if c.excluded_tail_chars and likely_incomplete_sentence(c.text)
    ]
    excluded_heading_chunks = [
        c.chunk_id for c in chunks if EXCLUDED_HEADING_RE.search(c.text)
    ]
    suspicious_fragment_starts = [
        c.chunk_id
        for c in chunks
        if re.match(r"^\s*(?:[,.;:]|(?:is|are|was|were|and|or|but|which|that)\b)", c.text, re.I)
    ]

    fatal = bool(
        duplicate_concept_ids
        or dangling_mentions
        or invalid_endpoints
        or self_loops
        or invalid_relation_types
        or duplicate_relations
        or reciprocal_directional
        or oversized
        or incomplete_sentence_chunks
    )
    return {
        "validation_status": "FAILED" if fatal else "PASSED",
        "duplicate_concept_ids": duplicate_concept_ids,
        "dangling_mentions": len(dangling_mentions),
        "invalid_relation_endpoints": len(invalid_endpoints),
        "self_loops": len(self_loops),
        "invalid_relation_types": len(invalid_relation_types),
        "duplicate_relation_keys": duplicate_relations,
        "reciprocal_directional_relations": len(reciprocal_directional),
        "oversized_chunks": oversized,
        "suspected_incomplete_sentence_chunks": incomplete_sentence_chunks,
        "excluded_heading_chunks_remaining": excluded_heading_chunks,
        "suspicious_fragment_start_chunks": suspicious_fragment_starts,
        "details": {
            "invalid_endpoints": invalid_endpoints[:20],
            "dangling_mentions": dangling_mentions[:20],
            "self_loops": self_loops[:20],
            "invalid_relation_types": invalid_relation_types[:20],
            "reciprocal_directional_relations": reciprocal_directional[:20],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a chapter graph using stable full-book concept IDs."
    )
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--kp", required=True, help="Chapter-local canonical KP v2 JSON")
    parser.add_argument(
        "--global-registry",
        required=True,
        help="OSC10E cumulative global concept registry JSON",
    )
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--mapping-overrides", default=None)
    parser.add_argument("--curated-relations", default=None, help="Reviewed relation JSON, e.g. diagram arrows")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--chunk-overlap", type=int, default=300)
    parser.add_argument("--min-chunk-chars", type=int, default=500)
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=2400,
        help="Hard post-chunking character limit",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=24,
        help="Maximum registry Concepts exposed to the extractor for one Chunk",
    )
    parser.add_argument(
        "--candidate-scope",
        choices=("chapter_scoped", "all_global_chunk_match"),
        default="chapter_scoped",
        help=(
            "chapter_scoped uses the chapter bridge as a candidate filter; "
            "all_global_chunk_match compares every Chunk against all APPROVED "
            "global Concepts and uses chapter metadata only as provenance."
        ),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--min-relation-confidence", type=float, default=0.82)
    parser.add_argument("--coverage-target", type=float, default=0.90)
    parser.add_argument("--coverage-recovery-max-calls", type=int, default=24)
    parser.add_argument("--no-coverage-recovery", action="store_true")
    parser.add_argument(
        "--skip-pedagogical-prerequisites",
        action="store_true",
        help="Do not build the separate verified direct-prerequisite layer",
    )
    parser.add_argument(
        "--prerequisite-candidates-per-concept",
        type=int,
        default=5,
        help="Maximum candidate teaching neighbours reviewed for one formal Concept",
    )
    parser.add_argument(
        "--prerequisite-batch-size",
        type=int,
        default=8,
        help="Candidate pairs evaluated in one prerequisite-verification model call",
    )
    parser.add_argument(
        "--min-prerequisite-confidence",
        type=float,
        default=0.85,
        help="Minimum verifier confidence for an importable direct prerequisite",
    )
    parser.add_argument(
        "--max-direct-prerequisites-per-concept",
        type=int,
        default=5,
        help="Maximum retained direct prerequisites for one dependent Concept",
    )
    parser.add_argument(
        "--no-relation-verification",
        action="store_true",
        help="Disable the second LLM relation-verification pass",
    )
    parser.add_argument(
        "--verification-failure-policy",
        choices=("drop", "keep"),
        default="drop",
        help="What to do when relation verification fails",
    )
    parser.add_argument(
        "--include-summary",
        action="store_true",
        help="Allow Summary/Exercises/Bibliography text to create evidence",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore existing checkpoint")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run alignment, chunking, and candidate statistics without calling Gemini",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_chunk_chars < args.min_chunk_chars:
        raise ValueError("--max-chunk-chars must be >= --min-chunk-chars")
    if args.chunk_overlap >= args.max_chunk_chars:
        raise ValueError("--chunk-overlap must be smaller than --max-chunk-chars")
    if not 0 <= args.min_relation_confidence <= 1:
        raise ValueError("--min-relation-confidence must be between 0 and 1")
    if not 0 < args.coverage_target <= 1:
        raise ValueError("--coverage-target must be in (0, 1]")
    if args.prerequisite_candidates_per_concept < 1:
        raise ValueError("--prerequisite-candidates-per-concept must be at least 1")
    if args.prerequisite_batch_size < 1:
        raise ValueError("--prerequisite-batch-size must be at least 1")
    if not 0 <= args.min_prerequisite_confidence <= 1:
        raise ValueError("--min-prerequisite-confidence must be between 0 and 1")
    if args.max_direct_prerequisites_per_concept < 1:
        raise ValueError("--max-direct-prerequisites-per-concept must be at least 1")

    pdf_path = Path(args.pdf)
    kp_path = Path(args.kp)
    global_path = Path(args.global_registry)
    for path in (pdf_path, kp_path, global_path):
        if not path.exists():
            raise FileNotFoundError(path)

    local_registry = load_kp_v2(str(kp_path))
    global_registry = GlobalRegistry(global_path)
    overrides = load_mapping_overrides(args.mapping_overrides)
    alignment = build_alignment(local_registry, global_registry, overrides)
    chapter_number = alignment["chapter_number"]

    filename_match = re.match(r"0*(\d+)_Chapter_", pdf_path.name, re.I)
    if filename_match and str(int(filename_match.group(1))) != chapter_number:
        raise ValueError(
            f"PDF filename chapter {filename_match.group(1)} does not match "
            f"KP registry chapter {chapter_number}"
        )

    print(
        f"[registry] local={len(local_registry.by_id)} KPs; "
        f"mapped={alignment['summary']['mapped_local_kps']}; "
        f"global nodes={alignment['summary']['unique_global_concepts']}; "
        f"candidates={alignment['summary']['unmapped_local_kps']}"
    )
    print(
        f"[candidate-scope] {args.candidate_scope}; "
        f"approved_global_pool={len(global_registry.by_id)}"
    )

    raw_chunks = chunk_pdf_by_sections(
        str(pdf_path),
        local_registry,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        min_chars=args.min_chunk_chars,
    )
    chunks, chunk_report = postprocess_chunks(
        raw_chunks,
        max_chunk_chars=args.max_chunk_chars,
        overlap=args.chunk_overlap,
        min_chars=args.min_chunk_chars,
        include_summary=args.include_summary,
    )
    print(
        f"[chunks] raw={chunk_report['raw_chunk_count']} -> final={len(chunks)}; "
        f"length={chunk_report['actual_length']['min']}-"
        f"{chunk_report['actual_length']['max']} chars"
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out.parent / f"{out.stem}_checkpoint.jsonl"
    alignment_path = out.parent / f"{out.stem}_alignment.json"
    quality_path = out.parent / f"{out.stem}_quality_report.json"
    chunk_results_path = out.parent / f"{out.stem}_chunk_results.jsonl"

    alignment_path.write_text(
        json.dumps(
            {
                "pipeline_version": PIPELINE_VERSION,
                "chapter": local_registry.chapter,
                "global_registry": global_path.name,
                **alignment,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.dry_run:
        print("\n[dry-run] First 15 chunks:")
        for chunk in chunks[:15]:
            candidates = candidate_global_concepts_for_chunk(
                chunk,
                local_registry,
                global_registry,
                alignment,
                args.max_candidates,
                args.candidate_scope,
            )
            print(
                f"  {chunk.chunk_id} sec={chunk.primary_section} "
                f"p.{chunk.page_start}-{chunk.page_end} chars={len(chunk.text)} "
                f"candidates={len(candidates)}"
            )
        print(f"[dry-run] alignment report: {alignment_path}")
        return 0

    model = init_gemini(
        model_name=args.model or settings.gemini_model,
        generation_config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
        },
    )

    checkpoint: Dict[str, Dict[str, Any]] = {}
    if not args.no_resume and not args.force:
        checkpoint = load_checkpoint(checkpoint_path)

    completed_results: Dict[str, Dict[str, Any]] = {}
    pending: List[CleanChunk] = []
    for chunk in chunks:
        expected = expected_fingerprint(
            chunk,
            local_registry,
            global_registry,
            alignment,
            args.max_candidates,
            args.candidate_scope,
        )
        previous = checkpoint.get(chunk.chunk_id)
        if previous and previous.get("fingerprint") == expected:
            completed_results[chunk.chunk_id] = previous
        else:
            pending.append(chunk)

    print(
        f"[extract] total={len(chunks)}, resumed={len(completed_results)}, "
        f"pending={len(pending)}, workers={args.workers}, "
        f"relation_verification={not args.no_relation_verification}"
    )
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                extract_chunk,
                model,
                chunk,
                local_registry,
                global_registry,
                alignment,
                max_candidates=args.max_candidates,
                retries=args.retries,
                min_relation_confidence=args.min_relation_confidence,
                relation_verification=not args.no_relation_verification,
                verification_failure_policy=args.verification_failure_policy,
                candidate_scope=args.candidate_scope,
            ): chunk
            for chunk in pending
        }
        finished = len(completed_results)
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "chunk_id": chunk.chunk_id,
                    "fingerprint": expected_fingerprint(
                        chunk,
                        local_registry,
                        global_registry,
                        alignment,
                        args.max_candidates,
                        args.candidate_scope,
                    ),
                    "candidate_ids": [],
                    "evidence": [],
                    "relations": [],
                    "rejected_evidence": [],
                    "rejected_relations": [],
                    "schema_gap_candidates": [],
                    "asset_candidates": [],
                    "api_candidates": [],
                    "rejected_schema_gap_candidates": [],
                    "errors": [f"unhandled_error: {exc}"],
                }
            completed_results[chunk.chunk_id] = result
            append_checkpoint(checkpoint_path, result)
            finished += 1
            print(
                f"  [{finished}/{len(chunks)}] {chunk.chunk_id}: "
                f"evidence={len(result.get('evidence', []))}, "
                f"relations={len(result.get('relations', []))}, "
                f"rejected={len(result.get('rejected_evidence', [])) + len(result.get('rejected_relations', []))}, "
                f"errors={len(result.get('errors', []))}"
            )
    print(f"[extract] completed in {time.time() - start_time:.1f}s")

    ordered_results = [completed_results[c.chunk_id] for c in chunks]
    chunk_by_id = {c.chunk_id: c for c in chunks}

    evidence_by_concept: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    evidence_types: Dict[str, Counter] = defaultdict(Counter)
    relations_flat: List[Dict[str, Any]] = []
    mentions: List[Dict[str, Any]] = []
    schema_gap_candidates: List[Dict[str, Any]] = []
    asset_candidates: List[Dict[str, Any]] = []
    api_candidates: List[Dict[str, Any]] = []
    rejected_schema_gap_candidates: List[Dict[str, Any]] = []
    seen_mentions: Set[Tuple[str, str, str, str]] = set()

    for result in ordered_results:
        chunk_id = str(result["chunk_id"])
        chunk = chunk_by_id[chunk_id]
        for evidence in result.get("evidence", []):
            cid = str(evidence["concept_id"])
            key = (
                cid,
                chunk_id,
                str(evidence["evidence_type"]),
                normalize_text(str(evidence["quote"])),
            )
            if key in seen_mentions:
                continue
            seen_mentions.add(key)
            item = {
                **dict(evidence),
                "chunk_id": chunk_id,
                "pages": chunk.pages,
                "primary_section": chunk.primary_section,
                "quote_validated": True,
            }
            evidence_by_concept[cid].append(item)
            evidence_types[cid][str(evidence["evidence_type"])] += 1
            mentions.append(item)
        for relation in result.get("relations", []):
            relations_flat.append({**dict(relation), "chunk_id": chunk_id})
        for key, destination in (
            ("schema_gap_candidates", schema_gap_candidates),
            ("asset_candidates", asset_candidates),
            ("api_candidates", api_candidates),
            ("rejected_schema_gap_candidates", rejected_schema_gap_candidates),
        ):
            for candidate in result.get(key, []):
                destination.append({
                    **dict(candidate),
                    "chunk_id": chunk_id,
                    "pages": chunk.pages,
                    "primary_section": chunk.primary_section,
                })

    mapped_global_ids = set(alignment["global_to_local"])
    all_global_chunk_match = args.candidate_scope == "all_global_chunk_match"
    schema_gap_candidates, asset_candidates, api_candidates = consolidate_schema_gap_queues(
        schema_gap_candidates, asset_candidates, api_candidates
    )
    schema_gap_candidates = aggregate_schema_gap_candidates(schema_gap_candidates)
    asset_candidates = aggregate_schema_gap_candidates(asset_candidates)
    api_candidates = aggregate_schema_gap_candidates(api_candidates)
    chapter_sections = set(local_registry.section_pages)
    # The local bridge is an extraction hint, not the full definition of this
    # chapter's evidence scope.  A global concept can legitimately carry a
    # CH03 evidence section even when its canonical owner is an earlier
    # chapter.  Keep this denominator visible instead of silently reporting
    # only the smaller local-KP coverage number.
    chapter_evidence_scope_ids = {
        cid
        for cid, concept in global_registry.by_id.items()
        if {
            str(concept.get("primary_section") or ""),
            *(str(s) for s in (concept.get("evidence_sections") or [])),
        }
        & chapter_sections
    }
    # In all-global mode, do not re-introduce a chapter-bridge bias after the
    # main pass.  Every formal Concept must originate from the per-Chunk
    # all-global comparison above.  The legacy deterministic recovery remains
    # useful only when the bridge is intentionally the expected scope.
    deterministic_recovery = (
        [] if all_global_chunk_match else recover_explicit_name_evidence(
            chunks, mapped_global_ids, evidence_by_concept, global_registry
        )
    )
    for item in deterministic_recovery:
        cid = str(item["concept_id"])
        key = (cid, str(item["chunk_id"]), str(item["evidence_type"]), normalize_text(str(item["quote"])))
        if key in seen_mentions:
            continue
        seen_mentions.add(key)
        evidence_by_concept[cid].append(item)
        evidence_types[cid][str(item["evidence_type"])] += 1
        mentions.append(item)

    target_count = math.ceil(len(mapped_global_ids) * args.coverage_target)
    targeted_recovery: List[Dict[str, Any]] = []
    missing_after_deterministic = mapped_global_ids - set(evidence_by_concept)
    if (
        not all_global_chunk_match
        and not args.no_coverage_recovery
        and len(evidence_by_concept.keys() & mapped_global_ids) < target_count
        and missing_after_deterministic
    ):
        targeted_recovery = recover_missing_evidence_with_targeted_pass(
            model, chunks, missing_after_deterministic, local_registry, global_registry,
            alignment, args.max_candidates, args.retries, args.min_relation_confidence,
            args.coverage_recovery_max_calls,
        )
        for item in targeted_recovery:
            cid = str(item["concept_id"])
            key = (cid, str(item["chunk_id"]), str(item["evidence_type"]), normalize_text(str(item["quote"])))
            if key not in seen_mentions:
                seen_mentions.add(key)
                evidence_by_concept[cid].append(item)
                evidence_types[cid][str(item["evidence_type"])] += 1
                mentions.append(item)

    # A PDF text layer can retain a figure caption but lose arrow direction.
    # Preserve these candidates for visual/manual verification instead of
    # inventing formal TRANSITIONS_TO edges from a diagram label alone.
    state_concept_ids = [
        cid for cid in mapped_global_ids
        if re.search(r"\b(?:process|thread|task)\s+state\b", normalize_name(global_registry.get(cid).get("canonical_name", "")))
    ]
    diagram_relation_review_candidates: List[Dict[str, Any]] = []
    for chunk in chunks:
        if not re.search(r"\b(?:figure|diagram)\s+\d+\.\d+\b", chunk.text, re.I):
            continue
        named_states = [cid for cid in state_concept_ids if concept_lexically_present(global_registry.get(cid), chunk.text)]
        if len(named_states) >= 2:
            diagram_relation_review_candidates.append({
                "candidate_relation": "TRANSITIONS_TO",
                "concept_ids": named_states,
                "chunk_id": chunk.chunk_id,
                "pages": chunk.pages,
                "primary_section": chunk.primary_section,
                "review_status": "REQUIRES_FIGURE_DIRECTION_REVIEW",
                "reason": "PDF text provides state labels but not reliably ordered diagram arrows",
                "source_preview": chunk.text[:700],
            })
    # An unmapped (cross-chapter) mention can only become a formal node when
    # its source explicitly supports the registry definition.  Bare mentions
    # are retained in `mentions` for auditability but cannot pollute retrieval.
    formal_concept_ids = (
        set(evidence_by_concept)
        if all_global_chunk_match
        else {
            cid
            for cid, evidence_items in evidence_by_concept.items()
            if cid in mapped_global_ids or any(item.get("matches_definition") for item in evidence_items)
        }
    )
    weak_cross_chapter_mentions = sorted(
        set(evidence_by_concept) - mapped_global_ids - formal_concept_ids
    )
    # Relations can only connect concepts that are formal nodes in this chapter output.
    relations_flat = [
        r
        for r in relations_flat
        if r["source_concept_id"] in formal_concept_ids
        and r["target_concept_id"] in formal_concept_ids
    ]
    aggregated_relations = aggregate_relations(relations_flat)
    relations, review_only_relations = semantic_relation_gate(
        aggregated_relations, global_registry
    )
    curated_relations, rejected_curated_relations = load_curated_relation_overrides(
        args.curated_relations, formal_concept_ids
    )
    existing_relation_keys = {
        (str(r["source_concept_id"]), str(r["target_concept_id"]), str(r["relation"]))
        for r in relations
    }
    relations.extend(
        relation for relation in curated_relations
        if (str(relation["source_concept_id"]), str(relation["target_concept_id"]), str(relation["relation"])) not in existing_relation_keys
    )
    review_only_relations.extend({**item, "review_status": "REVIEW_ONLY"} for item in rejected_curated_relations)

    concepts_out: List[Dict[str, Any]] = []
    for cid in sorted(formal_concept_ids):
        concept = dict(global_registry.get(cid))
        evidence = evidence_by_concept[cid]
        source_chunks = sorted({str(e["chunk_id"]) for e in evidence})
        evidence_pages = sorted(
            {
                int(page)
                for e in evidence
                for page in (e.get("pages") or [])
                if str(page).isdigit()
            }
        )
        strong = any(
            (
                e["evidence_type"] == "definition"
                and e.get("matches_definition")
                and e.get("confidence", 0) >= 0.75
            )
            or (
                e["evidence_type"] in {"example", "usage"}
                and e.get("confidence", 0) >= 0.80
            )
            for e in evidence
        )
        concept.update(
            {
                "local_kp_ids": alignment["global_to_local"].get(cid, []),
                "registry_action": "REUSE",
                "evidence_count": len(evidence),
                "evidence_types": dict(evidence_types[cid]),
                "has_strong_evidence": strong,
                "source_chunks": source_chunks,
                "chunk_count": len(source_chunks),
                "evidence_pages": evidence_pages,
                "chapter_evidence_sections": sorted(
                    (
                        {
                            str(concept.get("primary_section") or ""),
                            *(str(s) for s in (concept.get("evidence_sections") or [])),
                        }
                        & chapter_sections
                    )
                    - {""},
                    key=section_sort_key,
                ),
            }
        )
        concepts_out.append(concept)

    evidenced_mapped = mapped_global_ids & formal_concept_ids
    evidenced_chapter_scope = chapter_evidence_scope_ids & formal_concept_ids
    unmapped_formal = formal_concept_ids - mapped_global_ids  # lexical cross-chapter reuse
    extraction_errors = [
        {"chunk_id": r["chunk_id"], "errors": r.get("errors", [])}
        for r in ordered_results
        if r.get("errors")
    ]
    rejected_evidence_count = sum(len(r.get("rejected_evidence", [])) for r in ordered_results)
    rejected_relation_count = sum(len(r.get("rejected_relations", [])) for r in ordered_results)

    # Weak cross-chapter lexical matches are useful audit evidence, but they
    # are not formal retrieval nodes unless their text supports the registry
    # definition.  Keep them separate so Neo4j never receives a dangling
    # Chunk-[:MENTIONS]->Concept edge.
    formal_mentions = [
        item for item in mentions
        if str(item.get("concept_id") or "") in formal_concept_ids
    ]
    audit_only_mentions = [
        item for item in mentions
        if str(item.get("concept_id") or "") not in formal_concept_ids
    ]

    # Textbook semantic relations and pedagogical prerequisites deliberately
    # remain separate.  The latter is built only after formal Concept--Chunk
    # evidence has been fixed, so its verifier cannot invent a teaching path
    # for an unsupported or audit-only Concept.
    if args.skip_pedagogical_prerequisites:
        pedagogical_prerequisites: List[Dict[str, Any]] = []
        pedagogical_prerequisite_review_only: List[Dict[str, Any]] = []
        prerequisite_stats = {
            "candidate_pairs": 0,
            "proposed_direct_prerequisites": 0,
            "accepted_direct_prerequisites": 0,
            "review_only_candidates": 0,
            "skipped": True,
        }
    else:
        (
            pedagogical_prerequisites,
            pedagogical_prerequisite_review_only,
            prerequisite_stats,
        ) = build_verified_pedagogical_prerequisites(
            model=model,
            formal_concept_ids=formal_concept_ids,
            relations=relations,
            evidence_by_concept=evidence_by_concept,
            chunks_by_id=chunk_by_id,
            global_registry=global_registry,
            candidates_per_concept=args.prerequisite_candidates_per_concept,
            batch_size=args.prerequisite_batch_size,
            min_confidence=args.min_prerequisite_confidence,
            max_per_dependent=args.max_direct_prerequisites_per_concept,
            retries=args.retries,
        )
        prerequisite_stats["skipped"] = False
    prerequisite_validation = validate_pedagogical_prerequisites(
        pedagogical_prerequisites,
        concepts_out,
        chunks,
        args.min_prerequisite_confidence,
    )

    validation = validate_final_graph(
        concepts_out,
        relations,
        chunks,
        args.max_chunk_chars,
        mentions=formal_mentions,
    )
    validation["pedagogical_prerequisites"] = prerequisite_validation
    if prerequisite_validation["validation_status"] != "PASSED":
        validation["validation_status"] = "FAILED"
        validation.setdefault("details", {}).setdefault(
            "import_contract_errors", []
        ).extend(prerequisite_validation["errors"])
    # The formal importer must not receive unresolved local names or proposed
    # Concepts.  Preserve them in the audit artefact, but make this output
    # non-importable until the global registry has been updated and reviewed.
    validation["new_concept_candidates"] = len(alignment["new_concept_candidates"])
    validation["unresolved_local_kps"] = int(
        alignment["summary"].get("unmapped_local_kps") or 0
    )
    if (
        validation["new_concept_candidates"]
        or validation["unresolved_local_kps"]
    ):
        validation["validation_status"] = "FAILED"
        validation.setdefault("details", {}).setdefault(
            "import_contract_errors", []
        ).append("unresolved global-concept alignment")
    if extraction_errors and validation["validation_status"] == "PASSED":
        validation["validation_status"] = "WARN"
        validation["warning"] = "One or more chunks had extraction or verification errors"

    source_metadata = {
        "pdf": {
            "path": str(pdf_path),
            "filename": pdf_path.name,
            "sha256": sha256_file(pdf_path),
        },
        "local_kp_registry": {
            "path": str(kp_path),
            "filename": kp_path.name,
            "sha256": sha256_file(kp_path),
        },
        "global_concept_registry": {
            "path": str(global_path),
            "filename": global_path.name,
            "sha256": sha256_file(global_path),
            "schema_version": global_registry.raw.get("schema_version"),
        },
    }

    output = {
        "schema_version": PIPELINE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chapter": local_registry.chapter,
        "chapter_title": local_registry.chapter_title,
        "source": source_metadata,
        "identity_policy": {
            "formal_node_identity": "global concept_id",
            "candidate_scope": args.candidate_scope,
            "chapter_local_kp_id_is_identity": False,
            "unmapped_kp_policy": "NEW_CONCEPT_CANDIDATE only; not inserted as formal node",
            "structural_noise_edges": "SAME_TYPE_AS and SIBLING_IN_SECTION disabled",
            "cross_chapter_policy": (
                "Any approved global Concept may be selected only from a Chunk's own text; "
                "a validated source quote is required before it becomes a formal node"
                if all_global_chunk_match
                else "Only registry evidence_sections or definition-supported evidence may create formal cross-chapter nodes"
            ),
            "pedagogical_prerequisite_policy": (
                "Formal relation extraction never emits PREREQUISITE_OF. "
                "A separate post-extraction verifier may emit only evidence-grounded, "
                "confidence-gated, acyclic, non-transitive direct teaching prerequisites."
            ),
        },
        "chunk_config": {
            "requested_chunk_size": args.chunk_size,
            "requested_overlap": args.chunk_overlap,
            "min_chunk_chars": args.min_chunk_chars,
            "hard_max_chunk_chars": args.max_chunk_chars,
            "include_summary": args.include_summary,
            **chunk_report,
        },
        "alignment_summary": alignment["summary"],
        "new_concept_candidates": alignment["new_concept_candidates"],
        "concepts": concepts_out,
        "relations": relations,
        "review_only_relations": review_only_relations,
        "pedagogical_prerequisites": pedagogical_prerequisites,
        "pedagogical_prerequisite_review_only": pedagogical_prerequisite_review_only,
        "curated_relations_applied": curated_relations,
        "diagram_relation_review_candidates": diagram_relation_review_candidates,
        "mentions": formal_mentions,
        "audit_only_mentions": audit_only_mentions,
        "schema_gap_candidates": schema_gap_candidates,
        "asset_candidates": asset_candidates,
        "api_candidates": api_candidates,
        "rejected_schema_gap_candidates": rejected_schema_gap_candidates,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "parent_chunk_id": c.parent_chunk_id,
                "text": c.text,
                "pages": c.pages,
                "sections": c.sections,
                "primary_section": c.primary_section,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "text_length": len(c.text),
            }
            for c in chunks
        ],
        "coverage": {
            "mode": (
                "observed_all_global_chunk_evidence_no_predeclared_denominator"
                if all_global_chunk_match else "chapter_bridge_expected_concepts"
            ),
            "approved_global_candidate_pool": len(global_registry.by_id),
            "mapped_global_concepts": len(mapped_global_ids),
            "mapped_global_concepts_with_evidence": len(evidenced_mapped),
            "mapped_concept_coverage_rate": (
                None if all_global_chunk_match else round(
                    len(evidenced_mapped) / max(1, len(mapped_global_ids)), 4
                )
            ),
            "chapter_bridge_evidence_rate_diagnostic_only": round(
                len(evidenced_mapped) / max(1, len(mapped_global_ids)), 4
            ),
            "chapter_evidence_scope_concepts": len(chapter_evidence_scope_ids),
            "chapter_evidence_scope_concepts_with_evidence": len(
                evidenced_chapter_scope
            ),
            "chapter_evidence_scope_coverage_rate": round(
                len(evidenced_chapter_scope)
                / max(1, len(chapter_evidence_scope_ids)),
                4,
            ),
            "cross_chapter_lexical_concepts_with_evidence": len(unmapped_formal),
            "weak_cross_chapter_mentions_not_promoted": len(weak_cross_chapter_mentions),
            "audit_only_mentions": len(audit_only_mentions),
            "formal_nodes_in_chapter_output": len(concepts_out),
            "strong_evidence_nodes": sum(
                1 for c in concepts_out if c.get("has_strong_evidence")
            ),
            "deterministic_explicit_name_recoveries": len(deterministic_recovery),
            "coverage_target": None if all_global_chunk_match else args.coverage_target,
            "targeted_missing_concept_recoveries": len(targeted_recovery),
            "uncovered_mapped_concept_ids": (
                [] if all_global_chunk_match else sorted(mapped_global_ids - formal_concept_ids)
            ),
            "uncovered_chapter_bridge_concept_ids_diagnostic_only": (
                sorted(mapped_global_ids - formal_concept_ids)
                if all_global_chunk_match else []
            ),
        },
        "relation_summary": {
            "total": len(relations),
            "by_type": dict(Counter(r["relation"] for r in relations)),
            "review_only_total": len(review_only_relations),
            "curated_relation_total": len(curated_relations),
            "diagram_relation_review_candidates": len(diagram_relation_review_candidates),
            "review_only_by_reason": dict(
                Counter(r.get("review_reason", "unspecified") for r in review_only_relations)
            ),
            "ontology": sorted(ALLOWED_RELATIONS),
            "second_pass_verification": not args.no_relation_verification,
            "verification_failure_policy": args.verification_failure_policy,
            "pedagogical_prerequisites": {
                **prerequisite_stats,
                "minimum_confidence": args.min_prerequisite_confidence,
                "candidates_per_concept": args.prerequisite_candidates_per_concept,
                "batch_size": args.prerequisite_batch_size,
                "max_direct_prerequisites_per_concept": args.max_direct_prerequisites_per_concept,
                "validation_status": prerequisite_validation["validation_status"],
            },
        },
        "quality_summary": {
            "extraction_error_chunks": len(extraction_errors),
            "rejected_evidence": rejected_evidence_count,
            "rejected_relations": rejected_relation_count,
            "schema_gap_candidates": len(schema_gap_candidates),
            "asset_candidates": len(asset_candidates),
            "api_candidates": len(api_candidates),
            "rejected_schema_gap_candidates": len(rejected_schema_gap_candidates),
            "deterministic_explicit_name_recoveries": len(deterministic_recovery),
            "targeted_missing_concept_recoveries": len(targeted_recovery),
            "weak_cross_chapter_mentions_not_promoted": len(weak_cross_chapter_mentions),
            "pedagogical_prerequisites": len(pedagogical_prerequisites),
            "pedagogical_prerequisite_review_only": len(pedagogical_prerequisite_review_only),
            "unmapped_local_kps": alignment["summary"]["unmapped_local_kps"],
            "validation_status": validation["validation_status"],
        },
        "validation_summary": validation,
    }

    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    chunk_results_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in ordered_results),
        encoding="utf-8",
    )
    quality_report = {
        "pipeline_version": PIPELINE_VERSION,
        "output": str(out),
        "alignment": alignment["summary"],
        "chunk_report": chunk_report,
        "coverage": output["coverage"],
        "relation_summary": output["relation_summary"],
        "extraction_errors": extraction_errors,
        "rejected_evidence_count": rejected_evidence_count,
        "rejected_relation_count": rejected_relation_count,
        "validation": validation,
    }
    quality_path.write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print(f"Output: {out}")
    print(f"Alignment: {alignment_path}")
    print(f"Quality report: {quality_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Formal concepts: {len(concepts_out)}")
    print(f"Relations: {len(relations)}")
    print(
        "Direct pedagogical prerequisites: "
        f"{len(pedagogical_prerequisites)} "
        f"(review-only candidates={len(pedagogical_prerequisite_review_only)})"
    )
    if all_global_chunk_match:
        print(
            f"Observed all-global evidence: {len(formal_concept_ids)} formal Concepts; "
            f"chapter-bridge diagnostic={len(evidenced_mapped)}/{len(mapped_global_ids)} "
            f"({output['coverage']['chapter_bridge_evidence_rate_diagnostic_only'] * 100:.1f}%)"
        )
    else:
        print(
            f"Mapped concept coverage: {len(evidenced_mapped)}/{len(mapped_global_ids)} "
            f"({output['coverage']['mapped_concept_coverage_rate'] * 100:.1f}%)"
        )
    print(f"New-concept candidates: {alignment['summary']['unmapped_local_kps']}")
    print(f"Validation: {validation['validation_status']}")

    return 0 if validation["validation_status"] != "FAILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
