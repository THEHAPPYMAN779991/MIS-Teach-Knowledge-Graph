"""Evidence-grounded LLM audit for extracted knowledge graphs.

The agent only proposes changes. It never mutates Neo4j or source JSON files.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field

from config import settings
from pdf_processor.pdf_loader import PDFLoader
from pdf_processor.text_chunker import ChapterAwareChunker
from utils.gemini_client import generate_with_rotation, init_gemini


ALLOWED_ACTIONS = {
    "KEEP",
    "REMOVE",
    "FLIP",
    "RETYPE",
    "KEEP_MULTIPLE",
    "KEEP_SEPARATE",
    "MERGE",
    "REMOVE_ALIAS",
    "REASSIGN_ALIAS",
    "IS_A",
    "KEEP_SOURCE",
    "MARK_DERIVED",
    "REVIEW",
}

ALLOWED_RELATIONS = {
    "IS_A",
    "HAS_COMPONENT",
    "HAS_FIELD",
    "USES",
    "PREREQ_OF",
    "CREATES_OR_RESULTS_IN",
    "API_MEMBER_OF",
    "RELATED_TO",
}


class AuditDecision(BaseModel):
    item_id: str
    action: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    evidence_quote: str = ""
    # Some models return a list of proposed relation objects for one ambiguous
    # legacy edge. Keep this flexible and normalize it after validation.
    new_relation: Any = None
    proposed_relations: List[Dict[str, Any]] = Field(default_factory=list)
    new_source: Optional[str] = None
    new_target: Optional[str] = None
    canonical: Optional[str] = None
    merge_names: List[str] = Field(default_factory=list)
    alias_removals: List[Dict[str, str]] = Field(default_factory=list)


class AuditResponse(BaseModel):
    decisions: List[AuditDecision] = Field(default_factory=list)


@dataclass
class GraphFiles:
    concepts: List[Dict[str, Any]]
    subtype_edges: List[Dict[str, Any]]
    prereq_edges: List[Dict[str, Any]]
    typed_relations: List[Dict[str, Any]]
    review_only: List[Dict[str, Any]]
    relation_quality: Dict[str, Any]
    alias_conflicts: Dict[str, Any]
    cycle_report: Dict[str, Any]
    derived_report: Dict[str, Any]


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_graph_files(json_dir: Path, book_id: str) -> GraphFiles:
    def file(suffix: str) -> Path:
        return json_dir / f"{book_id}_{suffix}.json"

    concepts = _load_json(file("concepts"), [])
    if not concepts:
        raise FileNotFoundError(file("concepts"))
    return GraphFiles(
        concepts=concepts,
        subtype_edges=_load_json(file("subtype_edges"), []),
        prereq_edges=_load_json(file("prereq_edges"), []),
        typed_relations=_load_json(file("typed_relations"), []),
        review_only=_load_json(file("review_only_relations"), []),
        relation_quality=_load_json(file("relation_quality_report"), {}),
        alias_conflicts=_load_json(file("alias_conflict_report"), {}),
        cycle_report=_load_json(file("cycle_report"), {}),
        derived_report=_load_json(file("derived_concept_report"), {}),
    )


def _stable_id(kind: str, payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{kind}:{digest}"


def build_audit_candidates(graph: GraphFiles, kinds: set[str]) -> List[Dict[str, Any]]:
    concepts = {str(item.get("name")): item for item in graph.concepts}
    candidates: List[Dict[str, Any]] = []

    if "relation" in kinds:
        pair_data: Dict[tuple[str, str], Dict[str, Any]] = {}

        def relation_pair(source: str, target: str) -> Dict[str, Any]:
            key = (source, target)
            if key not in pair_data:
                pair_data[key] = {
                    "kind": "relation",
                    "source": source,
                    "target": target,
                    "flags": [],
                    "current_relations": [],
                }
            return pair_data[key]

        for item in graph.review_only:
            source, target = str(item.get("source") or ""), str(item.get("target") or "")
            if source and target:
                row = relation_pair(source, target)
                row["flags"].append("review_only")
                row["current_relations"].append(item)

        dual_pairs = graph.relation_quality.get("dual_relation_pairs") or []
        for item in dual_pairs:
            source, target = str(item.get("source") or ""), str(item.get("target") or "")
            if source and target:
                row = relation_pair(source, target)
                row["flags"].append("multiple_relations")
                row["current_relations"].extend(
                    rel for rel in graph.typed_relations
                    if rel.get("source") == source and rel.get("target") == target
                )

        for group in graph.cycle_report.get("subtype_cycle_groups") or []:
            for source, target in group.get("edges") or []:
                row = relation_pair(str(source), str(target))
                row["flags"].append("subtype_cycle")
                row["current_relations"].extend(
                    rel for rel in graph.typed_relations
                    if rel.get("source") == source and rel.get("target") == target
                )

        for row in pair_data.values():
            row["flags"] = sorted(set(row["flags"]))
            unique_relations = {}
            for relation in row["current_relations"]:
                key = (
                    relation.get("source"),
                    relation.get("target"),
                    relation.get("relation"),
                    relation.get("origin"),
                )
                unique_relations[key] = relation
            row["current_relations"] = list(unique_relations.values())
            row["concepts"] = [
                concepts.get(row["source"], {"name": row["source"]}),
                concepts.get(row["target"], {"name": row["target"]}),
            ]
            row["item_id"] = _stable_id(
                "relation",
                {"source": row["source"], "target": row["target"], "flags": row["flags"]},
            )
            candidates.append(row)

    if "alias" in kinds:
        for conflict in graph.alias_conflicts.get("conflicts") or []:
            names = [str(name) for name in conflict.get("concepts") or []]
            row = {
                "kind": "alias_conflict",
                "alias": conflict.get("normalized_alias"),
                "raw_aliases": conflict.get("raw_aliases") or [],
                "risk": conflict.get("risk"),
                "concepts": [concepts.get(name, {"name": name}) for name in names],
            }
            row["item_id"] = _stable_id(
                "alias", {"alias": row["alias"], "concepts": names}
            )
            candidates.append(row)

    if "derived" in kinds:
        for item in graph.derived_report.get("candidates") or []:
            name = str(item.get("name") or "")
            row = {
                "kind": "derived_concept",
                "concept": concepts.get(name, {"name": name}),
                "markers": item.get("markers") or [],
            }
            row["item_id"] = _stable_id("derived", {"name": name})
            candidates.append(row)

    priority = {
        "subtype_cycle": 0,
        "multiple_relations": 1,
        "review_only": 2,
        "alias_conflict": 3,
        "derived_concept": 4,
    }

    def order(item: Dict[str, Any]) -> tuple:
        flags = item.get("flags") or []
        rank = min([priority.get(flag, 9) for flag in flags] + [priority.get(item["kind"], 9)])
        return rank, item["item_id"]

    return sorted(candidates, key=order)


class EvidenceIndex:
    def __init__(self, concepts: Sequence[Dict[str, Any]], chunks: Dict[str, str]):
        self.concepts = {str(item.get("name")): item for item in concepts}
        self.chunks = chunks
        self._chunk_items = list(chunks.items())

    @classmethod
    def from_pdfs(
        cls,
        concepts: Sequence[Dict[str, Any]],
        pdf_dir: Optional[Path],
        book_prefix: str,
        chapters: Optional[set[str]] = None,
    ) -> "EvidenceIndex":
        chunks: Dict[str, str] = {}
        if pdf_dir and pdf_dir.is_dir():
            chunker = ChapterAwareChunker(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            for pdf in sorted(pdf_dir.glob("*.pdf")):
                match = re.match(r"^(\d+)_", pdf.stem)
                if not match:
                    continue
                chapter = f"ch{int(match.group(1)):02d}"
                if chapters and chapter not in chapters:
                    continue
                document = PDFLoader(pdf, book_id=f"{book_prefix}_{chapter}").load()
                for chunk in chunker.chunk(document):
                    chunks[chunk.chunk_id] = chunk.text
        return cls(concepts, chunks)

    @staticmethod
    def _terms(concept: Dict[str, Any]) -> List[str]:
        values = [concept.get("name"), *(concept.get("aliases") or [])]
        result = []
        for value in values:
            text = str(value or "").strip()
            if len(text) >= 2 and text.casefold() not in {x.casefold() for x in result}:
                result.append(text)
        return result[:10]

    @staticmethod
    def _snippet(text: str, terms: Sequence[str], limit: int = 900) -> str:
        folded = text.casefold()
        positions = [folded.find(term.casefold()) for term in terms if term and folded.find(term.casefold()) >= 0]
        start = max(0, (min(positions) if positions else 0) - 180)
        return re.sub(r"\s+", " ", text[start : start + limit]).strip()

    def for_concept(self, name: str, max_chunks: int = 2) -> Dict[str, Any]:
        concept = self.concepts.get(name, {"name": name})
        terms = self._terms(concept)
        matches: List[tuple[int, str, str]] = []
        preferred = str(concept.get("chunk_id") or "")
        if preferred in self.chunks:
            text = self.chunks[preferred]
            matches.append((100, preferred, self._snippet(text, terms)))
        for chunk_id, text in self._chunk_items:
            if chunk_id == preferred:
                continue
            folded = text.casefold()
            score = sum(1 for term in terms if term.casefold() in folded)
            if score:
                matches.append((score, chunk_id, self._snippet(text, terms)))
        matches.sort(key=lambda item: (-item[0], item[1]))
        return {
            "name": name,
            "definition": concept.get("definition") or "",
            "category": concept.get("category") or "",
            "aliases": concept.get("aliases") or [],
            "source_chunks": [
                {"chunk_id": chunk_id, "text": snippet}
                for _, chunk_id, snippet in matches[:max_chunks]
            ],
        }

    def enrich(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(candidate)
        if candidate["kind"] == "relation":
            names = [candidate["source"], candidate["target"]]
        elif candidate["kind"] == "alias_conflict":
            names = [str(item.get("name")) for item in candidate.get("concepts") or []]
        else:
            names = [str((candidate.get("concept") or {}).get("name"))]
        result["evidence"] = [self.for_concept(name) for name in names if name]
        # Avoid repeating full concept records in prompts and candidate files.
        if candidate["kind"] == "alias_conflict":
            result["concepts"] = [
                {"name": item.get("name"), "category": item.get("category")}
                for item in candidate.get("concepts") or []
            ]
        elif candidate["kind"] == "relation":
            result.pop("concepts", None)
        return result


SYSTEM_PROMPT = """You are a conservative knowledge-graph audit agent for computer-science textbooks.
You receive suspicious graph items plus definitions and source excerpts.

Rules:
1. Never merge concepts merely because names or embeddings are similar.
2. Prefer KEEP_SEPARATE or REVIEW when evidence is incomplete.
3. IS_A means the target is a kind/specialization of the source.
4. PREREQ_OF means learning the source is pedagogically required before the target.
5. Ontology and pedagogy may both be valid, but only keep multiple relations when each has direct support.
6. evidence_quote must be a short exact substring from a supplied source chunk. Leave it empty if unavailable.
7. Do not invent concepts, quotes, or relations.

Actions by item kind:
- relation: KEEP, REMOVE, FLIP, RETYPE, KEEP_MULTIPLE, REVIEW
- alias_conflict: KEEP_SEPARATE, MERGE, REMOVE_ALIAS, REASSIGN_ALIAS, IS_A, REVIEW
- derived_concept: KEEP_SOURCE, MARK_DERIVED, MERGE, REMOVE, REVIEW

Allowed relation names: IS_A, HAS_COMPONENT, HAS_FIELD, USES, PREREQ_OF,
CREATES_OR_RESULTS_IN, API_MEMBER_OF, RELATED_TO.

Return JSON only:
{
  "decisions": [{
    "item_id": "exact input item_id",
    "action": "one allowed action",
    "confidence": 0.0,
    "reason": "brief reason",
    "evidence_quote": "exact supplied quote or empty",
    "new_relation": null,
    "proposed_relations": [{"source": "...", "target": "...", "relation": "..."}],
    "new_source": null,
    "new_target": null,
    "canonical": null,
    "merge_names": [],
    "alias_removals": [{"concept": "...", "alias": "..."}]
  }]
}
"""


class GraphAuditAgent:
    def __init__(self, model_name: Optional[str] = None):
        self.model = init_gemini(
            model_name=model_name,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
                "max_output_tokens": 8192,
            },
        )

    @staticmethod
    def _parse(raw: str) -> AuditResponse:
        text = (raw or "").strip()
        fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
        if fenced_blocks:
            text = "\n".join(fenced_blocks)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as original_error:
            # Vertex occasionally emits one valid JSON object per input item
            # instead of the requested wrapper. Decode each object safely.
            decoder = json.JSONDecoder()
            values: List[Any] = []
            cursor = 0
            while cursor < len(text):
                starts = [pos for pos in (text.find("{", cursor), text.find("[", cursor)) if pos >= 0]
                if not starts:
                    break
                start = min(starts)
                try:
                    value, cursor = decoder.raw_decode(text, start)
                except json.JSONDecodeError:
                    cursor = start + 1
                    continue
                values.append(value)
            if not values:
                raise original_error

            decisions: List[Dict[str, Any]] = []
            for value in values:
                if isinstance(value, list):
                    decisions.extend(
                        item
                        for item in value
                        if isinstance(item, dict) and item.get("item_id") and item.get("action")
                    )
                elif isinstance(value, dict) and isinstance(value.get("decisions"), list):
                    decisions.extend(
                        item
                        for item in value["decisions"]
                        if isinstance(item, dict) and item.get("item_id") and item.get("action")
                    )
                elif (
                    isinstance(value, dict)
                    and value.get("item_id")
                    and value.get("action")
                ):
                    decisions.append(value)
            payload = {"decisions": decisions}
        if isinstance(payload, list):
            payload = {"decisions": payload}
        elif isinstance(payload, dict) and payload.get("item_id") and payload.get("action"):
            payload = {"decisions": [payload]}
        if isinstance(payload, dict):
            raw_decisions = payload.get("decisions") or []
            payload["decisions"] = [
                item
                for item in raw_decisions
                if isinstance(item, dict) and item.get("item_id") and item.get("action")
            ]
        return AuditResponse.model_validate(payload)

    @staticmethod
    def _evidence_text(candidate: Dict[str, Any]) -> str:
        parts = []
        for item in candidate.get("evidence") or []:
            parts.append(str(item.get("definition") or ""))
            parts.extend(str(chunk.get("text") or "") for chunk in item.get("source_chunks") or [])
        return "\n".join(parts)

    @staticmethod
    def _quote_verified(quote: str, evidence: str) -> bool:
        if not quote:
            return False
        normalize = lambda value: re.sub(r"\s+", " ", value).strip().casefold()
        return normalize(quote) in normalize(evidence)

    def audit_batch(self, candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompt = SYSTEM_PROMPT + "\n\nINPUT ITEMS:\n" + json.dumps(
            list(candidates), ensure_ascii=False, indent=2
        )
        response = self._parse(generate_with_rotation(self.model, prompt))
        by_id = {candidate["item_id"]: candidate for candidate in candidates}
        decisions: List[Dict[str, Any]] = []
        for decision in response.decisions:
            if decision.item_id not in by_id:
                continue
            record = decision.model_dump()
            record["action"] = str(record.get("action") or "REVIEW").upper()
            if record["action"] not in ALLOWED_ACTIONS:
                record["action"] = "REVIEW"
            relation = record.get("new_relation")
            if isinstance(relation, list):
                proposed = []
                for item in relation:
                    if isinstance(item, dict):
                        proposed.append(item)
                    elif isinstance(item, str):
                        proposed.append({"relation": item})
                record["proposed_relations"] = proposed
                record["new_relation"] = None
                # Multiple replacements need a human to choose or approve all.
                record["action"] = "REVIEW"
                relation = None
            if relation and relation not in ALLOWED_RELATIONS:
                record["new_relation"] = None
                record["action"] = "REVIEW"
            evidence = self._evidence_text(by_id[decision.item_id])
            record["evidence_verified"] = self._quote_verified(
                str(record.get("evidence_quote") or ""), evidence
            )
            record["review_status"] = "pending"
            record["candidate_kind"] = by_id[decision.item_id]["kind"]
            record["candidate"] = by_id[decision.item_id]
            decisions.append(record)
        returned_ids = {item["item_id"] for item in decisions}
        for candidate in candidates:
            if candidate["item_id"] in returned_ids:
                continue
            decisions.append({
                "item_id": candidate["item_id"],
                "candidate_kind": candidate["kind"],
                "action": "REVIEW",
                "confidence": 0.0,
                "reason": "LLM returned no valid decision for this candidate.",
                "evidence_quote": "",
                "evidence_verified": False,
                "review_status": "pending",
                "candidate": candidate,
            })
        return decisions


def write_markdown_report(
    path: Path,
    candidates: Sequence[Dict[str, Any]],
    decisions: Sequence[Dict[str, Any]],
) -> None:
    action_counts = defaultdict(int)
    verified = 0
    for item in decisions:
        action_counts[item.get("action", "UNKNOWN")] += 1
        verified += int(bool(item.get("evidence_verified")))
    lines = [
        "# Graph Audit Report",
        "",
        f"- Candidates: {len(candidates)}",
        f"- Decisions: {len(decisions)}",
        f"- Evidence-verified decisions: {verified}",
        "- Safety: proposals only; no source JSON or Neo4j changes were made.",
        "",
        "## Action Counts",
        "",
    ]
    for action, count in sorted(action_counts.items()):
        lines.append(f"- {action}: {count}")
    lines.extend(["", "## Decisions", ""])
    for item in decisions:
        candidate = item.get("candidate") or {}
        label = candidate.get("source") or candidate.get("alias") or (
            candidate.get("concept") or {}
        ).get("name") or item.get("item_id")
        lines.extend([
            f"### {label}",
            "",
            f"- Item: `{item.get('item_id')}`",
            f"- Kind: `{item.get('candidate_kind')}`",
            f"- Action: `{item.get('action')}`",
            f"- Confidence: {item.get('confidence')}",
            f"- Evidence verified: {item.get('evidence_verified')}",
            f"- Reason: {item.get('reason') or ''}",
            f"- Quote: {item.get('evidence_quote') or '(none)'}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")
