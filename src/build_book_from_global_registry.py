"""Build a section-aware GraphRAG chapter directly from the global registry.

The BOOK directory is the only source of truth supplied by the user:
  * 002_...pdf through 021_...pdf
  * OSC10E_CH02_CH21_Merged_Global_Concept_Registry.json

For each PDF, this script creates a chapter-local *bridge* file only for the
existing builder's section-aware chunker.  The bridge never creates a new
identity: every local record carries the original OS_* global_concept_id.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import fitz

# A chapter is not considered finished merely because the extraction script
# emitted JSON.  The downstream importer is the authoritative contract for
# Neo4j-safe data, so use the same validation before recording a build as
# completed.  This prevents a stale/output-schema mismatch from being found
# only after a costly full chapter conversion.
from ingest_global_registry_kg import validate as validate_neo4j_import_contract


_END_MATTER_COMPACT_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*)?(?:chapter)?summary|"
    r"bibliographicalnotes|furtherreading|bibliography|"
    r"(?:practice)?exercises?|programming(?:problems|projects|exercises)|"
    r"reviewquestions?$",
    re.I,
)


def _text_block_value(block: Dict[str, Any]) -> str:
    return "".join(
        str(span.get("text", ""))
        for line in block.get("lines", []) or []
        for span in line.get("spans", []) or []
    ).strip()


def detect_chapter_body_end(document: fitz.Document, first_body_page: int) -> Tuple[int, str]:
    """Return the page where actual Summary/end-matter begins.

    This is deliberately geometry-aware: the first textual ``3.9 Summary``
    may be a running header, while the genuine Summary title is lower on the
    page or typeset larger.  The returned page remains in the body range so
    the chunker can retain its preceding final paragraph and cut at the title.
    """
    for page_index in range(max(0, first_body_page - 1), len(document)):
        page = document[page_index]
        height = float(page.rect.height or 1.0)
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            value = _text_block_value(block)
            compact = re.sub(r"\s+", "", value)
            compact = re.sub(r"\d+$", "", compact)
            if not _END_MATTER_COMPACT_RE.fullmatch(compact):
                continue
            sizes = [
                float(span.get("size", 0) or 0)
                for line in block.get("lines", []) or []
                for span in line.get("spans", []) or []
            ]
            y0 = float((block.get("bbox") or (0, 0, 0, 0))[1])
            if max(sizes, default=0.0) >= 11.0 or y0 >= height * 0.18:
                return page_index + 1, " ".join(value.split())
    return len(document), ""

def section_key(value: str) -> Tuple[int, ...]:
    try:
        return tuple(int(part) for part in str(value).split("."))
    except ValueError:
        return (9999,)


def belongs_to_chapter(section: str, chapter: int) -> bool:
    try:
        return int(str(section).split(".", 1)[0]) == chapter
    except (AttributeError, ValueError):
        return False


def expected_sections(registry: Dict[str, Any], chapter: int) -> Set[str]:
    """All sections that can provide chapter evidence, including reused KPs."""
    result: Set[str] = set()
    for concept in registry.get("concepts") or []:
        for section in [concept.get("primary_section"), *(concept.get("evidence_sections") or [])]:
            section = str(section or "").strip()
            if belongs_to_chapter(section, chapter):
                result.add(section)
    return result


def detect_section_pages(pdf_path: Path, sections: Set[str]) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    """Map numbered section headings to inclusive page ranges in this PDF.

    The textbook's extracted text normally has a section number on one line
    and the title on the next, e.g. ``3.1\nProcess Concept``.  Some pages put
    both on one line, so both representations are accepted.
    """
    heading_re = re.compile(
        r"(?m)^\s*(?P<section>\d+\.\d+(?:\.\d+)*)\s*"
        r"(?:\n+\s*|[ \t]+)(?P<title>[A-Za-z][^\n]{0,160})$"
    )
    starts: Dict[str, int] = {}
    source: Dict[str, str] = {}
    # Keep *all* numbered body headings, not only headings represented by the
    # registry.  Otherwise a missing Global Registry section is invisible and
    # a chapter can be built with an apparently valid but empty subsection.
    all_heading_starts: Dict[str, int] = {}
    all_heading_titles: Dict[str, str] = {}
    outline_sections: Set[str] = set()

    document = fitz.open(pdf_path)
    try:
        # An outline is retained when available, but scanning the actual page
        # text is the authoritative fallback for PDFs without an outline.
        for _level, title, page in document.get_toc(simple=True):
            match = re.match(r"^\s*(\d+\.\d+(?:\.\d+)*)\b", str(title))
            section = match.group(1) if match else ""
            if section:
                outline_sections.add(section)
            if section in sections and int(page) > 0:
                starts.setdefault(section, int(page))
                source.setdefault(section, "pdf_outline")

        for page_index, page in enumerate(document):
            page_number = page_index + 1
            for match in heading_re.finditer(page.get_text("text", sort=True) or ""):
                section = match.group("section")
                all_heading_starts.setdefault(section, page_number)
                all_heading_titles.setdefault(section, " ".join(match.group("title").split()))
                if section not in sections:
                    continue
                if section not in starts or page_number < starts[section]:
                    starts[section] = page_number
                    source[section] = "heading_text"
        total_pages = len(document)
        first_body_page = min(starts.values()) if starts else 1
        chapter_body_end_page, body_end_heading = detect_chapter_body_end(
            document, first_body_page
        )
    finally:
        document.close()

    unique_starts = sorted(set(starts.values()))
    pages: Dict[str, List[int]] = {}
    for section, start_page in starts.items():
        later_starts = [page for page in unique_starts if page > start_page]
        natural_end = later_starts[0] - 1 if later_starts else chapter_body_end_page
        end_page = min(natural_end, chapter_body_end_page)
        pages[section] = list(range(start_page, max(start_page, end_page) + 1))

    expected = sorted(sections, key=section_key)

    def registry_covers_section(section: str) -> bool:
        """Whether a PDF heading is represented directly or by child KPs."""
        return section in sections or any(
            item.startswith(section + ".") for item in sections
        )

    def plausible_unregistered_section(section: str) -> bool:
        # A numbered parent heading (for example, 10.9 "Other
        # Considerations") can be a structural container rather than a
        # stand-alone knowledge-point section.  It is covered when the Global
        # Registry contains one or more of its child sections (10.9.1,
        # 10.9.2, ...).  Do this check before the heuristic below, otherwise
        # the parent is incorrectly reported as an unregistered body section.
        if registry_covers_section(section):
            return True
        if section in outline_sections:
            return True
        parts = section.split(".")
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            registered_top = [
                int(item.split(".")[1])
                for item in sections
                if len(item.split(".")) == 2
                and item.split(".")[0].isdigit()
                and item.split(".")[1].isdigit()
            ]
            return bool(registered_top) and int(parts[1]) <= max(registered_top) + 1
        parent = section.rsplit(".", 1)[0]
        return parent in sections or any(item.startswith(parent + ".") for item in sections)

    body_sections: List[str] = []
    ignored_numbered_references: List[str] = []
    for section, start_page in all_heading_starts.items():
        title = all_heading_titles.get(section, "")
        title_compact = re.sub(r"\d+$", "", re.sub(r"\s+", "", title))
        is_end_matter = bool(_END_MATTER_COMPACT_RE.fullmatch(title_compact))
        if is_end_matter:
            # A running header may make its first textual occurrence precede
            # the real Summary page.  It is never a body subsection.
            continue
        if start_page <= chapter_body_end_page and (
            section in sections or plausible_unregistered_section(section)
        ):
            body_sections.append(section)
        elif start_page <= chapter_body_end_page:
            ignored_numbered_references.append(section)
    body_sections = sorted(set(body_sections), key=section_key)
    report = {
        "pdf": str(pdf_path),
        "pdf_pages": total_pages,
        "expected_sections": expected,
        "detected_sections": sorted(pages, key=section_key),
        "missing_sections": [section for section in expected if section not in pages],
        "body_sections_detected_from_pdf": body_sections,
        "unregistered_body_sections": [
            section
            for section in body_sections
            if not registry_covers_section(section)
        ],
        "ignored_numbered_references": sorted(set(ignored_numbered_references), key=section_key),
        "source_by_section": source,
        "chapter_body_end_page": chapter_body_end_page,
        "chapter_body_end_heading": body_end_heading,
    }
    return pages, report


def attach_section_pages(kp_path: Path, registry_path: Path, pdf_path: Path, chapter: int) -> Dict[str, Any]:
    """Attach PDF provenance to the generated local bridge without new KPs."""
    global_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    pages, report = detect_section_pages(pdf_path, expected_sections(global_registry, chapter))
    bridge = json.loads(kp_path.read_text(encoding="utf-8"))
    bridge["source_pdf"] = str(pdf_path)

    for section in bridge.get("sections") or []:
        section_id = str(section.get("section") or "")
        page_range = list(pages.get(section_id) or [])
        section["source_pdf_pages"] = page_range
        for kp in section.get("knowledge_points") or []:
            kp["source_pdf_pages"] = page_range

    validation = bridge.setdefault("validation", {})
    validation["section_page_mapping"] = report
    validation["missing_primary_section_pages"] = [
        section.get("section")
        for section in bridge.get("sections") or []
        if not section.get("source_pdf_pages")
    ]
    kp_path.write_text(json.dumps(bridge, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def is_relevant_to_chapter(concept: Dict[str, Any], chapter: int) -> bool:
    """Return whether a globally canonical Concept has evidence in this PDF.

    ``canonical_owner_chapter`` is an identity/first-owner field, not a
    retrieval boundary.  A Chapter 4-owned ``Thread`` can be introduced or
    materially explained in Chapter 3, so it needs a local bridge record while
    Chapter 3 is being built.  The global ID still prevents duplicate nodes.
    """
    owner = str(concept.get("canonical_owner_chapter") or "").strip()
    if owner == str(chapter):
        return True
    if belongs_to_chapter(str(concept.get("primary_section") or ""), chapter):
        return True
    return any(
        belongs_to_chapter(str(section or ""), chapter)
        for section in (concept.get("evidence_sections") or [])
    )


def local_section_for_concept(concept: Dict[str, Any], chapter: int) -> str:
    """Choose a physical section in *this* PDF for a global concept.

    Canonical ownership and a concept's first appearance are deliberately
    different fields in the global registry.  A concept owned by Chapter 3 can
    have first been introduced in Chapter 2.  Such a concept must not be
    assigned to a non-existent ``2.x`` page range while building Chapter 3.
    """
    primary = str(concept.get("primary_section") or "").strip()
    if belongs_to_chapter(primary, chapter):
        return primary
    candidates = [
        str(section).strip()
        for section in (concept.get("evidence_sections") or [])
        if belongs_to_chapter(str(section).strip(), chapter)
    ]
    return sorted(set(candidates), key=section_key)[0] if candidates else ""


def make_local_kp_id(chapter: int, section: str, counters: Dict[str, int]) -> str:
    section_key_value = str(section or "0").replace(".", "_")
    counters[section_key_value] = counters.get(section_key_value, 0) + 1
    return f"CH{chapter:02d}_{section_key_value}_KP{counters[section_key_value]:03d}"


def build_bridge_from_global_registry(
    registry_path: Path,
    pdf_path: Path,
    chapter: int,
    output_path: Path,
) -> Dict[str, Any]:
    """Build the required canonical-KP-v2 bridge from the shared registry.

    This is deliberately mechanical: it copies only approved concepts and
    their existing global IDs.  It does not ask a model to invent, rename, or
    merge knowledge points.
    """
    global_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if global_registry.get("registry_type") != "CUMULATIVE_GLOBAL_CONCEPT_REGISTRY":
        raise ValueError("--global-registry must be a cumulative global concept registry")

    relevant_concepts = [
        item for item in (global_registry.get("concepts") or [])
        if item.get("status") == "APPROVED" and is_relevant_to_chapter(item, chapter)
    ]
    concepts = [
        (item, local_section_for_concept(item, chapter))
        for item in relevant_concepts
    ]
    skipped_owner_concepts = [
        str(item.get("concept_id") or "")
        for item, local_section in concepts if not local_section
    ]
    concepts = [(item, local_section) for item, local_section in concepts if local_section]
    if not concepts:
        raise ValueError(f"No approved global concepts belong to chapter {chapter}")

    all_sections = expected_sections(global_registry, chapter)
    pages_by_section, section_report = detect_section_pages(pdf_path, all_sections)
    by_section: Dict[str, List[Dict[str, Any]]] = {}
    for concept, section in concepts:
        by_section.setdefault(section, []).append(concept)

    processed = {
        int(item.get("number")): str(item.get("title") or "")
        for item in (global_registry.get("processed_chapters") or [])
        if str(item.get("number") or "").isdigit()
    }
    counters: Dict[str, int] = {}
    sections_out: List[Dict[str, Any]] = []
    empty_definitions: List[str] = []
    # Keep every physical section that appears in the shared registry, even
    # when this chapter owns no new KP in that section.  Those empty section
    # records preserve labels for cross-chapter concepts mentioned again here.
    for section in sorted(set(by_section) | all_sections, key=section_key):
        local_kps: List[Dict[str, Any]] = []
        for concept in sorted(by_section.get(section, []), key=lambda item: str(item.get("concept_id") or "")):
            local_id = make_local_kp_id(chapter, section, counters)
            aliases = list(concept.get("aliases") or [])
            name_zh = str(concept.get("name_zh") or "").strip()
            if name_zh and name_zh not in aliases:
                aliases.insert(0, name_zh)
            definition = str(concept.get("short_definition") or "")
            if not definition:
                empty_definitions.append(local_id)
            local_kps.append({
                "id": local_id,
                "name": str(concept.get("canonical_name") or ""),
                "aliases": aliases,
                "concept_type": str(concept.get("granularity") or "concept"),
                "definition": definition,
                "primary_section": section,
                "also_discussed_in": list(concept.get("evidence_sections") or []),
                "source_pdf_pages": list(pages_by_section.get(section) or []),
                "global_concept_id": str(concept.get("concept_id") or ""),
            })
        sections_out.append({
            "section": section,
            "title": f"Section {section}",
            "source_pdf_pages": list(pages_by_section.get(section) or []),
            "knowledge_points": local_kps,
        })

    missing_primary = [
        section["section"] for section in sections_out if not section["source_pdf_pages"]
    ]
    body_sections = set(section_report.get("body_sections_detected_from_pdf") or [])
    kp_counts = {section["section"]: len(section["knowledge_points"]) for section in sections_out}
    # Only leaf sections are required to have local KPs.  A parent heading can
    # legitimately be a label whose actual teaching material is in children.
    empty_leaf_kp_sections = [
        section
        for section in sorted(body_sections, key=section_key)
        if not any(other.startswith(section + ".") for other in body_sections)
        and kp_counts.get(section, 0) == 0
    ]
    bridge = {
        "chapter": f"CH{chapter:02d}",
        "chapter_title": processed.get(chapter, f"Chapter {chapter}"),
        "source_pdf": str(pdf_path),
        "schema_version": "canonical-knowledge-points-v2",
        "language": "en",
        "scope": {
            "included": "Chapter body; stable concept identities copied from the global registry",
            "excluded": ["Summary", "Exercises", "Bibliography"],
        },
        "canonicalization_policy": [
            "This bridge is generated from the cumulative global concept registry.",
            "global_concept_id is the only graph identity; chapter-local KP IDs are extraction hints.",
            "PDF heading text supplies source_pdf_pages for section-aware chunking.",
        ],
        "sections": sections_out,
        "validation": {
            "total_knowledge_points": sum(len(section["knowledge_points"]) for section in sections_out),
            "section_counts": {section["section"]: len(section["knowledge_points"]) for section in sections_out},
            "duplicate_ids": [],
            "duplicate_canonical_names": [],
            "empty_definitions": empty_definitions,
            "alias_to_other_canonical_collisions": [],
            "summary_nodes_created": False,
            "relations_included": False,
            "chunk_mappings_included": False,
            "section_page_mapping": section_report,
            "missing_primary_section_pages": missing_primary,
            "empty_leaf_kp_sections": empty_leaf_kp_sections,
            "unregistered_body_sections": list(section_report.get("unregistered_body_sections") or []),
            "skipped_owner_concepts_without_current_chapter_evidence": skipped_owner_concepts,
            "passed": not missing_primary
            and not empty_leaf_kp_sections
            and not section_report.get("unregistered_body_sections"),
        },
        "source_global_registry": str(registry_path),
    }
    output_path.write_text(json.dumps(bridge, ensure_ascii=False, indent=2), encoding="utf-8")
    return section_report


def parse_chapters(value: str) -> List[int]:
    result: List[int] = []
    for item in (value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(part.strip()) for part in item.split("-", 1))
            if start > end:
                raise ValueError(f"Invalid chapter range: {item}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(item))
    result = sorted(set(result))
    if not result:
        raise ValueError("--chapters must contain at least one chapter number")
    return result


def find_pdf(book_dir: Path, chapter: int) -> Path:
    matches = sorted(book_dir.glob(f"{chapter:03d}_*.pdf"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one PDF matching {chapter:03d}_*.pdf in {book_dir}; found {len(matches)}"
        )
    return matches[0]


def execute(command: Iterable[str]) -> None:
    values = list(command)
    printable = " ".join(f'"{value}"' if " " in value else value for value in values)
    print("[build]", printable)
    subprocess.run(values, check=True)


def validate_graph_output_for_import(graph_path: Path) -> None:
    """Fail the chapter build when its JSON would be refused by Neo4j ingest.

    The builder has its own quality report, but the importer additionally
    enforces referential integrity and evidence requirements for the dedicated
    pedagogical ``PREREQUISITE_OF`` layer.  Keeping this preflight here makes
    the generated artifact self-consistent with the next pipeline stage.
    """
    if not graph_path.is_file():
        raise FileNotFoundError(f"Builder did not create graph output: {graph_path}")
    try:
        graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Graph output is not valid JSON: {graph_path}") from exc
    errors = validate_neo4j_import_contract(graph_data)
    if errors:
        raise RuntimeError(
            "Neo4j import-contract preflight failed: " + "; ".join(errors)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build chapter GraphRAG JSON using BOOK PDFs and one global concept registry."
    )
    parser.add_argument("--book-dir", default="BOOK")
    parser.add_argument("--global-registry", default=None)
    parser.add_argument("--output-dir", default="outputs/global_registry_build")
    parser.add_argument("--chapters", default="2-21", help="For example: 3, 2-21, or 2,3,8")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--chunk-overlap", type=int, default=300)
    parser.add_argument("--min-chunk-chars", type=int, default=500)
    parser.add_argument("--max-chunk-chars", type=int, default=2400)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=24,
        help="Maximum scoped Concept candidates supplied to each Chunk extraction",
    )
    parser.add_argument(
        "--candidate-scope",
        choices=("chapter_scoped", "all_global_chunk_match"),
        default="chapter_scoped",
        help=(
            "Use all_global_chunk_match to select Concepts from the complete "
            "approved registry by each Chunk's own text rather than chapter metadata."
        ),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--coverage-target", type=float, default=0.90)
    parser.add_argument("--coverage-recovery-max-calls", type=int, default=24)
    parser.add_argument("--no-coverage-recovery", action="store_true")
    parser.add_argument(
        "--skip-pedagogical-prerequisites",
        action="store_true",
        help="Skip the separate evidence-verified direct-prerequisite layer.",
    )
    parser.add_argument(
        "--prerequisite-candidates-per-concept",
        type=int,
        default=5,
        help="Maximum prerequisite candidates reviewed for each formal Concept.",
    )
    parser.add_argument(
        "--prerequisite-batch-size",
        type=int,
        default=8,
        help="Candidate pairs evaluated together by the direct-prerequisite verifier.",
    )
    parser.add_argument(
        "--min-prerequisite-confidence",
        type=float,
        default=0.85,
        help="Minimum confidence for a Neo4j-importable direct prerequisite.",
    )
    parser.add_argument(
        "--max-direct-prerequisites-per-concept",
        type=int,
        default=5,
        help="Maximum retained direct prerequisites for one dependent Concept.",
    )
    parser.add_argument(
        "--curated-relations-dir",
        default=None,
        help="Directory containing optional chNN relation-override JSON files.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Check page mapping and chunking without calling Gemini")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-relation-verification", action="store_true")
    parser.add_argument("--allow-missing-section-pages", action="store_true")
    parser.add_argument(
        "--allow-empty-kp-sections",
        action="store_true",
        help="Permit a detected leaf body section with no approved registry Concept (audit-only escape hatch).",
    )
    parser.add_argument(
        "--allow-unregistered-sections",
        action="store_true",
        help="Permit numbered body headings absent from the Global Registry (audit-only escape hatch).",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    book_dir = Path(args.book_dir).resolve()
    registry_path = (
        Path(args.global_registry).resolve()
        if args.global_registry
        else (book_dir / "OSC10E_CH02_CH21_Merged_Global_Concept_Registry.json").resolve()
    )
    output_root = Path(args.output_dir).resolve()
    curated_relation_dir = (
        Path(args.curated_relations_dir).resolve()
        if args.curated_relations_dir else book_dir / "curated_relations"
    )
    builder = Path(__file__).with_name("build_kg_v2_corrected.py")
    if not book_dir.is_dir():
        raise FileNotFoundError(f"BOOK directory not found: {book_dir}")
    if not registry_path.is_file():
        raise FileNotFoundError(f"Global registry not found: {registry_path}")
    if not builder.is_file():
        raise FileNotFoundError(f"Builder not found: {builder}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "pipeline": "global-registry-chapter-builder-v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "book_dir": str(book_dir),
        "global_registry": str(registry_path),
        "chapters": [],
    }

    for chapter in parse_chapters(args.chapters):
        record: Dict[str, Any] = {"chapter": chapter}
        manifest["chapters"].append(record)
        chapter_dir = output_root / f"ch{chapter:02d}"
        chapter_dir.mkdir(parents=True, exist_ok=True)
        try:
            pdf_path = find_pdf(book_dir, chapter)
            kp_path = chapter_dir / f"CH{chapter:02d}_canonical_kps_from_global_registry.json"
            graph_path = chapter_dir / f"ch{chapter:02d}.json"
            print(f"\n{'=' * 72}\n[chapter {chapter:02d}] {pdf_path.name}")

            # Create the local section-aware bridge from stable global IDs.
            # No LLM is used here and no new concept identity can be created.
            section_report = build_bridge_from_global_registry(
                registry_path, pdf_path, chapter, kp_path
            )
            missing = section_report["missing_sections"]
            if missing and not args.allow_missing_section_pages:
                raise RuntimeError(
                    "Section headings were not fully detected. Refusing an ambiguous build; "
                    f"missing: {', '.join(missing)}"
                )
            bridge_validation = json.loads(kp_path.read_text(encoding="utf-8")).get("validation") or {}
            empty_leaf = bridge_validation.get("empty_leaf_kp_sections") or []
            unregistered = bridge_validation.get("unregistered_body_sections") or []
            all_global_chunk_match = args.candidate_scope == "all_global_chunk_match"
            if empty_leaf and not args.allow_empty_kp_sections and not all_global_chunk_match:
                raise RuntimeError(
                    "Detected body sections with no approved Global Registry Concept. "
                    "Refusing a silently incomplete build; empty leaf sections: "
                    f"{', '.join(empty_leaf)}"
                )
            if unregistered and not args.allow_unregistered_sections and not all_global_chunk_match:
                raise RuntimeError(
                    "PDF contains numbered body headings absent from the Global Registry. "
                    "Update the registry or use the explicit audit override; sections: "
                    f"{', '.join(unregistered)}"
                )

            command = [
                sys.executable, str(builder),
                "--pdf", str(pdf_path),
                "--kp", str(kp_path),
                "--global-registry", str(registry_path),
                "--output", str(graph_path),
                "--chunk-size", str(args.chunk_size),
                "--chunk-overlap", str(args.chunk_overlap),
                "--min-chunk-chars", str(args.min_chunk_chars),
                "--max-chunk-chars", str(args.max_chunk_chars),
                "--max-candidates", str(args.max_candidates),
                "--candidate-scope", args.candidate_scope,
                "--coverage-target", str(args.coverage_target),
                "--coverage-recovery-max-calls", str(args.coverage_recovery_max_calls),
                "--prerequisite-candidates-per-concept", str(args.prerequisite_candidates_per_concept),
                "--prerequisite-batch-size", str(args.prerequisite_batch_size),
                "--min-prerequisite-confidence", str(args.min_prerequisite_confidence),
                "--max-direct-prerequisites-per-concept", str(args.max_direct_prerequisites_per_concept),
                "--workers", str(args.workers),
            ]
            curated_relation_path = curated_relation_dir / f"ch{chapter:02d}_process_state_transitions.json"
            if curated_relation_path.is_file():
                command += ["--curated-relations", str(curated_relation_path)]
            if args.model:
                command += ["--model", args.model]
            if args.dry_run:
                command.append("--dry-run")
            if args.force:
                command.append("--force")
            if args.no_relation_verification:
                command.append("--no-relation-verification")
            if args.no_coverage_recovery:
                command.append("--no-coverage-recovery")
            if args.skip_pedagogical_prerequisites:
                command.append("--skip-pedagogical-prerequisites")
            execute(command)
            if not args.dry_run:
                validate_graph_output_for_import(graph_path)

            record.update({
                "status": "completed",
                "pdf": str(pdf_path),
                "kp_bridge": str(kp_path),
                "graph_output": str(graph_path),
                "section_pages_detected": len(section_report["detected_sections"]),
                "section_pages_missing": missing,
                "empty_leaf_kp_sections": empty_leaf,
                "unregistered_body_sections": unregistered,
                "neo4j_import_contract": "PASSED" if not args.dry_run else "NOT_RUN_DRY_RUN",
            })
        except Exception as exc:
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[chapter {chapter:02d}] FAILED: {record['error']}", file=sys.stderr)
            if args.stop_on_error:
                manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
                (output_root / "build_manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                return 2
        finally:
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            (output_root / "build_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    completed = sum(item.get("status") == "completed" for item in manifest["chapters"])
    print(f"\n[done] {completed}/{len(manifest['chapters'])} chapters completed")
    print("[manifest]", output_root / "build_manifest.json")
    return 0 if completed == len(manifest["chapters"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
