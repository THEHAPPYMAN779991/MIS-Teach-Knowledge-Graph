#!/usr/bin/env python
"""Import supported reviewed global-registry GraphRAG JSON into Neo4j."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from neo4j import GraphDatabase

from config import settings
from kg_builder.graph_schema import IMPORT_RELATION_TYPES


ALLOWED_REL_TYPES = set(IMPORT_RELATION_TYPES)


def has_directed_path(edges, start, goal):
    """Small dependency-free cycle check for the dedicated prerequisite layer."""
    graph = defaultdict(set)
    for source, target in edges:
        graph[source].add(target)
    pending = [start]
    visited = set()
    while pending:
        current = pending.pop()
        if current == goal:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(graph.get(current, set()) - visited)
    return False


def batches(rows, size=200):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def validate(data):
    errors = []
    schema_version = str(data.get("schema_version", ""))
    if not schema_version.startswith((
        "kp-first-global-v3.2-reviewed",
        "kp-first-global-v3.3-reviewed",
        "kp-first-global-v3.4-all-global-candidate-scope-v1",
        "kp-first-global-v3.5-verified-pedagogical-prerequisites-v1",
        "kp-first-global-v3.6-import-contract-preflight-v1",
    )):
        errors.append("unsupported schema_version")

    report = data.get("validation_summary") or {}
    if report.get("validation_status") != "PASSED":
        errors.append("validation_status is not PASSED")
    if int(report.get("new_concept_candidates", 0) or 0):
        errors.append("new concept candidates remain")
    if int(report.get("unresolved_local_kps", 0) or 0):
        errors.append("unresolved local KPs remain")

    concepts = data.get("concepts") or []
    concept_ids = [str(x.get("concept_id") or "") for x in concepts]
    concept_set = set(concept_ids)
    if len(concept_ids) != len(concept_set):
        errors.append("duplicate concept_id")
    if any(not x.startswith("OS_") for x in concept_ids):
        errors.append("non-global concept_id found")

    chunks = data.get("chunks") or []
    chunk_ids = [str(x.get("chunk_id") or "") for x in chunks]
    chunk_set = set(chunk_ids)
    if len(chunk_ids) != len(chunk_set):
        errors.append("duplicate chunk_id")

    for m in data.get("mentions") or []:
        if str(m.get("concept_id") or "") not in concept_set:
            errors.append("dangling mention concept")
        if str(m.get("chunk_id") or "") not in chunk_set:
            errors.append("dangling mention chunk")

    relation_keys = set()
    for r in data.get("relations") or []:
        source = str(r.get("source_concept_id") or "")
        target = str(r.get("target_concept_id") or "")
        rel_type = str(r.get("relation") or "").upper()
        if source not in concept_set or target not in concept_set:
            errors.append("dangling relation endpoint")
        if source == target:
            errors.append("self relation")
        if rel_type not in ALLOWED_REL_TYPES:
            errors.append(f"unsupported relation type: {rel_type}")
        key = (source, rel_type, target)
        if key in relation_keys:
            errors.append("duplicate relation")
        relation_keys.add(key)

    # PREREQUISITE_OF is intentionally stored in a separate output field.  It
    # is only imported when it has passed the builder's evidence, confidence,
    # acyclicity and directness gates.  Keep this validation here as an
    # importer-side defence against edited or stale JSON.
    prerequisite_config = (
        (data.get("relation_summary") or {}).get("pedagogical_prerequisites") or {}
    )
    prerequisite_min_confidence = float(
        prerequisite_config.get("minimum_confidence", 0.85) or 0.85
    )
    prerequisite_edges = []
    prerequisite_keys = set()
    for r in data.get("pedagogical_prerequisites") or []:
        source = str(r.get("source_concept_id") or "")
        target = str(r.get("target_concept_id") or "")
        if source not in concept_set or target not in concept_set:
            errors.append("dangling pedagogical prerequisite endpoint")
        if not source or source == target:
            errors.append("self pedagogical prerequisite")
        if str(r.get("relation") or "") != "PREREQUISITE_OF":
            errors.append("invalid pedagogical prerequisite type")
        key = (source, target)
        if key in prerequisite_keys:
            errors.append("duplicate pedagogical prerequisite")
        prerequisite_keys.add(key)
        if float(r.get("confidence") or 0) < prerequisite_min_confidence:
            errors.append("pedagogical prerequisite below import confidence")
        relationship_evidence = r.get("relationship_evidence") or {}
        relationship_chunk = str(relationship_evidence.get("chunk_id") or "")
        relationship_quote = str(relationship_evidence.get("quote") or "")
        if relationship_chunk not in chunk_set or not relationship_quote:
            errors.append("invalid pedagogical prerequisite relationship evidence")
        evidenced_ids = set()
        for evidence in r.get("evidence") or []:
            evidence_concept = str(evidence.get("concept_id") or "")
            evidence_chunk = str(evidence.get("chunk_id") or "")
            quote = str(evidence.get("quote") or "")
            if evidence_chunk not in chunk_set or not quote:
                errors.append("invalid pedagogical prerequisite evidence")
            if evidence_concept in {source, target}:
                evidenced_ids.add(evidence_concept)
        if not {source, target} <= evidenced_ids:
            errors.append("pedagogical prerequisite lacks both endpoint evidence")
        prerequisite_edges.append(key)
    for source, target in prerequisite_edges:
        if has_directed_path(
            [edge for edge in prerequisite_edges if edge != (source, target)],
            target,
            source,
        ):
            errors.append("cyclic pedagogical prerequisite layer")
            break

    return list(dict.fromkeys(errors))


def ensure_schema(session, vector_dim=None):
    statements = [
        "CREATE CONSTRAINT book_id_unique IF NOT EXISTS "
        "FOR (b:Book) REQUIRE b.bookId IS UNIQUE",
        "CREATE CONSTRAINT chapter_id_unique IF NOT EXISTS "
        "FOR (c:Chapter) REQUIRE c.chapterId IS UNIQUE",
        "CREATE CONSTRAINT section_id_unique IF NOT EXISTS "
        "FOR (s:Section) REQUIRE s.sectionId IS UNIQUE",
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
        "FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE",
        "CREATE CONSTRAINT concept_global_id_unique IF NOT EXISTS "
        "FOR (c:Concept) REQUIRE c.conceptId IS UNIQUE",
    ]
    for statement in statements:
        session.run(statement).consume()

    if vector_dim:
        for statement in [
            f"""CREATE VECTOR INDEX concept_embedding_index IF NOT EXISTS
                FOR (c:Concept) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: {vector_dim},
                  `vector.similarity_function`: 'cosine'
                }}}}""",
            f"""CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: {vector_dim},
                  `vector.similarity_function`: 'cosine'
                }}}}""",
        ]:
            session.run(statement).consume()


def run_batch(session, query, rows, **params):
    total = 0
    for batch in batches(rows):
        record = session.run(query, rows=batch, **params).single()
        total += int(record["n"]) if record else 0
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--book-title", required=True)
    parser.add_argument("--chapter-id", required=True)
    parser.add_argument("--chapter-num", required=True)
    parser.add_argument("--chapter-title", required=True)
    parser.add_argument(
        "--database",
        default=getattr(settings, "neo4j_database", "neo4j"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--staging", action="store_true")
    parser.add_argument("--skip-vector", action="store_true")
    parser.add_argument("--vector-dim", type=int, default=384)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    errors = validate(data)
    if errors:
        print("Import refused:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2

    chapter_id = (
        args.chapter_id
        if args.chapter_id.startswith(args.book_id + "_")
        else f"{args.book_id}_{args.chapter_id}"
    )

    chunk_id_map = {}
    chunks = []
    for item in data.get("chunks") or []:
        raw_id = str(item["chunk_id"])
        full_id = f"{chapter_id}_{raw_id}"
        chunk_id_map[raw_id] = full_id
        section = str(item.get("primary_section") or "")
        pages = item.get("pages") or []
        chunks.append({
            "chunk_id": full_id,
            "raw_id": raw_id,
            "text": str(item.get("text") or ""),
            "page_start": int(item.get("page_start") or min(pages or [0])),
            "page_end": int(item.get("page_end") or max(pages or [0])),
            "section": section,
            "section_id": f"{chapter_id}_sec_{section}" if section else None,
        })

    sections = [
        {
            "section_id": f"{chapter_id}_sec_{section}",
            "section_num": section,
            "title": f"Section {section}",
        }
        for section in sorted({x["section"] for x in chunks if x["section"]})
    ]

    concepts = []
    for item in data.get("concepts") or []:
        concepts.append({
            "concept_id": str(item["concept_id"]),
            "name": str(item["canonical_name"]),
            "name_zh": str(item.get("name_zh") or ""),
            "definition": str(item.get("short_definition") or ""),
            "aliases": list(item.get("aliases") or []),
            "status": str(item.get("status") or "APPROVED"),
            "role": str(item.get("chapter_role") or ""),
            "retrieval_enabled": bool(item.get("retrieval_enabled", False)),
            "retrieval_weight": float(item.get("retrieval_weight") or 0),
            "evidence_count": int(item.get("evidence_count") or 0),
            "local_kp_ids": list(item.get("local_kp_ids") or []),
        })

    mentions = []
    for item in data.get("mentions") or []:
        mentions.append({
            "concept_id": str(item["concept_id"]),
            "chunk_id": chunk_id_map[str(item["chunk_id"])],
            "quote": str(item.get("quote") or "")[:2000],
            "evidence_type": str(item.get("evidence_type") or "mention"),
            "confidence": float(item.get("confidence") or 0),
            "chapter_id": chapter_id,
        })

    relation_groups = defaultdict(list)
    relation_items = list(data.get("relations") or []) + list(
        data.get("pedagogical_prerequisites") or []
    )
    for item in relation_items:
        rel_type = str(item["relation"]).upper()
        evidence = item.get("evidence") or []
        relation_groups[rel_type].append({
            "source": str(item["source_concept_id"]),
            "target": str(item["target_concept_id"]),
            "chapter_id": chapter_id,
            "source_chunk_ids": [
                chunk_id_map.get(str(x), str(x))
                for x in (item.get("source_chunks") or [])
            ],
            "evidence_texts": [
                str(x.get("quote") or "")
                for x in evidence
                if str(x.get("quote") or "")
            ],
            "evidence_json": json.dumps(evidence, ensure_ascii=False),
            "review_reason": str(item.get("review_reason") or ""),
            "confidence": float(item.get("confidence") or item.get("confidence_max") or 0),
            "verification_status": str(item.get("verification_status") or "VERIFIED"),
            "verification_reason": str(item.get("verification_reason") or ""),
            "selection_method": str(item.get("selection_method") or ""),
        })

    print("=" * 64)
    print("input      =", args.input)
    print("database   =", args.database)
    print("chapter    =", chapter_id)
    print("concepts   =", len(concepts))
    print("chunks     =", len(chunks))
    print("mentions   =", len(mentions))
    print("relations  =", sum(len(x) for x in relation_groups.values()))
    print("validation = PASSED")

    if args.dry_run:
        print("dry-run: no Neo4j writes")
        return 0

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )

    try:
        driver.verify_connectivity()
        with driver.session(database=args.database) as session:
            ensure_schema(
                session,
                None if args.skip_vector else args.vector_dim,
            )

            session.run(
                """
                MERGE (b:Book {bookId: $book_id})
                SET b.title = $book_title
                MERGE (ch:Chapter {chapterId: $chapter_id})
                SET ch.chapterNum = $chapter_num,
                    ch.title = $chapter_title,
                    ch.staging = $staging,
                    ch.updatedAt = datetime()
                MERGE (b)-[:HAS_CHAPTER]->(ch)
                """,
                book_id=args.book_id,
                book_title=args.book_title,
                chapter_id=chapter_id,
                chapter_num=args.chapter_num,
                chapter_title=args.chapter_title,
                staging=args.staging,
            ).consume()

            n_sections = run_batch(
                session,
                """
                UNWIND $rows AS row
                MERGE (s:Section {sectionId: row.section_id})
                SET s.sectionNum = row.section_num, s.title = row.title
                WITH s
                MATCH (ch:Chapter {chapterId: $chapter_id})
                MERGE (ch)-[:HAS_SECTION]->(s)
                RETURN count(s) AS n
                """,
                sections,
                chapter_id=chapter_id,
            )

            n_chunks = run_batch(
                session,
                """
                UNWIND $rows AS row
                MERGE (k:Chunk {chunkId: row.chunk_id})
                SET k.rawChunkId = row.raw_id,
                    k.text = row.text,
                    k.pageStart = row.page_start,
                    k.pageEnd = row.page_end,
                    k.chapter = $chapter_id,
                    k.section = row.section
                WITH k, row
                MATCH (ch:Chapter {chapterId: $chapter_id})
                MERGE (ch)-[:HAS_CHUNK]->(k)
                FOREACH (_ IN CASE WHEN row.section_id IS NULL THEN [] ELSE [1] END |
                    MERGE (s:Section {sectionId: row.section_id})
                    MERGE (s)-[:CONTAINS_CHUNK]->(k)
                )
                RETURN count(k) AS n
                """,
                chunks,
                chapter_id=chapter_id,
            )

            pairs = [
                {"a": chunks[i]["chunk_id"], "b": chunks[i + 1]["chunk_id"]}
                for i in range(len(chunks) - 1)
            ]
            n_next = run_batch(
                session,
                """
                UNWIND $rows AS row
                MATCH (a:Chunk {chunkId: row.a})
                MATCH (b:Chunk {chunkId: row.b})
                MERGE (a)-[:NEXT_CHUNK]->(b)
                RETURN count(*) AS n
                """,
                pairs,
            )

            n_concepts = run_batch(
                session,
                """
                UNWIND $rows AS row
                MERGE (c:Concept {conceptId: row.concept_id})
                ON CREATE SET c.createdAt = datetime(), c.name = row.name
                SET c.concept_id = row.concept_id,
                    c.canonicalName = row.name,
                    c.nameZh = row.name_zh,
                    c.definition = CASE
                      WHEN coalesce(c.definition, '') = ''
                      THEN row.definition ELSE c.definition END,
                    c.aliases = CASE
                      WHEN c.aliases IS NULL THEN row.aliases
                      ELSE c.aliases + [x IN row.aliases WHERE NOT x IN c.aliases] END,
                    c.status = row.status
                WITH c, row
                MATCH (ch:Chapter {chapterId: $chapter_id})
                MERGE (c)-[r:IN_CHAPTER]->(ch)
                SET r.role = row.role,
                    r.retrievalEnabled = row.retrieval_enabled,
                    r.retrievalWeight = row.retrieval_weight,
                    r.evidenceCount = row.evidence_count,
                    r.localKpIds = row.local_kp_ids
                RETURN count(c) AS n
                """,
                concepts,
                chapter_id=chapter_id,
            )

            n_mentions = run_batch(
                session,
                """
                UNWIND $rows AS row
                MATCH (k:Chunk {chunkId: row.chunk_id})
                MATCH (c:Concept {conceptId: row.concept_id})
                MERGE (k)-[r:MENTIONS]->(c)
                ON CREATE SET r.evidenceQuotes = [], r.evidenceTypes = []
                SET r.evidenceQuotes = CASE
                      WHEN row.quote IN r.evidenceQuotes THEN r.evidenceQuotes
                      ELSE r.evidenceQuotes + row.quote END,
                    r.evidenceTypes = CASE
                      WHEN row.evidence_type IN r.evidenceTypes THEN r.evidenceTypes
                      ELSE r.evidenceTypes + row.evidence_type END,
                    r.confidence = CASE
                      WHEN row.confidence > coalesce(r.confidence, 0)
                      THEN row.confidence ELSE r.confidence END,
                    r.chapterId = row.chapter_id
                MERGE (c)-[:MENTIONED_IN]->(k)
                RETURN count(r) AS n
                """,
                mentions,
            )

            n_relations = 0
            for rel_type, rows in relation_groups.items():
                n_relations += run_batch(
                    session,
                    f"""
                    UNWIND $rows AS row
                    MATCH (a:Concept {{conceptId: row.source}})
                    MATCH (b:Concept {{conceptId: row.target}})
                    MERGE (a)-[r:`{rel_type}`]->(b)
                    ON CREATE SET r.chapterIds = [],
                                  r.sourceChunkIds = [],
                                  r.evidenceTexts = []
                    SET r.chapterIds = CASE
                          WHEN row.chapter_id IN r.chapterIds THEN r.chapterIds
                          ELSE r.chapterIds + row.chapter_id END,
                        r.sourceChunkIds = reduce(
                          acc = r.sourceChunkIds, x IN row.source_chunk_ids |
                          CASE WHEN x IN acc THEN acc ELSE acc + x END),
                        r.evidenceTexts = reduce(
                          acc = r.evidenceTexts, x IN row.evidence_texts |
                          CASE WHEN x IN acc THEN acc ELSE acc + x END),
                        r.evidenceJson = row.evidence_json,
                        r.reviewReason = row.review_reason,
                        r.reviewStatus = 'APPROVED',
                        r.confidence = CASE
                          WHEN row.confidence > coalesce(r.confidence, 0)
                          THEN row.confidence ELSE r.confidence END,
                        r.verificationStatus = row.verification_status,
                        r.verificationReason = row.verification_reason,
                        r.selectionMethod = row.selection_method
                    RETURN count(r) AS n
                    """,
                    rows,
                )

        print("=" * 64)
        print("sections   =", n_sections)
        print("chunks     =", n_chunks)
        print("NEXT_CHUNK =", n_next)
        print("concepts   =", n_concepts)
        print("MENTIONS   =", n_mentions)
        print("relations  =", n_relations)
        print("Import completed.")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
