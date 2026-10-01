"""embed_kp_first_kg.py — 為 Concept + Chunk 產 embedding, 寫回 Neo4j

搭配 GeminiEmbedder (Vertex AI 優先, AI Studio fallback):
    - Concept: 用 "name. definition" 當 embedding source
    - Chunk:   用 chunk.text 當 embedding source

寫入 Neo4j 的 vector property:
    (:Concept).embedding
    (:Chunk).embedding

只算 embedding IS NULL 的節點 (idempotent)。

用法:
    python embed_kp_first_kg.py --chapter-id ch02
    python embed_kp_first_kg.py --chapter-id ch02 --targets concept
    python embed_kp_first_kg.py --chapter-id ch02 --targets chunk
    python embed_kp_first_kg.py --all-chapters
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from tqdm import tqdm

from config import settings
from graphrag.embedder import GeminiEmbedder


BATCH = 100


def resolve_chapter_id(session, requested_id: Optional[str]) -> Optional[str]:
    """Accept both ``ch03`` and the importer-created full Chapter ID.

    The importer scopes Chapter IDs as ``{book_id}_ch03``.  The old embedder
    silently accepted ``ch03`` but matched nothing, then incorrectly reported
    that no embeddings were needed.
    """
    if not requested_id:
        return None
    exact = session.run(
        "MATCH (ch:Chapter {chapterId: $id}) RETURN ch.chapterId AS id",
        id=requested_id,
    ).single()
    if exact:
        return str(exact["id"])
    rows = session.run(
        "MATCH (ch:Chapter) WHERE ch.chapterId ENDS WITH $suffix RETURN ch.chapterId AS id LIMIT 2",
        suffix="_" + requested_id,
    ).data()
    if len(rows) == 1:
        resolved = str(rows[0]["id"])
        print(f"[chapter] {requested_id} → {resolved}")
        return resolved
    if not rows:
        raise ValueError(f"Neo4j 找不到 Chapter: {requested_id}")
    raise ValueError(f"Chapter 簡寫不唯一: {requested_id}; 請使用完整 chapterId")


# ============================================================
# Concept
# ============================================================
def fetch_missing_concepts(session, chapter_id: Optional[str]) -> List[Dict[str, Any]]:
    if chapter_id:
        cypher = """
        MATCH (c:Concept)-[:IN_CHAPTER]->(ch:Chapter {chapterId: $chapter_id})
        WHERE c.embedding IS NULL
        RETURN c.name AS name, c.definition AS definition
        """
        rows = session.run(cypher, chapter_id=chapter_id).data()
    else:
        cypher = """
        MATCH (c:Concept) WHERE c.embedding IS NULL
        RETURN c.name AS name, c.definition AS definition
        """
        rows = session.run(cypher).data()
    return rows


def write_concept_embeddings(session, updates: List[Dict[str, Any]]) -> int:
    """updates = [{name, embedding: [float]}, ...]"""
    cypher = """
    UNWIND $rows AS row
    MATCH (c:Concept {name: row.name})
    CALL db.create.setNodeVectorProperty(c, 'embedding', row.embedding)
    RETURN count(c) AS n
    """
    n = 0
    for i in range(0, len(updates), BATCH):
        rec = session.run(cypher, rows=updates[i:i + BATCH]).single()
        n += int(rec["n"]) if rec else 0
    return n


def embed_concepts(embedder: GeminiEmbedder, session, chapter_id: Optional[str]) -> int:
    rows = fetch_missing_concepts(session, chapter_id)
    if not rows:
        print("[concept] 沒有需要 embedding 的 concept")
        return 0
    print(f"[concept] 找到 {len(rows)} 個需要 embedding")
    texts = [
        f"{r['name']}. {r.get('definition') or ''}".strip()
        for r in rows
    ]
    embeddings = embedder.embed_documents(texts)
    updates = [
        {"name": r["name"], "embedding": emb}
        for r, emb in zip(rows, embeddings)
    ]
    n = write_concept_embeddings(session, updates)
    print(f"[concept] 寫入 {n} 個 embedding")
    return n


# ============================================================
# Chunk
# ============================================================
def fetch_missing_chunks(session, chapter_id: Optional[str]) -> List[Dict[str, Any]]:
    if chapter_id:
        cypher = """
        MATCH (ch:Chapter {chapterId: $chapter_id})-[:HAS_CHUNK]->(k:Chunk)
        WHERE k.embedding IS NULL
        RETURN k.chunkId AS chunkId, k.text AS text
        """
        rows = session.run(cypher, chapter_id=chapter_id).data()
    else:
        cypher = """
        MATCH (k:Chunk) WHERE k.embedding IS NULL
        RETURN k.chunkId AS chunkId, k.text AS text
        """
        rows = session.run(cypher).data()
    return rows


def write_chunk_embeddings(session, updates: List[Dict[str, Any]]) -> int:
    cypher = """
    UNWIND $rows AS row
    MATCH (k:Chunk {chunkId: row.chunkId})
    CALL db.create.setNodeVectorProperty(k, 'embedding', row.embedding)
    RETURN count(k) AS n
    """
    n = 0
    for i in range(0, len(updates), BATCH):
        rec = session.run(cypher, rows=updates[i:i + BATCH]).single()
        n += int(rec["n"]) if rec else 0
    return n


def embed_chunks(embedder: GeminiEmbedder, session, chapter_id: Optional[str]) -> int:
    rows = fetch_missing_chunks(session, chapter_id)
    if not rows:
        print("[chunk] 沒有需要 embedding 的 chunk")
        return 0
    print(f"[chunk] 找到 {len(rows)} 個需要 embedding")
    texts = [(r.get("text") or "")[:8000] for r in rows]  # 限長
    embeddings = embedder.embed_documents(texts)
    updates = [
        {"chunkId": r["chunkId"], "embedding": emb}
        for r, emb in zip(rows, embeddings)
    ]
    n = write_chunk_embeddings(session, updates)
    print(f"[chunk] 寫入 {n} 個 embedding")
    return n


# ============================================================
# Main
# ============================================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--chapter-id", default=None, help="e.g. ch02; 省略則跑全 DB")
    p.add_argument("--all-chapters", action="store_true", help="跑所有章節 (等同不給 chapter-id)")
    p.add_argument(
        "--targets",
        choices=["both", "concept", "chunk"],
        default="both",
        help="embedding 對象",
    )
    args = p.parse_args()

    chapter_id = None if args.all_chapters else args.chapter_id

    settings.validate()
    embedder = GeminiEmbedder()
    print(f"[embedder] 後端: {embedder.backend}, model: {embedder._model_name}")
    print(f"[embedder] 輸出維度: {embedder.output_dim}")

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    try:
        with driver.session(database=settings.neo4j_database) as session:
            chapter_id = resolve_chapter_id(session, chapter_id)
            if args.targets in ("both", "concept"):
                embed_concepts(embedder, session, chapter_id)
            if args.targets in ("both", "chunk"):
                embed_chunks(embedder, session, chapter_id)
    finally:
        driver.close()

    print()
    print("✅ Embedding 完成. Neo4j vector index 現可用於 GraphRAG.")
    print("   驗證: MATCH (c:Concept) WHERE c.embedding IS NOT NULL RETURN count(c)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
