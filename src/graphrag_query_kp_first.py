"""graphrag_query_kp_first.py — KP-First GraphRAG 檢索 API

流程 (Hybrid Retrieval):
    1. Query text → embedding
    2. Vector search on Chunk.embedding (top K)
    3. Vector search on Concept.embedding (top M)
    4. 對命中 Chunk 走 MENTIONS → 找到相關 Concept
    5. 對命中 Concept 走 (PREREQUISITE_OF|HAS_SUBTYPE|USED_FOR|IMPLEMENTS|REQUIRES|...)
       擴展相關 Concept (max_hops)
    6. 反查所有相關 Concept 的 MENTIONS chunks (evidence)
    7. 排序 + 去重 + 回傳:
       {
         concepts: [...],    # 所有相關 KP
         chunks: [...],      # 所有相關 chunk (含原始命中 + 展開的)
         paths: [...],       # 概念之間的路徑 (Cypher path)
         prerequisites: [...]  # 學生若不會 X 要先補的 KP 清單
       }

用法 (CLI):
    python graphrag_query_kp_first.py --query "How do system calls work?"
    python graphrag_query_kp_first.py --query "..." --chapter ch02 --top-k 5 --top-m 8 --hops 2
    python graphrag_query_kp_first.py --query "..." --format json

用法 (Python API):
    from graphrag_query_kp_first import GraphRAGRetriever
    r = GraphRAGRetriever()
    result = r.retrieve("How do system calls work?", chapter_id="ch02", top_k=5)
    print(result.answer_context())
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from neo4j import GraphDatabase

from config import settings
from graphrag.embedder import GeminiEmbedder


# ============================================================
# Data classes
# ============================================================
@dataclass
class ChunkHit:
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    score: float  # semantic similarity
    source: str  # "semantic" | "expanded_from_concept"
    mentioned_kps: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ConceptHit:
    name: str  # PK (跨章合併)
    kp_ids: List[str]  # 可能對應多章的 KP id
    definition: str
    section: str
    score: float
    source: str  # "semantic" | "chunk_mention" | "graph_expand"


@dataclass
class PrereqChain:
    target: str  # target KP name
    prereqs: List[Dict[str, Any]]  # [{name, confidence, reason}, ...]


@dataclass
class RetrievalResult:
    query: str
    concepts: List[ConceptHit] = field(default_factory=list)
    chunks: List[ChunkHit] = field(default_factory=list)
    prerequisites: List[PrereqChain] = field(default_factory=list)
    paths: List[Dict[str, Any]] = field(default_factory=list)

    def answer_context(self, max_chunks: int = 8) -> str:
        """組給 LLM 用的 context (含引用)"""
        lines = ["# Related Concepts"]
        for c in self.concepts[:10]:
            lines.append(f"- **{c.name}** (sec {c.section}, score {c.score:.2f})")
            if c.definition:
                lines.append(f"  {textwrap.shorten(c.definition, 200)}")
        lines.append("\n# Related Chunks (with citation)")
        for k in self.chunks[:max_chunks]:
            kps = ", ".join(f"{m['name']}" for m in k.mentioned_kps[:5])
            lines.append(
                f"\n### [chunk {k.chunk_id}, p.{k.page_start}-{k.page_end}] "
                f"score={k.score:.2f}"
            )
            if kps:
                lines.append(f"*Mentions: {kps}*")
            lines.append(textwrap.shorten(k.text, 800))
        if self.prerequisites:
            lines.append("\n# Prerequisite chains")
            for chain in self.prerequisites[:5]:
                names = " ← ".join(p["name"] for p in chain.prereqs[:5])
                lines.append(f"- To learn **{chain.target}**: {names}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "concepts": [vars(c) for c in self.concepts],
            "chunks": [vars(k) for k in self.chunks],
            "prerequisites": [vars(p) for p in self.prerequisites],
            "paths": self.paths,
        }


# ============================================================
# Retriever
# ============================================================
class GraphRAGRetriever:
    """混合檢索: 向量搜尋 + Graph walk"""

    # 一次展開的關係類型
    EXPAND_RELS = [
        "HAS_SUBTYPE", "PART_OF", "USED_FOR", "IMPLEMENTS",
        "REQUIRES", "CAUSES", "CONTRASTED_WITH", "EXAMPLE_OF",
        "PREREQUISITE_OF",
    ]

    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        self.embedder = GeminiEmbedder()
        # 動態解析 vector index 名字 (舊 DB 可能用不同名字)
        self._concept_idx: Optional[str] = None
        self._chunk_idx: Optional[str] = None
        self._resolve_vector_indexes()

    def _resolve_vector_indexes(self):
        """SHOW VECTOR INDEXES 找 Concept/Chunk 的 embedding index 名字"""
        with self.driver.session(database=settings.neo4j_database) as sess:
            rows = sess.run("""
                SHOW VECTOR INDEXES
                YIELD name, labelsOrTypes, properties, state
                WHERE state = 'ONLINE' OR state = 'POPULATING'
                RETURN name, labelsOrTypes, properties
            """).data()
        for r in rows:
            labels = r.get("labelsOrTypes") or []
            props = r.get("properties") or []
            if "Concept" in labels and "embedding" in props and not self._concept_idx:
                self._concept_idx = r["name"]
            if "Chunk" in labels and "embedding" in props and not self._chunk_idx:
                self._chunk_idx = r["name"]
        print(f"[retriever] vector indexes: Concept='{self._concept_idx}', Chunk='{self._chunk_idx}'")
        if not self._concept_idx or not self._chunk_idx:
            print("[retriever] ⚠️ 缺 vector index. 執行以下 Cypher 建立:")
            if not self._concept_idx:
                print("  CREATE VECTOR INDEX concept_embedding_index FOR (c:Concept) ON c.embedding "
                      "OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}")
            if not self._chunk_idx:
                print("  CREATE VECTOR INDEX chunk_embedding_index FOR (c:Chunk) ON c.embedding "
                      "OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}")

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ============================================================
    def retrieve(
        self,
        query: str,
        chapter_id: Optional[str] = None,
        top_k: int = 5,          # top chunks (semantic)
        top_m: int = 6,          # top concepts (semantic)
        hops: int = 2,           # graph walk 深度
        expand_chunks: bool = True,  # 展開 concept 的 mention chunks
        include_prereq: bool = True,
    ) -> RetrievalResult:
        """執行完整 retrieval"""
        result = RetrievalResult(query=query)

        # 1. Query embedding
        q_vec = self.embedder.embed_query(query)

        with self.driver.session(database=settings.neo4j_database) as sess:
            chapter_id = self._resolve_chapter_id(sess, chapter_id)
            # 2. Vector search on Chunk
            chunk_hits = self._vector_search_chunk(sess, q_vec, top_k, chapter_id)
            # 3. Vector search on Concept
            concept_hits = self._vector_search_concept(sess, q_vec, top_m, chapter_id)

            # 4. Chunk → mentioned concepts
            for ch in chunk_hits:
                ch.mentioned_kps = self._chunk_mentions(sess, ch.chunk_id)

            # 5. Merge concept sources (key = name)
            concept_map: Dict[str, ConceptHit] = {}
            for c in concept_hits:
                concept_map[c.name] = c
            for ch in chunk_hits:
                for m in ch.mentioned_kps:
                    name = m["name"]
                    if name not in concept_map:
                        cinfo = self._get_concept(sess, name)
                        if cinfo:
                            concept_map[name] = ConceptHit(
                                name=name,
                                kp_ids=cinfo.get("kp_ids") or [],
                                definition=cinfo.get("definition", ""),
                                section=cinfo.get("section", ""),
                                score=ch.score * 0.7,
                                source="chunk_mention",
                            )

            # 6. Graph walk: 展開相關 concept
            seed_names = list(concept_map.keys())
            expanded = self._expand_concepts(sess, seed_names, hops)
            for name, info in expanded.items():
                if name not in concept_map:
                    concept_map[name] = ConceptHit(
                        name=name,
                        kp_ids=info.get("kp_ids") or [],
                        definition=info.get("definition", ""),
                        section=info.get("section", ""),
                        score=info.get("score", 0.3),
                        source="graph_expand",
                    )

            # 7. 展開所有相關 chunk (透過 MENTIONS)
            chunk_map: Dict[str, ChunkHit] = {ch.chunk_id: ch for ch in chunk_hits}
            if expand_chunks:
                extra = self._chunks_of_concepts(
                    sess, list(concept_map.keys()),
                    exclude=set(chunk_map.keys()),
                )
                for ch in extra:
                    chunk_map[ch.chunk_id] = ch

            # 8. Prerequisite chains
            prereqs: List[PrereqChain] = []
            if include_prereq:
                top_concepts = sorted(concept_map.values(), key=lambda x: -x.score)[:5]
                for c in top_concepts:
                    chain = self._prereq_chain(sess, c.name)
                    if chain:
                        prereqs.append(PrereqChain(target=c.name, prereqs=chain))

            # 9. Concept 間 path
            paths = self._concept_paths(sess, [c.name for c in concept_hits[:3]])

        # 排序輸出
        result.concepts = sorted(concept_map.values(), key=lambda x: -x.score)
        result.chunks = sorted(chunk_map.values(), key=lambda x: -x.score)
        result.prerequisites = prereqs
        result.paths = paths
        return result

    def _resolve_chapter_id(self, sess, requested: Optional[str]) -> Optional[str]:
        """Resolve CLI-friendly IDs such as ``ch10`` to their stored ID."""
        if not requested:
            return None
        rows = sess.run(
            """
            MATCH (ch:Chapter)
            WHERE ch.chapterId = $requested
               OR ch.chapterId ENDS WITH ('_' + $requested)
            RETURN ch.chapterId AS chapter_id
            ORDER BY CASE WHEN ch.chapterId = $requested THEN 0 ELSE 1 END
            """,
            requested=requested,
        ).data()
        chapter_ids = list(dict.fromkeys(
            str(row.get("chapter_id") or "") for row in rows if row.get("chapter_id")
        ))
        if not chapter_ids:
            raise ValueError(f"Chapter not found in Neo4j: {requested}")
        if len(chapter_ids) > 1:
            raise ValueError(
                f"Ambiguous chapter ID {requested!r}; use one of: {', '.join(chapter_ids)}"
            )
        return chapter_ids[0]

    # ============================================================
    # Vector searches
    # ============================================================
    def _vector_search_chunk(
        self, sess, q_vec: List[float], top_k: int, chapter_id: Optional[str]
    ) -> List[ChunkHit]:
        if not self._chunk_idx:
            return []
        cypher = f"""
        CALL db.index.vector.queryNodes('{self._chunk_idx}', $k, $vec)
        YIELD node AS k, score
        WITH k, score
        """ + ("""
        MATCH (ch:Chapter {chapterId: $chapter_id})-[:HAS_CHUNK]->(k)
        """ if chapter_id else "") + """
        RETURN k.chunkId AS chunk_id, k.text AS text,
               k.pageStart AS page_start, k.pageEnd AS page_end, score
        ORDER BY score DESC
        LIMIT $k
        """
        params = {"vec": q_vec, "k": top_k}
        if chapter_id:
            params["chapter_id"] = chapter_id
        rows = sess.run(cypher, **params).data()
        return [ChunkHit(
            chunk_id=r["chunk_id"], text=r["text"] or "",
            page_start=r.get("page_start") or 0,
            page_end=r.get("page_end") or 0,
            score=float(r["score"]), source="semantic",
        ) for r in rows]

    def _vector_search_concept(
        self, sess, q_vec: List[float], top_m: int, chapter_id: Optional[str]
    ) -> List[ConceptHit]:
        if not self._concept_idx:
            return []
        if chapter_id:
            cypher = f"""
            CALL db.index.vector.queryNodes('{self._concept_idx}', $m, $vec)
            YIELD node AS c, score
            MATCH (c)-[ic:IN_CHAPTER]->(:Chapter {{chapterId: $chapter_id}})
            OPTIONAL MATCH (c)-[:MENTIONED_IN]->(k:Chunk)
            WHERE k.chapter = $chapter_id
            RETURN coalesce(c.canonicalName, c.name) AS name,
                   coalesce(ic.localKpIds, []) AS kp_ids,
                   c.definition AS definition, min(k.section) AS section, score
            ORDER BY score DESC
            LIMIT $m
            """
        else:
            cypher = f"""
            CALL db.index.vector.queryNodes('{self._concept_idx}', $m, $vec)
            YIELD node AS c, score
            OPTIONAL MATCH (c)-[:MENTIONED_IN]->(k:Chunk)
            RETURN coalesce(c.canonicalName, c.name) AS name,
                   [] AS kp_ids, c.definition AS definition,
                   min(k.section) AS section, score
            ORDER BY score DESC
            LIMIT $m
            """
        params = {"vec": q_vec, "m": top_m}
        if chapter_id:
            params["chapter_id"] = chapter_id
        rows = sess.run(cypher, **params).data()
        return [ConceptHit(
            name=r["name"], kp_ids=r.get("kp_ids") or [],
            definition=r.get("definition") or "",
            section=r.get("section") or "",
            score=float(r["score"]), source="semantic",
        ) for r in rows]

    # ============================================================
    # Graph walks
    # ============================================================
    def _chunk_mentions(self, sess, chunk_id: str) -> List[Dict[str, str]]:
        rows = sess.run("""
            MATCH (k:Chunk {chunkId: $cid})-[m:MENTIONS]->(c:Concept)
            RETURN coalesce(c.canonicalName, c.name) AS name,
                   coalesce(m.evidenceQuotes, []) AS quotes
        """, cid=chunk_id).data()
        return [{
            "name": r["name"],
            "quote": ((r.get("quotes") or [""])[0] or ""),
        } for r in rows]

    def _get_concept(self, sess, name: str) -> Optional[Dict[str, Any]]:
        rec = sess.run("""
            MATCH (c:Concept)
            WHERE coalesce(c.canonicalName, c.name) = $name
            OPTIONAL MATCH (c)-[ic:IN_CHAPTER]->(:Chapter)
            OPTIONAL MATCH (c)-[:MENTIONED_IN]->(k:Chunk)
            RETURN coalesce(c.canonicalName, c.name) AS name,
                   collect(DISTINCT ic.localKpIds) AS kp_id_groups,
                   c.definition AS definition, min(k.section) AS section
        """, name=name).single()
        if not rec:
            return None
        result = dict(rec)
        result["kp_ids"] = [
            value for group in (result.pop("kp_id_groups", []) or [])
            for value in (group or [])
        ]
        return result

    def _expand_concepts(
        self, sess, seed_names: List[str], hops: int
    ) -> Dict[str, Dict[str, Any]]:
        """從 seed concept 走 EXPAND_RELS 展開 hops 層"""
        if not seed_names:
            return {}
        rel_pattern = "|".join(self.EXPAND_RELS)
        cypher = f"""
        MATCH (seed:Concept) WHERE coalesce(seed.canonicalName, seed.name) IN $seeds
        MATCH path = (seed)-[:{rel_pattern}*1..{hops}]-(related:Concept)
        WHERE NOT coalesce(related.canonicalName, related.name) IN $seeds
        WITH related, min(length(path)) AS dist
        OPTIONAL MATCH (related)-[:MENTIONED_IN]->(k:Chunk)
        RETURN coalesce(related.canonicalName, related.name) AS name,
               [] AS kp_ids, related.definition AS definition,
               min(k.section) AS section, dist
        ORDER BY dist
        LIMIT 40
        """
        rows = sess.run(cypher, seeds=seed_names).data()
        result = {}
        for r in rows:
            score = 0.5 / (r["dist"] or 1)
            result[r["name"]] = {
                "kp_ids": r.get("kp_ids") or [],
                "definition": r.get("definition") or "",
                "section": r.get("section") or "",
                "score": score,
            }
        return result

    def _chunks_of_concepts(
        self, sess, concept_names: List[str], exclude: Set[str],
    ) -> List[ChunkHit]:
        if not concept_names:
            return []
        cypher = """
        MATCH (k:Chunk)-[:MENTIONS]->(c:Concept)
        WHERE coalesce(c.canonicalName, c.name) IN $names
          AND NOT k.chunkId IN $exclude
        WITH k, count(DISTINCT c) AS n_mentions
        RETURN k.chunkId AS chunk_id, k.text AS text,
               k.pageStart AS page_start, k.pageEnd AS page_end, n_mentions
        ORDER BY n_mentions DESC
        LIMIT 12
        """
        rows = sess.run(cypher, names=concept_names, exclude=list(exclude)).data()
        return [ChunkHit(
            chunk_id=r["chunk_id"], text=r["text"] or "",
            page_start=r.get("page_start") or 0,
            page_end=r.get("page_end") or 0,
            score=0.3 + (r["n_mentions"] * 0.05),
            source="expanded_from_concept",
        ) for r in rows]

    def _prereq_chain(self, sess, name: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        cypher = f"""
        MATCH path = (prereq:Concept)-[:PREREQUISITE_OF*1..{max_depth}]->(target:Concept)
        WHERE coalesce(target.canonicalName, target.name) = $name
        WITH prereq, length(path) AS depth, [r IN relationships(path) | r.confidence] AS confs
        RETURN coalesce(prereq.canonicalName, prereq.name) AS name, depth,
               reduce(s = 1.0, c IN confs | s * c) AS chain_conf
        ORDER BY depth, chain_conf DESC
        LIMIT 10
        """
        rows = sess.run(cypher, name=name).data()
        return [{"name": r["name"], "depth": r["depth"], "confidence": r["chain_conf"]}
                for r in rows]

    def _concept_paths(self, sess, seed_names: List[str], max_hops: int = 3) -> List[Dict[str, Any]]:
        if len(seed_names) < 2:
            return []
        rel_pattern = "|".join(self.EXPAND_RELS)
        cypher = f"""
        MATCH (a:Concept) WHERE coalesce(a.canonicalName, a.name) IN $names
        MATCH (b:Concept)
        WHERE coalesce(b.canonicalName, b.name) IN $names
          AND coalesce(a.canonicalName, a.name) < coalesce(b.canonicalName, b.name)
        MATCH path = shortestPath((a)-[:{rel_pattern}*..{max_hops}]-(b))
        RETURN coalesce(a.canonicalName, a.name) AS from_name,
               coalesce(b.canonicalName, b.name) AS to_name,
               [n IN nodes(path) | coalesce(n.canonicalName, n.name)] AS node_path,
               [r IN relationships(path) | type(r)] AS rel_path
        LIMIT 10
        """
        rows = sess.run(cypher, names=seed_names).data()
        return [{
            "from": r["from_name"], "to": r["to_name"],
            "nodes": r["node_path"], "rels": r["rel_path"],
        } for r in rows]


# ============================================================
# CLI
# ============================================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--query", "-q", required=True)
    p.add_argument("--chapter", default=None, help="限定章節 e.g. ch02")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--top-m", type=int, default=6)
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--no-expand", action="store_true", help="不展開 concept mention chunks")
    p.add_argument("--no-prereq", action="store_true", help="不查 prereq chain")
    p.add_argument("--format", choices=["context", "json"], default="context")
    args = p.parse_args()

    with GraphRAGRetriever() as r:
        result = r.retrieve(
            args.query,
            chapter_id=args.chapter,
            top_k=args.top_k,
            top_m=args.top_m,
            hops=args.hops,
            expand_chunks=not args.no_expand,
            include_prereq=not args.no_prereq,
        )

    if args.format == "json":
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.answer_context())
        print("\n" + "=" * 60)
        print(f"Concepts: {len(result.concepts)}, Chunks: {len(result.chunks)}, "
              f"Prereq chains: {len(result.prerequisites)}, Paths: {len(result.paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
