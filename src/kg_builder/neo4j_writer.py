"""Neo4j writer."""
from __future__ import annotations

from typing import List, Optional

from neo4j import GraphDatabase, Driver
from tqdm import tqdm

from config import settings
from kg_builder.concept_extractor import (
    Concept, LocalPrerequisiteEdge, SubtypeEdge, get_concept_book_ids,
)
from kg_builder.prerequisite_analyzer import PrerequisiteEdge
from pdf_processor.pdf_loader import ParsedDocument
from pdf_processor.text_chunker import Chunk
from utils.cypher_queries import (
    CONSTRAINTS, CREATE_VECTOR_INDEX, DETECT_CYCLE_FOR_EDGE,
    LINK_CONCEPT_TO_CHUNK, LINK_HAS_SUBTYPE, LINK_PREREQUISITE,
    MERGE_ALIAS_INTO_MAIN, MERGE_BOOK, MERGE_CHAPTER, MERGE_CHUNK,
    MERGE_CONCEPT, MERGE_CONCEPT_NO_APOC,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jWriter:
    def __init__(self, uri=None, username=None, password=None,
                 database=None, use_apoc=True):
        self.uri = uri or settings.neo4j_uri
        self.username = username or settings.neo4j_username
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database
        self.use_apoc = use_apoc
        self._driver: Optional[Driver] = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def connect(self):
        self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        self._driver.verify_connectivity()
        logger.info(f"已連線 Neo4j: {self.uri}")

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    def _run(self, cypher, **params):
        assert self._driver is not None, "尚未呼叫 connect()"
        with self._driver.session(database=self.database) as session:
            return session.run(cypher, **params).data()

    def setup_schema(self):
        for c in CONSTRAINTS:
            try:
                self._run(c)
            except Exception as e:
                logger.warning(f"建立 constraint 失敗 (可能已存在): {e}")
        try:
            assert self._driver is not None
            with self._driver.session(database=self.database) as session:
                session.run(CREATE_VECTOR_INDEX,
                            index_name=settings.vector_index_name,
                            dimensions=settings.vector_dimensions)
            logger.info(f"已建立向量索引: {settings.vector_index_name}")
        except Exception as e:
            logger.warning(f"建立向量索引失敗: {e}")

    def write_book_and_chapters(self, doc: ParsedDocument):
        self._run(MERGE_BOOK, bookId=doc.book_id, title=doc.title)
        for ch in doc.chapters:
            self._run(MERGE_CHAPTER, bookId=doc.book_id,
                      chapterId=ch.chapter_id, title=ch.title,
                      order=ch.order, level=ch.level)
        logger.info(f"寫入 Book + {len(doc.chapters)} Chapter")

    def write_chunks(self, chunks: List[Chunk]):
        for ck in tqdm(chunks, desc="寫入 Chunk"):
            self._run(MERGE_CHUNK, chunkId=ck.chunk_id, text=ck.text,
                      chunkSeqId=ck.chunk_seq_id, chapterId=ck.chapter_id,
                      bookId=ck.book_id, page=ck.page_start)
        logger.info(f"寫入 {len(chunks)} Chunk")

    def write_concepts(self, concepts: List[Concept]):
        # 每次迭代都依當下的 use_apoc 重選 cypher，避免 fallback 後變數沒更新
        for c in tqdm(concepts, desc="寫入 Concept"):
            book_ids = get_concept_book_ids(c) or ([c.book_id] if c.book_id else [])
            params = dict(name=c.name,
                          definition=c.definition or "",
                          aliases=list(c.aliases or []),
                          category=c.category or "其他",
                          isFineGrained=bool(c.is_fine_grained),
                          bookIds=book_ids)
            cypher = MERGE_CONCEPT if self.use_apoc else MERGE_CONCEPT_NO_APOC
            try:
                self._run(cypher, **params)
            except Exception as e:
                if self.use_apoc and ("apoc" in str(e).lower() or "function" in str(e).lower()):
                    logger.warning(f"APOC 寫入失敗，從此改用 no-APOC: {e}")
                    self.use_apoc = False
                    self._run(MERGE_CONCEPT_NO_APOC, **params)
                else:
                    raise
        logger.info(f"寫入 {len(concepts)} Concept")

    def link_concepts_to_chunks(self, concepts: List[Concept]):
        for c in tqdm(concepts, desc="連結 Concept->Chunk"):
            if not c.chunk_id:
                continue
            self._run(LINK_CONCEPT_TO_CHUNK, name=c.name, chunkId=c.chunk_id)

    def write_subtype_edges(self, edges: List[SubtypeEdge]) -> int:
        written = 0
        for e in tqdm(edges, desc="寫入 HAS_SUBTYPE"):
            try:
                self._run(LINK_HAS_SUBTYPE, parent=e.parent, child=e.child,
                          source="extraction")
                written += 1
            except Exception as ex:
                logger.warning(f"HAS_SUBTYPE 失敗 {e.parent}->{e.child}: {ex}")
        logger.info(f"寫入 {written} 條 HAS_SUBTYPE")
        return written

    def write_local_prerequisites(self, edges: List[LocalPrerequisiteEdge]) -> int:
        promoted = [
            PrerequisiteEdge(prereq=e.prereq, target=e.target,
                             confidence=e.confidence, reason=e.reason,
                             source=e.source or "extraction")
            for e in edges
        ]
        return self.write_prerequisite_edges(promoted)

    def write_prerequisite_edges(self, edges: List[PrerequisiteEdge],
                                 batch_size: int = 500) -> int:
        """快速寫入 PREREQUISITE_OF 邊：
        - 在 Python (networkx) 做環路檢查，避免每條邊都跑昂貴的 Cypher 路徑查詢
        - 批次 UNWIND 寫入，速度比逐條快 100+ 倍
        """
        import networkx as nx

        if not edges:
            return 0

        # 1. 載入既有的 PREREQUISITE_OF 建立 DAG 基準
        logger.info("載入既有 PREREQUISITE_OF 建立 DAG 基準...")
        rows = self._run(
            "MATCH (a:Concept)-[:PREREQUISITE_OF]->(b:Concept) "
            "RETURN a.name AS src, b.name AS tgt"
        )
        g = nx.DiGraph()
        for r in rows:
            g.add_edge(r["src"], r["tgt"])
        logger.info(f"既有邊: {len(rows)}")

        # 2. 高 confidence 優先加邊
        edges_sorted = sorted(edges, key=lambda e: -float(e.confidence or 0))

        # 3. 環路檢查 (in-memory)
        kept = []
        dropped = 0
        for e in tqdm(edges_sorted, desc="DAG 環路檢查"):
            if e.prereq == e.target:
                dropped += 1
                continue
            if g.has_node(e.target) and g.has_node(e.prereq):
                if nx.has_path(g, e.target, e.prereq):
                    dropped += 1
                    continue
            g.add_edge(e.prereq, e.target)
            kept.append(e)
        logger.info(f"DAG 檢查: 保留 {len(kept)} / 環路丟棄 {dropped}")

        # 4. 批次 UNWIND 寫入
        unwind_cypher = """
        UNWIND $edges AS e
        MATCH (a:Concept {name: e.prereq})
        MATCH (b:Concept {name: e.target})
        MERGE (a)-[r:PREREQUISITE_OF]->(b)
          ON CREATE SET r.confidence = e.confidence,
                        r.reason     = e.reason,
                        r.source     = e.source,
                        r.createdAt  = datetime()
          ON MATCH SET
            r.confidence = CASE WHEN e.confidence > coalesce(r.confidence, 0)
                                THEN e.confidence ELSE r.confidence END,
            r.reason     = coalesce(r.reason, e.reason)
        RETURN count(*) AS n
        """
        written = 0
        for i in tqdm(range(0, len(kept), batch_size),
                      desc=f"批次寫入 PREREQUISITE_OF (每批 {batch_size})"):
            batch = kept[i:i + batch_size]
            payload = [
                {
                    "prereq": e.prereq,
                    "target": e.target,
                    "confidence": float(e.confidence or 0),
                    "reason": (e.reason or "")[:500],
                    "source": e.source or "extraction",
                }
                for e in batch
            ]
            r = self._run(unwind_cypher, edges=payload)
            if r:
                written += r[0]["n"]
        logger.info(f"實際寫入 {written}/{len(edges)} 條先輩邊")
        return written

    def store_embeddings(self, embeddings):
        cypher = (
            "MATCH (c:Concept {name: $name}) "
            "CALL db.create.setNodeVectorProperty(c, 'embedding', $vec) "
            "RETURN c.name AS n"
        )
        fallback = "MATCH (c:Concept {name: $name}) SET c.embedding = $vec"
        for name, vec in tqdm(embeddings.items(), desc="寫入向量"):
            try:
                self._run(cypher, name=name, vec=vec)
            except Exception:
                self._run(fallback, name=name, vec=vec)
        logger.info(f"寫入 {len(embeddings)} 個概念向量")

    def merge_concept_alias(self, alias_name: str, main_name: str):
        try:
            res = self._run(MERGE_ALIAS_INTO_MAIN,
                            alias_name=alias_name, main_name=main_name)
            if res:
                logger.info(f"已合併 [{alias_name}] -> [{main_name}]")
                return res[0].get("canonical")
        except Exception as e:
            logger.warning(f"合併失敗 {alias_name} -> {main_name}: {e}")
        return None

    def export_graph(self) -> dict:
        nodes = self._run(
            "MATCH (c:Concept) RETURN c.name AS name, "
            "c.definition AS definition, c.category AS category, "
            "c.isFineGrained AS isFineGrained, c.bookIds AS bookIds"
        )
        prereq_edges = self._run(
            "MATCH (a:Concept)-[r:PREREQUISITE_OF]->(b:Concept) "
            "RETURN a.name AS source, b.name AS target, "
            "r.confidence AS confidence, r.reason AS reason, "
            "'prerequisite' AS type"
        )
        subtype_edges = self._run(
            "MATCH (a:Concept)-[:HAS_SUBTYPE]->(b:Concept) "
            "RETURN a.name AS source, b.name AS target, "
            "1.0 AS confidence, 'subtype of' AS reason, "
            "'subtype' AS type"
        )
        return {"nodes": nodes, "edges": prereq_edges + subtype_edges}
