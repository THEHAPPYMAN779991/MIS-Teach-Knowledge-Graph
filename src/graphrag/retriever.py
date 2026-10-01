"""GraphRAG 混合檢索：向量搜尋 → 圖譜展開。

流程：
1. query → embedding
2. 用 vector index 從 Neo4j 找出 top-k 相關 Concept
3. 沿 PREREQUISITE_OF 關係展開上游 (先輩) 與下游 (後續)
4. 取出每個概念出現過的 chunk 文字 (前 N 筆)
5. 打包成 RetrievalResult 給 QA chain
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from neo4j import GraphDatabase, Driver

from config import settings
from graphrag.embedder import GeminiEmbedder
from utils.cypher_queries import (
    EXPAND_CONCEPT_CONTEXT,
    GET_DESCENDANTS,
    GET_PREREQUISITES,
    VECTOR_SEARCH,
)
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    query: str
    seed_concepts: List[dict] = field(default_factory=list)  # 向量搜尋命中的概念
    expanded: List[dict] = field(default_factory=list)       # 含 chunks/prereq/leads_to
    prereq_chains: List[dict] = field(default_factory=list)  # 上游鏈
    descendant_chains: List[dict] = field(default_factory=list)  # 下游鏈


class GraphRetriever:
    def __init__(
        self,
        embedder: Optional[GeminiEmbedder] = None,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        self.embedder = embedder or GeminiEmbedder()
        self.uri = uri or settings.neo4j_uri
        self.username = username or settings.neo4j_username
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database
        self._driver: Optional[Driver] = None
        self.vector_index_name = settings.vector_index_name

    def __enter__(self):
        self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        self.vector_index_name = self._resolve_concept_vector_index()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._driver:
            self._driver.close()

    def _run(self, cypher: str, **params):
        assert self._driver is not None, "請使用 with GraphRetriever() as r"
        with self._driver.session(database=self.database) as session:
            return session.run(cypher, **params).data()

    def _resolve_concept_vector_index(self) -> str:
        """Resolve the online Concept.embedding index across old/new schemas.

        The legacy importer created ``concept_embeddings`` while the KP-first
        importer creates ``concept_embedding_index``. Retrieval follows the
        database schema instead of failing because the configured name is
        stale.
        """
        assert self._driver is not None
        with self._driver.session(database=self.database) as session:
            rows = session.run(
                """
                SHOW VECTOR INDEXES
                YIELD name, state, labelsOrTypes, properties
                WHERE state = 'ONLINE'
                RETURN name, labelsOrTypes, properties
                """
            ).data()
        candidates = [
            str(row["name"])
            for row in rows
            if "Concept" in (row.get("labelsOrTypes") or [])
            and "embedding" in (row.get("properties") or [])
        ]
        configured = str(settings.vector_index_name or "").strip()
        if configured in candidates:
            return configured
        if candidates:
            resolved = candidates[0]
            logger.warning(
                "Configured Concept vector index '%s' was not found; using '%s'.",
                configured,
                resolved,
            )
            return resolved
        raise RuntimeError(
            "No ONLINE vector index exists for (:Concept).embedding. "
            f"Configured index: {configured!r}"
        )

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        prereq_depth: int = 1,
        descendant_depth: int = 1,
    ) -> RetrievalResult:
        # 1. 向量
        vec = self.embedder.embed_query(query)
        seeds = self._run(
            VECTOR_SEARCH,
            index_name=self.vector_index_name,
            top_k=top_k,
            embedding=vec,
        )
        seed_names = [s["name"] for s in seeds]
        if not seed_names:
            logger.warning("向量檢索沒有命中任何概念")
            return RetrievalResult(query=query, seed_concepts=[], expanded=[])

        # 2. 展開 context
        expanded = self._run(EXPAND_CONCEPT_CONTEXT, names=seed_names)

        # 3. 各別取上游與下游（研究流程預設只展開 distance=1）
        prereq_chains: List[dict] = []
        descendant_chains: List[dict] = []
        for n in seed_names:
            ups = self._run(
                GET_PREREQUISITES.replace("$depth", str(prereq_depth)),
                name=n,
            )
            downs = self._run(
                GET_DESCENDANTS.replace("$depth", str(descendant_depth)),
                name=n,
            )
            prereq_chains.append({"concept": n, "ancestors": ups})
            descendant_chains.append({"concept": n, "descendants": downs})

        return RetrievalResult(
            query=query,
            seed_concepts=seeds,
            expanded=expanded,
            prereq_chains=prereq_chains,
            descendant_chains=descendant_chains,
        )
