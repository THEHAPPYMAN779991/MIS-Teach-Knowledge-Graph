"""把 GraphRAG 暴露成 REST API，方便其他語言模型/應用呼叫。

執行：
    pip install fastapi uvicorn
    python api_server.py

然後用任何 HTTP client 呼叫：
    POST http://localhost:8001/query
    {"question": "什麼是時間複雜度？", "top_k": 5}

或直接給 OpenAI Function Calling / LangChain Tool 用。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from graphrag.qa_chain import GraphRAGChain
from config import settings
from kg_builder.neo4j_writer import Neo4jWriter
from utils.cypher_queries import (
    GET_DESCENDANTS, GET_PARENT_CONCEPTS,
    GET_PREREQUISITES, GET_SUBTYPES, VECTOR_SEARCH,
)

app = FastAPI(title="計算機概論 GraphRAG API")

# 開放跨來源，方便 mis_teach_frontend (Angular dev server 4200) 直接打
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_chain: Optional[GraphRAGChain] = None

# book_id → {pdf_path, title, page_start_in_book, page_count}
_BOOK_INDEX: Dict[str, dict] = {}

# 專案根目錄（api_server.py 所在）
_PROJECT_ROOT = Path(__file__).resolve().parent


def get_chain() -> GraphRAGChain:
    global _chain
    if _chain is None:
        _chain = GraphRAGChain()
        _chain.__enter__()
    return _chain


# ============================================================
# Book index：建立 book_id → PDF 檔案路徑的對應
# ============================================================
def _build_book_index() -> Dict[str, dict]:
    """掃 data/chapters/<subject>/manifest.json，建出 book_id → metadata 索引。

    對應規則（兩種來源都會嘗試，後者覆寫前者，以 user 設定的 mapping 優先）：

    1) 自動：以每個 chapter folder 內 manifest.json 的 suggested_book_id 當 book_id
       （split_pdf_by_chapter.py 預設產生的格式：<split_book_id>_ch<NN>）

    2) 設定檔：data/book_prefix_mapping.json（使用者建立，用來覆寫）
        {
          "os_ch": "os",
          "csi_ch": "計算機概論總攬"
        }
       key = extract_only.py 用的 --book-prefix；value = data/chapters/ 下的資料夾名。
       會以 prefix + 兩位數編號 (e.g. csi_ch08) 對應到該資料夾裡 008_*.pdf。
    """
    global _BOOK_INDEX
    _BOOK_INDEX = {}

    chapters_root = _PROJECT_ROOT / "data" / "chapters"
    if not chapters_root.exists():
        return _BOOK_INDEX

    # ---- (1) 用 manifest.suggested_book_id 自動收 ----
    for folder in chapters_root.iterdir():
        if not folder.is_dir():
            continue
        manifest_path = folder / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for entry in entries:
            sbid = entry.get("suggested_book_id")
            filename = entry.get("filename")
            if not sbid or not filename:
                continue
            _BOOK_INDEX[sbid] = {
                "pdf_path": str((folder / filename).resolve()),
                "title": entry.get("title", sbid),
                "folder": folder.name,
                "page_start_in_book": entry.get("page_start", 1),
                "page_count": entry.get("page_count", 0),
            }

    # ---- (2) 用 book_prefix_mapping 覆寫 / 補強 ----
    mapping_path = _PROJECT_ROOT / "data" / "book_prefix_mapping.json"
    if mapping_path.exists():
        try:
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        except Exception:
            mapping = {}
        for prefix, folder_name in mapping.items():
            # 跳過以底線開頭的 key（例如 "_comment"）和非字串 value
            if not isinstance(prefix, str) or prefix.startswith("_"):
                continue
            if not isinstance(folder_name, str):
                continue
            folder = chapters_root / folder_name
            manifest_path = folder / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                entries = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for entry in entries:
                filename = entry.get("filename", "")
                m = re.match(r"^(\d+)_", filename)
                if not m:
                    continue
                num = int(m.group(1))
                bid = f"{prefix}{num:02d}"
                _BOOK_INDEX[bid] = {
                    "pdf_path": str((folder / filename).resolve()),
                    "title": entry.get("title", bid),
                    "folder": folder.name,
                    "page_start_in_book": entry.get("page_start", 1),
                    "page_count": entry.get("page_count", 0),
                }

    return _BOOK_INDEX


@app.on_event("startup")
def _startup():
    _build_book_index()
    print(f"[api_server] Book index 建立完成，共 {len(_BOOK_INDEX)} 本")


# ============================================================
# Schemas
# ============================================================
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    question: str
    answer: str
    seed_concepts: List[str]
    # ✨ 每個 seed 展開後拿到的補充資料
    # 每筆包含: {name, definition, sample_chunks, subtypes, parents,
    #           prerequisites, leads_to, book_ids, ...}
    # 沒有這格的話 MIS 側 expanded_count 會永遠是 0
    expanded: list = []
    prereq_chains: list
    descendant_chains: list


class RetrievalRequest(BaseModel):
    question: str
    top_k: int = 5
    prereq_depth: int = 1
    descendant_depth: int = 1


class SeedRetrievalRequest(BaseModel):
    """Vector-only Concept retrieval for experiments that expand the graph separately."""
    question: str
    top_k: int = 10


class SemanticScoreRequest(BaseModel):
    """Score candidate texts against one complete original question."""
    query: str
    texts: List[str]


# ============================================================
# Routes
# ============================================================
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """GraphRAG 問答：回傳 LLM 答案 + 引用的圖譜結構。"""
    return get_chain().ask(req.question, top_k=req.top_k)


@app.get("/health")
def health():
    """Report whether Neo4j and the active Concept vector index are ready."""
    try:
        chain = get_chain()
        return {
            "status": "ok",
            "neo4j_database": settings.neo4j_database,
            "concept_vector_index": getattr(
                chain.retriever,
                "vector_index_name",
                settings.vector_index_name,
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/retrieve")
def retrieve(req: RetrievalRequest):
    """Embedding vector search plus graph expansion, without an answer-model call."""
    top_k = max(1, min(req.top_k, 20))
    prereq_depth = max(1, min(req.prereq_depth, 4))
    descendant_depth = max(1, min(req.descendant_depth, 4))
    chain = get_chain()
    try:
        result = chain.retriever.retrieve(
            req.question,
            top_k=top_k,
            prereq_depth=prereq_depth,
            descendant_depth=descendant_depth,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "stage": "graphrag_retrieval",
                "error_type": type(exc).__name__,
                "message": str(exc)[:1000],
            },
        ) from exc
    return {
        "question": result.query,
        "seed_concepts": result.seed_concepts,
        "expanded": result.expanded,
        "prereq_chains": result.prereq_chains,
        "descendant_chains": result.descendant_chains,
        "retrieval_trace": {
            "embedding": {
                "model": getattr(chain.retriever.embedder, "active_model_name", getattr(chain.retriever.embedder, "model", "")),
                "vector_length": getattr(
                    chain.retriever.embedder,
                    "output_dim",
                    settings.vector_dimensions,
                ),
            },
            "vector_search": {
                "index_name": getattr(
                    chain.retriever,
                    "vector_index_name",
                    settings.vector_index_name,
                ),
                "top_k": top_k,
                "hit_count": len(result.seed_concepts),
            },
            "graph_expansion": {
                "prereq_depth": prereq_depth,
                "descendant_depth": descendant_depth,
                "expanded_count": len(result.expanded),
                "prereq_chain_count": len(result.prereq_chains),
                "descendant_chain_count": len(result.descendant_chains),
            },
        },
    }


@app.post("/retrieve-seeds")
def retrieve_seeds(req: SeedRetrievalRequest):
    """Return the same vector-ranked Concept seeds without graph expansion.

    This endpoint uses the retriever's existing embedding model, Neo4j vector
    index, and ``VECTOR_SEARCH`` Cypher.  It intentionally skips expanded
    Chunks, prerequisite chains, and descendant chains so a controlled audit
    can query those layers itself without paying for or mixing in hidden graph
    traversal.
    """
    question = str(req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")
    top_k = max(1, min(req.top_k, 20))
    chain = get_chain()
    vector = chain.retriever.embedder.embed_query(question)
    seeds = chain.retriever._run(
        VECTOR_SEARCH,
        index_name=getattr(
            chain.retriever,
            "vector_index_name",
            settings.vector_index_name,
        ),
        top_k=top_k,
        embedding=vector,
    )
    return {
        "question": question,
        "seed_concepts": seeds,
        "retrieval_trace": {
            "mode": "vector_seeds_only_no_graph_expansion",
            "embedding": {
                "model": getattr(chain.retriever.embedder, "active_model_name", getattr(chain.retriever.embedder, "model", "")),
                "vector_length": getattr(
                    chain.retriever.embedder,
                    "output_dim",
                    settings.vector_dimensions,
                ),
            },
            "vector_search": {
                "index_name": getattr(
                    chain.retriever,
                    "vector_index_name",
                    settings.vector_index_name,
                ),
                "top_k": top_k,
                "hit_count": len(seeds),
            },
            "graph_expansion": "disabled; caller performs explicit audited traversal",
        },
    }


@app.post("/semantic-scores")
def semantic_scores(req: SemanticScoreRequest):
    """Use the same GeminiEmbedder as Concept vector retrieval for second-stage ranking.

    No lexical fallback is used here. If embedding fails, return 503 so the MIS
    research-policy client can fail closed instead of silently changing methods.
    """
    query = str(req.query or "").strip()
    texts = [str(value or "") for value in (req.texts or [])]
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")
    if not texts:
        return {
            "scores": [],
            "backend": "ComputerScienceKG/GeminiEmbedder",
            "model": getattr(get_chain().retriever.embedder, "active_model_name", getattr(get_chain().retriever.embedder, "model", "")),
            "vector_length": getattr(get_chain().retriever.embedder, "output_dim", settings.vector_dimensions),
        }
    chain = get_chain()
    try:
        scores = chain.retriever.embedder.semantic_scores(query, texts)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "stage": "semantic_rerank",
                "error_type": type(exc).__name__,
                "message": str(exc)[:1000],
            },
        ) from exc
    return {
        "scores": scores,
        "backend": "ComputerScienceKG/GeminiEmbedder",
        "model": getattr(chain.retriever.embedder, "active_model_name", getattr(chain.retriever.embedder, "model", "")),
        "vector_length": getattr(chain.retriever.embedder, "output_dim", settings.vector_dimensions),
        "query_count": 1,
        "document_count": len(texts),
    }


@app.get("/concept/{name}")
def get_concept(name: str, depth: int = 3):
    """查詢某概念的先輩、後續、子類、父概念（不走 LLM）。"""
    with Neo4jWriter() as w:
        ups = w._run(GET_PREREQUISITES.replace("$depth", str(depth)), name=name)
        downs = w._run(GET_DESCENDANTS.replace("$depth", str(depth)), name=name)
        children = w._run(GET_SUBTYPES, name=name)
        parents = w._run(GET_PARENT_CONCEPTS, name=name)
    return {
        "name": name,
        "prerequisites": ups,
        "descendants": downs,
        "subtypes": children,
        "parents": parents,
    }


@app.get("/concept/{name}/locations")
def get_concept_locations(name: str, limit: int = 30):
    """回傳該概念在教材中出現的位置 (book_id, page, chapterTitle, snippet)。

    來源：Neo4j 中 (Concept)-[:MENTIONED_IN]->(Chunk) 的關聯。
    每個 location 都會附 hasPdf 欄位，告訴前端能否打開原書 PDF。
    """
    cypher = """
    MATCH (c:Concept {name: $name})-[:MENTIONED_IN]->(ck:Chunk)
    OPTIONAL MATCH (ch:Chapter {chapterId: ck.chapterId})
    WITH ck.bookId       AS bookId,
         ck.chapterId    AS chapterId,
         ck.page         AS page,
         ck.chunkSeqId   AS chunkSeqId,
         coalesce(ch.title, ck.chapterId) AS chapterTitle,
         substring(ck.text, 0, 200)       AS snippet
    RETURN bookId, chapterId, page, chunkSeqId, chapterTitle, snippet
    ORDER BY bookId, chunkSeqId
    LIMIT $limit
    """
    with Neo4jWriter() as w:
        rows = w._run(cypher, name=name, limit=limit)

    # 補上 PDF 是否可用 / 書名
    for r in rows:
        bid = r.get("bookId") or ""
        idx = _BOOK_INDEX.get(bid, {})
        pdf_path = idx.get("pdf_path")
        r["hasPdf"] = bool(pdf_path and Path(pdf_path).exists())
        r["bookTitle"] = idx.get("title", bid)

    # 同個 bookId 只保留前 5 個 chunk，避免一個概念塞 30 筆同章資訊
    grouped: Dict[str, list] = {}
    for r in rows:
        grouped.setdefault(r["bookId"], []).append(r)
    deduped = []
    for bid, items in grouped.items():
        deduped.extend(items[:5])

    return {"name": name, "locations": deduped, "total_books": len(grouped)}


@app.get("/pdf/{book_id}")
def serve_pdf(book_id: str):
    """直接回傳該 book_id 對應的 PDF（讓瀏覽器內建 viewer 開啟）。"""
    info = _BOOK_INDEX.get(book_id)
    if not info:
        raise HTTPException(404, detail=f"未找到 book_id={book_id} 的 PDF 對應")
    pdf_path = Path(info["pdf_path"])
    if not pdf_path.exists():
        raise HTTPException(
            404, detail=f"book_id={book_id} 對應檔不存在: {pdf_path}")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        # inline 讓瀏覽器內嵌顯示而非下載
        headers={
            "Content-Disposition":
                f'inline; filename="{pdf_path.name}"',
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.get("/books")
def list_books():
    """列出已索引到的所有 book_id 與 metadata（debug 用）。"""
    return {
        "count": len(_BOOK_INDEX),
        "books": _BOOK_INDEX,
    }


@app.post("/books/rebuild")
def rebuild_index():
    """手動重建 book 索引（資料夾改動後呼叫）。"""
    idx = _build_book_index()
    return {"count": len(idx)}


@app.get("/stats")
def stats():
    """整張圖譜的統計指標。"""
    with Neo4jWriter() as w:
        rows = w._run("""
        CALL { MATCH (c:Concept) RETURN count(c) AS concept_count }
        CALL { MATCH ()-[r:HAS_SUBTYPE]->() RETURN count(r) AS subtype_count }
        CALL { MATCH ()-[r:PREREQUISITE_OF]->() RETURN count(r) AS prereq_count }
        CALL { MATCH (b:Book) RETURN count(b) AS book_count }
        RETURN concept_count, subtype_count, prereq_count, book_count
        """)
    return rows[0] if rows else {}


@app.get("/")
def root():
    return {
        "name": "計算機概論 GraphRAG API",
        "endpoints": [
            "/query (POST)",
            "/retrieve (POST)",
            "/concept/{name} (GET)",
            "/concept/{name}/locations (GET)",
            "/pdf/{book_id} (GET, 回傳 PDF binary)",
            "/health (GET)",
        ],
    }


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
    )
