"""Gemini Embedding 包裝（雙後端：Vertex AI 或 AI Studio）。

支援：
    - gemini-embedding-001（輸出 3072 維，可降到 768/1536）
    - text-embedding-005 / text-embedding-004（768 維）

後端自動判斷（與 LLM 邏輯一致）：
    - GEMINI_BACKEND=vertex 或 有 VERTEX_PROJECT → 走 Vertex AI（用 gcloud ADC）
    - 否則走 AI Studio（用 GOOGLE_API_KEY）

這樣使用者只用 gcloud ADC 就能算 embedding，不用另外申請 AI Studio key。
"""
from __future__ import annotations

import math
import os
from typing import List, Optional

from tqdm import tqdm

from config import settings, GEMINI_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


def _backend() -> str:
    """回傳 'vertex' 或 'aistudio'。"""
    return (GEMINI_CONFIG.get("backend") or "aistudio").lower()


class GeminiEmbedder:
    """Gemini Embedding 統一包裝；自動選 Vertex AI 或 AI Studio。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        output_dim: Optional[int] = None,
        backend: Optional[str] = None,
    ):
        self.backend = (backend or _backend()).lower()
        self.model = model or settings.gemini_embedding_model
        # 為了與既有 vector index (768) 相容，預設請降到 768
        self.output_dim = output_dim or settings.vector_dimensions

        if self.backend == "vertex":
            self._init_vertex()
        else:
            self._init_aistudio(api_key)

    # ============================================================
    # Vertex AI 後端（用 gcloud ADC，不需 API key）
    # ============================================================
    def _init_vertex(self):
        project = GEMINI_CONFIG.get("vertex_project") or os.getenv("VERTEX_PROJECT")
        location = GEMINI_CONFIG.get("vertex_location") or os.getenv("VERTEX_LOCATION", "us-central1")
        if not project:
            raise RuntimeError(
                "Vertex 後端但缺少 VERTEX_PROJECT。請在 .env 設 VERTEX_PROJECT=xxx"
            )
        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel
        except ImportError as e:
            raise RuntimeError(
                "缺少 vertexai 套件。請執行: pip install google-cloud-aiplatform"
            ) from e

        vertexai.init(project=project, location=location)
        # Vertex AI 的模型名稱不需 "models/" prefix
        model_name = self.model.replace("models/", "")
        # 【STRICT】研究檢索不可靜默切換 embedding model。
        # Concept.embedding 與查詢向量若不是由同一模型建立，即使維度相同，
        # cosine / Neo4j vector score 也失去可比較性。因此模型不可用時直接失敗。
        try:
            self._vertex_model = TextEmbeddingModel.from_pretrained(model_name)
            self._model_name = model_name
            self.active_model_name = model_name
        except Exception as e:
            raise RuntimeError(
                f"Vertex AI 無法載入指定 embedding model '{model_name}'。"
                "為避免與既有 Concept.embedding 模型不一致，嚴格模式禁止自動 fallback。"
            ) from e

        logger.info(
            f"✅ GeminiEmbedder 使用 Vertex AI 後端 "
            f"(project={project}, location={location}, model={self._model_name})"
        )

    # ============================================================
    # AI Studio 後端（用 GOOGLE_API_KEY）
    # ============================================================
    def _init_aistudio(self, api_key: Optional[str]):
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise RuntimeError(
                "缺少 google-generativeai 套件。請執行: pip install google-generativeai"
            ) from e

        self.api_key = (
            api_key
            or settings.google_api_key
            or GEMINI_CONFIG.get("api_key")
        )
        if not self.api_key:
            raise RuntimeError(
                "AI Studio 後端但缺少 GOOGLE_API_KEY / GEMINI_API_KEYS。"
                "請在 .env 設定 key，或改用 Vertex 後端（設 VERTEX_PROJECT）"
            )
        genai.configure(api_key=self.api_key)
        self._genai = genai
        if not self.model.startswith("models/"):
            self._model_path = f"models/{self.model}"
        else:
            self._model_path = self.model
        self.active_model_name = self._model_path
        logger.info(
            f"✅ GeminiEmbedder 使用 AI Studio 後端 (model={self._model_path})"
        )

    # ============================================================
    # 對外 API（不管後端都長一樣）
    # ============================================================
    def embed_query(self, text: str) -> List[float]:
        return self._embed(text, task_type="RETRIEVAL_QUERY")

    def embed_document(self, text: str) -> List[float]:
        """Embed one document without creating a one-item progress bar."""
        return self._embed(text, task_type="RETRIEVAL_DOCUMENT")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in tqdm(texts, desc=f"計算向量 ({self.backend})"):
            out.append(self._embed(t, task_type="RETRIEVAL_DOCUMENT"))
        return out

    def semantic_scores(self, query: str, texts: List[str]) -> List[float]:
        """使用同一個 Gemini embedding model 計算 Query↔候選文本 cosine similarity。

        嚴格研究規則：
        - query 固定使用 RETRIEVAL_QUERY；
        - Seed Chunk、distance=1 prerequisite Concept、prerequisite Chunk
          都使用 RETRIEVAL_DOCUMENT；
        - 不使用其他 embedding model / lexical fallback；
        - 向量維度不一致時直接報錯，避免 zip() 靜默截斷造成錯誤分數。
        """
        query = str(query or "").strip()
        if not query:
            raise ValueError("semantic_scores: query must not be empty")
        if not texts:
            return []

        normalized_texts = [str(text or "") for text in texts]
        query_vector = self.embed_query(query)
        if not query_vector:
            raise RuntimeError("semantic_scores: query embedding is empty")

        # 保留輸入位置；空白候選不送 embedding API，分數固定 0。
        nonempty_indexes = [
            index for index, value in enumerate(normalized_texts) if value.strip()
        ]
        document_vectors_by_index = {}
        if nonempty_indexes:
            vectors = self.embed_documents([normalized_texts[index] for index in nonempty_indexes])
            if len(vectors) != len(nonempty_indexes):
                raise RuntimeError(
                    "semantic_scores: document embedding count mismatch "
                    f"({len(vectors)} != {len(nonempty_indexes)})"
                )
            document_vectors_by_index = dict(zip(nonempty_indexes, vectors))

        expected_dim = len(query_vector)
        if self.output_dim and expected_dim != int(self.output_dim):
            raise RuntimeError(
                "semantic_scores: query embedding dimension mismatch: "
                f"expected {self.output_dim}, got {expected_dim}"
            )

        def cosine(left: List[float], right: List[float]) -> float:
            if len(left) != len(right):
                raise RuntimeError(
                    "semantic_scores: embedding dimension mismatch: "
                    f"query={len(left)}, document={len(right)}"
                )
            if not left or not right:
                return 0.0
            numerator = sum(float(a) * float(b) for a, b in zip(left, right))
            left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
            right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
            if left_norm == 0.0 or right_norm == 0.0:
                return 0.0
            score = numerator / (left_norm * right_norm)
            return max(-1.0, min(1.0, float(score)))

        scores: List[float] = []
        for index in range(len(normalized_texts)):
            vector = document_vectors_by_index.get(index)
            scores.append(0.0 if vector is None else cosine(query_vector, vector))
        return scores

    # ============================================================
    # 內部：依後端 dispatch
    # ============================================================
    def _embed(self, text: str, task_type: str) -> List[float]:
        if self.backend == "vertex":
            return self._embed_vertex(text, task_type)
        return self._embed_aistudio(text, task_type)

    def _embed_vertex(self, text: str, task_type: str) -> List[float]:
        """Vertex AI 算 embedding。"""
        from vertexai.language_models import TextEmbeddingInput
        # Vertex AI 的 task_type 是大寫底線格式
        inp = TextEmbeddingInput(text=text, task_type=task_type)
        kwargs = {}
        # gemini-embedding-001 才支援 output_dimensionality
        if "gemini-embedding" in self._model_name:
            kwargs["output_dimensionality"] = self.output_dim
        embeddings = self._vertex_model.get_embeddings([inp], **kwargs)
        return list(embeddings[0].values)

    def _embed_aistudio(self, text: str, task_type: str) -> List[float]:
        """AI Studio 算 embedding（原邏輯保留）。"""
        # AI Studio 的 task_type 是小寫底線
        ts_lower = task_type.lower()
        kwargs = dict(
            model=self._model_path,
            content=text,
            task_type=ts_lower,
        )
        if "gemini-embedding" in self._model_path:
            kwargs["output_dimensionality"] = self.output_dim
        resp = self._genai.embed_content(**kwargs)
        return resp["embedding"]
