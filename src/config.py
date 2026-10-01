"""統一設定載入器。所有模組透過 settings 物件取得設定值。

Gemini API 使用風格已對齊 MIS 專案 (mis_teach_backend/src/rag_sys/config.py)：
- 環境變數同時支援 GEMINI_API_KEY 與 GOOGLE_API_KEY（GEMINI_API_KEY 優先）
- 提供集中式 GEMINI_CONFIG dict
- 預設模型升級為 gemini-2.5-flash
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# Public-package root.  The runtime modules live under ``src/`` while local
# configuration, ignored data, and generated outputs live one level above it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 載入 .env (位於專案根目錄)
# override=True：.env 永遠優先於 OS 環境變數，避免舊的系統 key 干擾
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _get_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


# 同時支援 GEMINI_API_KEY (MIS 風格) 與 GOOGLE_API_KEY (原本)，
# 加上 GEMINI_API_KEYS 複數版（逗號分隔，給自動輪替用）。
# 解析順序：
#   1) GEMINI_API_KEYS (複數，逗號分隔，最優先)
#   2) GEMINI_API_KEY  (MIS 命名)
#   3) GOOGLE_API_KEY  (原本命名)
#   4) AI_API_KEYS     (MIS 別名)
# 最終回傳 list[str]，順序就是嘗試順序。
def _resolve_gemini_keys() -> list[str]:
    plural = os.getenv("GEMINI_API_KEYS", "")
    if plural:
        keys = [k.strip() for k in plural.split(",") if k.strip()]
        if keys:
            return keys
    single = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("AI_API_KEYS")
        or ""
    )
    return [single] if single else []


# ============================================================
# 後端選擇：vertex (Google Cloud / 走 ADC) 或 aistudio (用 API key)
# 自動偵測規則：
#   - 若 GEMINI_BACKEND 顯式指定 → 用該值
#   - 若 VERTEX_PROJECT 有設 → 預設 vertex
#   - 都沒有 → aistudio
# ============================================================
def _resolve_backend() -> str:
    explicit = os.getenv("GEMINI_BACKEND", "").strip().lower()
    if explicit in ("vertex", "aistudio"):
        return explicit
    if os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT"):
        return "vertex"
    return "aistudio"


# ============================================================
# 集中式 Gemini 配置（對齊 MIS 風格 + 多 key 輪替 + 雙後端）
# 任何模組想要 raw config 都從這裡拿。
# ============================================================
_KEYS = _resolve_gemini_keys()
GEMINI_CONFIG = {
    # 後端選擇
    "backend": _resolve_backend(),

    # Vertex AI 設定（backend=vertex 時用）
    "vertex_project": (
        os.getenv("VERTEX_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or ""
    ),
    "vertex_location": os.getenv("VERTEX_LOCATION", "us-central1"),

    # AI Studio key 設定（backend=aistudio 時用）
    # 對齊 MIS 介面：api_key 仍然是單一字串（取第一個）
    "api_key": _KEYS[0] if _KEYS else "",
    # 新增：完整 key 清單，gemini_client 用來輪替
    "api_keys": _KEYS,

    # 共用
    "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 16384,
}

# 可用 AI 模型選項（為日後接 Ollama / OpenAI 等預留）
AVAILABLE_AI_MODELS = {
    "gemini": {
        "name": "Gemini (Google API)",
        "description": "Google Gemini 2.5 Flash",
        "type": "api",
        "config": GEMINI_CONFIG,
    }
}

DEFAULT_AI_MODEL = "gemini"


@dataclass
class Settings:
    # ----- Google Gemini (向後相容欄位 — 內部都從 GEMINI_CONFIG 派生) -----
    google_api_key: str = field(default_factory=lambda: GEMINI_CONFIG["api_key"])
    gemini_model: str = field(default_factory=lambda: GEMINI_CONFIG["model"])
    gemini_embedding_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
    )

    # ----- Neo4j -----
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", ""))
    neo4j_username: str = field(default_factory=lambda: os.getenv("NEO4J_USERNAME", ""))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    neo4j_database: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))

    # ----- Pipeline -----
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 1500))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 200))
    max_concepts_per_chunk: int = field(default_factory=lambda: _get_int("MAX_CONCEPTS_PER_CHUNK", 8))
    prerequisite_batch_size: int = field(
        default_factory=lambda: _get_int("PREREQUISITE_BATCH_SIZE", 15)
    )
    vector_index_name: str = field(
        default_factory=lambda: os.getenv("VECTOR_INDEX_NAME", "concept_embeddings")
    )
    vector_dimensions: int = field(default_factory=lambda: _get_int("VECTOR_DIMENSIONS", 768))

    # ----- 路徑 -----
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)

    def validate(self) -> None:
        """執行前先檢查必要欄位（依後端而異）。"""
        missing = []
        backend = GEMINI_CONFIG.get("backend", "aistudio")
        if backend == "vertex":
            if not GEMINI_CONFIG.get("vertex_project"):
                missing.append("VERTEX_PROJECT (或 GOOGLE_CLOUD_PROJECT)")
        else:
            if not self.google_api_key:
                missing.append("GEMINI_API_KEY (或 GOOGLE_API_KEY)")
        for key, value in (
            ("NEO4J_URI", self.neo4j_uri),
            ("NEO4J_USERNAME", self.neo4j_username),
            ("NEO4J_PASSWORD", self.neo4j_password),
        ):
            if not value:
                missing.append(key)
        if missing:
            raise RuntimeError(
                f"缺少必要的環境變數: {', '.join(missing)}。請檢查 .env 檔案。"
            )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
