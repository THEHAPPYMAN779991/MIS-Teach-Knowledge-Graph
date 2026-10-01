"""Gemini API 統一入口 — 雙後端架構（含 Vertex timeout + retry）。

支援兩種後端：
1) `vertex`    — Google Cloud Vertex AI，用 gcloud ADC 認證（無需 API key）
2) `aistudio`  — Google AI Studio，用 API key（支援多 key 輪替）

選擇邏輯（config.GEMINI_CONFIG['backend']）：
- 顯式設 GEMINI_BACKEND=vertex / aistudio
- 沒設但有 VERTEX_PROJECT → vertex
- 都沒有 → aistudio

對外介面與 MIS call_gemini_api 對齊：
    from utils.gemini_client import init_gemini, call_gemini_api, generate_with_rotation

    model = init_gemini(generation_config={"temperature": 0.1,
                                           "response_mime_type": "application/json"})
    text = generate_with_rotation(model, prompt)

    # 或一次性
    text = call_gemini_api("Hello")
"""
from __future__ import annotations

import concurrent.futures
import os
import re
import time
from typing import List, Optional

from config import GEMINI_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 後端偵測 + 套件 lazy import + Vertex 超時/重試設定
# ============================================================
_BACKEND = GEMINI_CONFIG.get("backend", "aistudio")
_VERTEX_INITED = False

# Vertex AI 單次呼叫 timeout（秒）
# 預設 90 秒：足夠 Gemini 2.5 Flash 處理長 chunk，但遠低於 Windows TCP 預設 ~54 分
_VERTEX_TIMEOUT = int(os.getenv("VERTEX_TIMEOUT_SECONDS", "90"))

# Vertex 連線錯誤時的重試次數（gemini_client 內部，跟 concept_extractor 的 retry 獨立）
_VERTEX_MAX_RETRIES = int(os.getenv("VERTEX_MAX_RETRIES", "3"))

# 退避秒數：第 N 次失敗後等多久（指數退避）
_VERTEX_BACKOFF_BASE = float(os.getenv("VERTEX_BACKOFF_BASE", "3.0"))

# ===== 主動速率限制 (RPM Rate Limiter) =====
# 預設每分鐘最多 50 次（Vertex AI 預設配額為 60 RPM，留 10 buffer 給 retry）
# 想跑快一點 → 申請提升配額後，把 .env 的 VERTEX_MAX_RPM 調高（例如 200/300）
_VERTEX_MAX_RPM = int(os.getenv("VERTEX_MAX_RPM", "50"))

# rate limit 內部狀態：最近 60 秒內的呼叫時間戳
import collections
import threading as _threading
_VERTEX_INIT_LOCK = _threading.Lock()
_VERTEX_CALL_TIMES: collections.deque = collections.deque()
_VERTEX_RATE_LOCK = _threading.Lock()


def _vertex_rate_limit_wait():
    """在送出 Vertex AI 呼叫前主動等待，確保每分鐘呼叫數不超過 _VERTEX_MAX_RPM。

    用滑動視窗：若最近 60 秒內已有 max_rpm 次呼叫，就 sleep 到最早那筆超出視窗為止。
    這比 retry 後才 backoff 友善很多 — 配額連碰都不碰，自然不會 429。
    """
    if _VERTEX_MAX_RPM <= 0:
        return  # 0 / 負值 = 關閉限速
    with _VERTEX_RATE_LOCK:
        now = time.time()
        # 移除已超過 60 秒的舊紀錄
        while _VERTEX_CALL_TIMES and now - _VERTEX_CALL_TIMES[0] > 60.0:
            _VERTEX_CALL_TIMES.popleft()
        # 若視窗內已滿，等到最早那筆超出視窗為止
        if len(_VERTEX_CALL_TIMES) >= _VERTEX_MAX_RPM:
            wait = 60.0 - (now - _VERTEX_CALL_TIMES[0]) + 0.2  # 多 0.2 秒緩衝
            if wait > 0:
                logger.info(
                    f"⏸ Vertex AI 速率限制：60 秒內已 {len(_VERTEX_CALL_TIMES)} 次呼叫，"
                    f"暫停 {wait:.1f} 秒避免 429"
                )
                time.sleep(wait)
                now = time.time()
                while _VERTEX_CALL_TIMES and now - _VERTEX_CALL_TIMES[0] > 60.0:
                    _VERTEX_CALL_TIMES.popleft()
        _VERTEX_CALL_TIMES.append(now)


def _import_vertex():
    """延後 import，避免沒裝 google-cloud-aiplatform 的環境 crash。"""
    import vertexai
    from vertexai.generative_models import GenerativeModel as VxGenerativeModel
    return vertexai, VxGenerativeModel


def _import_aistudio():
    import google.generativeai as genai
    return genai


def _ensure_vertex_inited() -> None:
    global _VERTEX_INITED
    if _VERTEX_INITED:
        return
    with _VERTEX_INIT_LOCK:
        if _VERTEX_INITED:
            return
        project = GEMINI_CONFIG.get("vertex_project")
        location = GEMINI_CONFIG.get("vertex_location", "us-central1")
        if not project:
            raise RuntimeError(
                "Vertex 後端需要 VERTEX_PROJECT 或 GOOGLE_CLOUD_PROJECT。"
                "請在 .env 設定或執行 `gcloud config set project <id>`。"
            )
        vertexai, _ = _import_vertex()
        vertexai.init(project=project, location=location)
        logger.info(f"Vertex AI 初始化: project={project}, location={location}")
        _VERTEX_INITED = True


# ============================================================
# AI Studio 用：多 key 輪替池
# ============================================================
class _KeyPool:
    """AI Studio key 輪替池。Vertex 後端不會用到。"""

    def __init__(self, keys: List[str]):
        self.keys = [k for k in keys if k]
        self.idx = 0
        self._last_configured: Optional[str] = None
        self._lock = _threading.RLock()

    def has_keys(self) -> bool:
        return len(self.keys) > 0

    def current(self) -> Optional[str]:
        with self._lock:
            return self.keys[self.idx] if self.keys else None

    def advance(self) -> Optional[str]:
        with self._lock:
            if self.idx + 1 >= len(self.keys):
                return None
            previous = self.keys[self.idx]
            self.idx += 1
            nxt = self.keys[self.idx]
            logger.warning(
                f"切換 Gemini API key: {self._mask(previous)} → "
                f"{self._mask(nxt)} ({self.idx + 1}/{len(self.keys)})"
            )
            return nxt

    def ensure_configured(self) -> None:
        with self._lock:
            cur = self.keys[self.idx] if self.keys else None
            if cur is None:
                raise RuntimeError(
                    "沒有可用的 Gemini API key。請在 .env 設定 GEMINI_API_KEY "
                    "或 GEMINI_API_KEYS（多個逗號分隔），或改用 Vertex 後端。"
                )
            if cur != self._last_configured:
                genai = _import_aistudio()
                genai.configure(api_key=cur)
                self._last_configured = cur

    @staticmethod
    def _mask(k: str) -> str:
        if not k:
            return "(empty)"
        if len(k) <= 10:
            return k[:3] + "***"
        return k[:8] + "..." + k[-4:]


_POOL: Optional[_KeyPool] = None


def _get_pool() -> _KeyPool:
    global _POOL
    if _POOL is None:
        keys = GEMINI_CONFIG.get("api_keys") or []
        if not keys and GEMINI_CONFIG.get("api_key"):
            keys = [GEMINI_CONFIG["api_key"]]
        _POOL = _KeyPool(keys)
    return _POOL


def reset_pool() -> None:
    global _POOL, _VERTEX_INITED
    _POOL = None
    _VERTEX_INITED = False


# ============================================================
# 錯誤類型判斷（用於 AI Studio key 輪替）
# ============================================================
_ROTATE_PATTERNS = [
    r"API_KEY_INVALID",
    r"API[_ ]?key not found",
    r"API key not valid",
    r"permission_denied",
    r"PERMISSION_DENIED",
    r"RESOURCE_EXHAUSTED",
    r"quota.*exceed",
    r"rate.?limit",
    r"\b429\b",
    r"\b503\b",
    r"UNAUTHENTICATED",
]
_ROTATE_RE = re.compile("|".join(_ROTATE_PATTERNS), re.IGNORECASE)


def _should_rotate(err_text: str) -> bool:
    return bool(_ROTATE_RE.search(err_text))


# ============================================================
# Model 包裝層：兩個後端都長一樣的介面
# ============================================================
class _GeminiModelWrapper:
    """讓 vertex / aistudio model 對外有同樣的 generate_content 介面。"""

    def __init__(self, backend: str, raw_model, model_name: str, generation_config: dict):
        self.backend = backend
        self._raw = raw_model
        self.model_name = model_name
        self._generation_config = generation_config

    def generate_content(self, prompt):
        # 兩個後端的 generate_content 簽名實際相同；包一層方便除錯
        return self._raw.generate_content(prompt)


def init_gemini(
    model_name: Optional[str] = None,
    generation_config: Optional[dict] = None,
    api_key: Optional[str] = None,
) -> _GeminiModelWrapper:
    """依當前後端建立 model。介面對齊 MIS 風格。"""
    name = model_name or GEMINI_CONFIG.get("model", "gemini-2.5-flash")
    merged_gc = {
        "temperature": GEMINI_CONFIG.get("temperature", 0.7),
        "top_p": GEMINI_CONFIG.get("top_p", 0.8),
        "top_k": GEMINI_CONFIG.get("top_k", 40),
        "max_output_tokens": GEMINI_CONFIG.get("max_output_tokens", 8192),
    }
    if generation_config:
        merged_gc.update(generation_config)

    if _BACKEND == "vertex":
        _ensure_vertex_inited()
        _, VxGenerativeModel = _import_vertex()
        raw = VxGenerativeModel(name, generation_config=merged_gc)
    else:
        genai = _import_aistudio()
        if api_key:
            genai.configure(api_key=api_key)
        else:
            _get_pool().ensure_configured()
        raw = genai.GenerativeModel(name, generation_config=merged_gc)

    return _GeminiModelWrapper(_BACKEND, raw, name, merged_gc)


def generate_with_rotation(
    model: _GeminiModelWrapper,
    prompt: str,
    *,
    max_attempts: Optional[int] = None,
) -> str:
    """跑 generate_content，按後端做不同的錯誤處理：

    - Vertex: timeout + 內部 retry（網路錯誤 / 503 / 連線斷掉自動重試）
    - AI Studio: 遇到可恢復錯誤自動切下一個 key
    """
    if model.backend == "vertex":
        return _vertex_call_with_retry(model, prompt)

    # ===== AI Studio：多 key 輪替 =====
    pool = _get_pool()
    if max_attempts is None:
        max_attempts = max(1, len(pool.keys))

    model_name = model.model_name
    gc = model._generation_config
    genai = _import_aistudio()

    last_err: Optional[Exception] = None
    for _ in range(max_attempts):
        try:
            resp = model.generate_content(prompt)
            return _extract_text(resp)
        except Exception as e:
            last_err = e
            err_text = str(e)
            if not _should_rotate(err_text):
                raise
            next_key = pool.advance()
            if next_key is None:
                logger.error(
                    f"所有 {len(pool.keys)} 個 Gemini key 都失敗，最後錯誤: {err_text[:200]}"
                )
                raise
            pool.ensure_configured()
            new_raw = genai.GenerativeModel(model_name, generation_config=gc)
            model = _GeminiModelWrapper("aistudio", new_raw, model_name, gc)

    if last_err:
        raise last_err
    raise RuntimeError("generate_with_rotation: 未知失敗")


def call_gemini_api(
    prompt: str,
    model_name: Optional[str] = None,
    generation_config: Optional[dict] = None,
    ai_type: str = "gemini",
    max_attempts: Optional[int] = None,
) -> str:
    """單次 prompt → text，介面對齊 MIS call_gemini_api。"""
    if ai_type != "gemini":
        raise NotImplementedError(f"目前只支援 ai_type='gemini'，收到: {ai_type}")
    try:
        model = init_gemini(model_name=model_name, generation_config=generation_config)
        return generate_with_rotation(model, prompt, max_attempts=max_attempts).strip()
    except Exception as e:
        logger.error(f"call_gemini_api 失敗: {str(e)[:200]}")
        return ""


def get_backend() -> str:
    """給診斷工具查當前後端用。"""
    return _BACKEND


# ============================================================
# Vertex AI 專用：timeout + retry
# ============================================================
# Vertex 端可重試的錯誤模式
_VERTEX_RETRYABLE_PATTERNS = [
    # === 配額 / 限流（必須重試，這是 Vertex 最常見的暫時性錯誤）===
    r"\b429\b",                                # Too Many Requests
    r"RESOURCE_EXHAUSTED",                     # gRPC 配額用盡
    r"resource exhausted",
    r"quota.*exceed",
    r"rate.?limit",
    # === 伺服器端錯誤 ===
    r"\b503\b",                                # Service Unavailable
    r"\b502\b",                                # Bad Gateway
    r"\b504\b",                                # Gateway Timeout
    r"\b500\b",                                # Internal Server Error
    r"ServiceUnavailable",
    r"InternalServerError",
    r"UNAVAILABLE",                            # gRPC 服務不可用
    # === 網路 / 連線錯誤 ===
    r"connection (?:aborted|reset|refused)",   # 各種 connection 錯誤
    r"WSAGetOverlappedResult",                 # Windows socket 錯誤
    r"\b10053\b",                              # WSAECONNABORTED
    r"\b10054\b",                              # WSAECONNRESET
    r"DEADLINE_EXCEEDED",                      # gRPC 超時
    r"timed? *out",                            # 一般超時
    r"timeout",
]
_VERTEX_RETRYABLE_RE = re.compile("|".join(_VERTEX_RETRYABLE_PATTERNS), re.IGNORECASE)


def _is_vertex_retryable(err_text: str) -> bool:
    return bool(_VERTEX_RETRYABLE_RE.search(err_text))


# 共用 thread pool（給 timeout wrapper 用，避免每次呼叫都 create 新 pool）。
# 原本固定為 1 會讓上層的 chunk worker 全部假並行、實序列。
# 保留保守預設 4，並允許長批次建圖以環境變數調整；實際
# 請求啟動速率仍會受 _vertex_rate_limit_wait 的 RPM 限制。
_VERTEX_CONCURRENCY = max(1, int(os.getenv("VERTEX_CONCURRENCY", "4")))
_VERTEX_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=_VERTEX_CONCURRENCY)


def _call_with_timeout(callable_fn, timeout_sec: int):
    """用 thread 包呼叫，達到真正的 timeout。

    為何要 thread：vertexai.generative_models.GenerativeModel.generate_content
    不支援 request_options/timeout 參數（會被靜默吃掉），導致 Windows TCP
    預設 timeout 約 54 分鐘才會放棄連線。用 thread + future.result(timeout=)
    才能在 Python 層強制 timeout。

    注意：超時後底層 thread 仍會繼續跑直到自己結束，但對外已經拋 TimeoutError，
    主流程會走 retry 邏輯。
    """
    future = _VERTEX_EXECUTOR.submit(callable_fn)
    try:
        return future.result(timeout=timeout_sec)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(
            f"Vertex AI 呼叫超過 {timeout_sec} 秒未回應（thread timeout）"
        )


def _vertex_call_with_retry(model, prompt: str) -> str:
    """Vertex AI 呼叫 + 真正生效的 timeout + 指數退避重試。

    timeout 預設 90 秒（從 .env 的 VERTEX_TIMEOUT_SECONDS 讀），用 thread 包裝
    強制觸發，避免 Windows TCP 預設 54 分鐘才放棄。
    連線斷掉、503、超時、429 都會自動重試。
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, _VERTEX_MAX_RETRIES + 1):
        try:
            # 主動速率限制：每分鐘最多 _VERTEX_MAX_RPM 次，避免 429
            _vertex_rate_limit_wait()
            # thread-based timeout（vertexai SDK 沒有原生 timeout 參數）
            resp = _call_with_timeout(
                lambda: model.generate_content(prompt),
                _VERTEX_TIMEOUT,
            )
            return _extract_text(resp)
        except Exception as e:
            last_err = e
            err_text = str(e)
            if not _is_vertex_retryable(err_text):
                # 非網路錯誤（例如 prompt 格式錯）直接拋
                logger.error(f"Vertex AI 不可重試錯誤: {err_text[:200]}")
                raise
            if attempt >= _VERTEX_MAX_RETRIES:
                logger.error(
                    f"Vertex AI 重試 {_VERTEX_MAX_RETRIES} 次仍失敗: {err_text[:200]}"
                )
                raise
            backoff = _VERTEX_BACKOFF_BASE * (2 ** (attempt - 1))  # 3, 6, 12 秒
            logger.warning(
                f"Vertex AI 第 {attempt}/{_VERTEX_MAX_RETRIES} 次失敗 ({err_text[:100]})，"
                f"{backoff:.0f} 秒後重試"
            )
            time.sleep(backoff)

    if last_err:
        raise last_err
    raise RuntimeError("_vertex_call_with_retry: 未知失敗")


def _extract_text(response) -> str:
    """從 SDK response 取純文字，兩個後端都通用。"""
    try:
        if hasattr(response, "text") and response.text:
            return response.text
    except Exception:
        pass
    try:
        if hasattr(response, "candidates") and response.candidates:
            parts_text = []
            for cand in response.candidates:
                content = getattr(cand, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []) or []:
                    t = getattr(part, "text", None)
                    if t:
                        parts_text.append(t)
            if parts_text:
                return "\n".join(parts_text)
    except Exception:
        pass
    return ""
