"""使用 Google Gemini 從文字 chunk 同時抽取：
   1. 細分後的知識點 (含 parent/subtypes)
   2. 子類關係 HAS_SUBTYPE
   3. 同片段內可立即判定的先輩關係 PREREQUISITE_OF

每個 chunk 一次 Gemini 呼叫即可拿到三種輸出，避免後續再做大量 pair 比對。
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationError, model_validator
from tqdm import tqdm

from config import settings
from kg_builder.prompts import CONCEPT_EXTRACTION_SYSTEM, CONCEPT_EXTRACTION_USER
from pdf_processor.text_chunker import Chunk
from utils.gemini_client import init_gemini, generate_with_rotation
from utils.logger import get_logger

logger = get_logger(__name__)


def _evidence_key(value: str) -> str:
    """Normalize PDF line breaks and punctuation for literal-evidence checks."""
    return re.sub(r"[\W_]+", "", str(value or "").casefold(), flags=re.UNICODE)


# ==========================================================================
# Schemas
# ==========================================================================
class Concept(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    definition: str = Field(default="")
    aliases: List[str] = Field(default_factory=list)
    category: str = Field(default="其他")
    parent: Optional[str] = None
    is_fine_grained: bool = False
    # Defaults preserve compatibility with legacy import scripts.  LLM output
    # uses ExtractionConcept below, where these fields are required.
    keep: bool = True
    keep_reason: str = Field(default="", max_length=1000)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_evidence: str = Field(default="", max_length=2000)

    # 來源資訊（後續由 extractor 填入）
    chunk_id: Optional[str] = None
    chapter_id: Optional[str] = None
    chapter_order: Optional[int] = None
    book_id: Optional[str] = None

    def normalize_name(self) -> str:
        return self.name.strip()


class ExtractionConcept(Concept):
    """Strict schema used only for new LLM extraction responses."""

    keep: bool
    keep_reason: str = Field(..., min_length=1, max_length=1000)
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_evidence: str = Field(..., min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_kept_concept(self) -> "ExtractionConcept":
        """Fail closed when a formal concept lacks auditable evidence."""
        if not self.keep:
            return self
        missing = []
        if not self.definition.strip():
            missing.append("definition")
        if not self.keep_reason.strip():
            missing.append("keep_reason")
        if self.confidence <= 0:
            missing.append("confidence")
        if not self.source_evidence.strip():
            missing.append("source_evidence")
        if missing:
            raise ValueError(
                "keep=true concept is missing required quality fields: "
                + ", ".join(missing)
            )
        return self


class SubtypeEdge(BaseModel):
    parent: str
    child: str


class LocalPrerequisiteEdge(BaseModel):
    prereq: str
    target: str
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    reason: str = ""
    source: str = "extraction"     # 來自抽取階段 (相對於 cross-chunk inference)


class ExtractionResult(BaseModel):
    concepts: List[ExtractionConcept] = Field(default_factory=list)
    rejected_concepts: List[ExtractionConcept] = Field(default_factory=list)
    subtype_edges: List[SubtypeEdge] = Field(default_factory=list)
    local_prerequisites: List[LocalPrerequisiteEdge] = Field(default_factory=list)


# ==========================================================================
# Extractor
# ==========================================================================
class ConceptExtractor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_concepts_per_chunk: Optional[int] = None,
        max_retries: int = 5,
        retry_delay: float = 3.0,
    ):
        self.model_name = model_name or settings.gemini_model
        self.max_concepts = max_concepts_per_chunk or settings.max_concepts_per_chunk
        self.max_retries = max(1, max_retries)
        self.retry_delay = max(0.0, retry_delay)
        self.failed_chunks: List[dict] = []
        # 統一走 utils.gemini_client.init_gemini，集中讀 GEMINI_CONFIG
        # 抽取需要 JSON 結構化輸出，因此覆寫 generation_config
        self.model = init_gemini(
            model_name=self.model_name,
            generation_config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
            api_key=api_key,  # 仍允許 caller 顯式指定
        )

    # ----------------------------------------------------------------------
    def extract_from_chunk(
        self,
        chunk: Chunk,
        exclude_concepts: Optional[Sequence[str]] = None,
        gleaning_round: int = 0,
    ) -> ExtractionResult:
        """抽取單一 chunk。

        重要修正：
        - JSON 解析失敗 / schema 校驗失敗 / Gemini 呼叫失敗時，不再立刻跳過。
        - 會依 self.max_retries 自動重試。
        - 多次失敗後才回傳空結果，並把失敗資訊寫入 outputs/failed_chunks.jsonl。
        """
        base_prompt = (
            CONCEPT_EXTRACTION_SYSTEM.format(max_concepts=self.max_concepts)
            + "\n\n"
            + CONCEPT_EXTRACTION_USER.format(
                chapter_title=chunk.chapter_title,
                chunk_text=chunk.text,
                max_concepts=self.max_concepts,
            )
        )
        excluded_names = {
            str(name).strip() for name in (exclude_concepts or []) if str(name).strip()
        }
        if excluded_names:
            base_prompt += (
                "\n\nADDITIONAL GLEANING PASS " + str(gleaning_round) + ".\n"
                "The following concepts were already extracted from this exact text chunk:\n"
                + json.dumps(sorted(excluded_names), ensure_ascii=False)
                + "\nDo not output those concepts again. Extract only important concepts "
                  "that were missed in the earlier pass, up to the stated limit. "
                  "Prefer technically meaningful textbook concepts over incidental words. "
                  "Subtype and prerequisite edges may connect a newly extracted concept "
                  "to one of the already-extracted concept names above. Return the same JSON schema."
            )

        last_error = ""
        last_raw = ""

        for attempt in range(1, self.max_retries + 1):
            prompt = base_prompt
            if attempt > 1:
                prompt += (
                    "\n\n【重試要求】\n"
                    "上一輪輸出無法被程式解析。請只輸出單一合法 JSON 物件，"
                    "不要使用 Markdown code fence，不要加入解釋文字。"
                    "JSON 必須包含 concepts、subtype_edges、local_prerequisites 三個欄位；"
                    "concepts 中每個候選都必須包含 keep、keep_reason、confidence、"
                    "source_evidence；"
                    "沒有資料時請使用空陣列 []。"
                )

            try:
                logger.info(
                    f"開始處理 chunk {chunk.chunk_id} "
                    f"(attempt {attempt}/{self.max_retries})"
                )
                # generate_with_rotation 會自動偵測 INVALID_KEY / 429 / 503 切下一個 key
                raw = generate_with_rotation(self.model, prompt) or ""
                last_raw = raw
            except Exception as e:
                last_error = f"Gemini 呼叫失敗: {e}"
                logger.warning(
                    f"chunk {chunk.chunk_id} 第 {attempt}/{self.max_retries} 次失敗：{last_error}"
                )
                self._sleep_before_retry(attempt)
                continue

            parsed = self._parse_json_safe(raw)
            if parsed is None:
                last_error = "JSON 解析失敗"
                logger.warning(
                    f"chunk {chunk.chunk_id} 第 {attempt}/{self.max_retries} 次 JSON 解析失敗，準備重試。"
                )
                self._sleep_before_retry(attempt)
                continue

            try:
                result = ExtractionResult.model_validate(parsed)
            except ValidationError as e:
                last_error = f"schema 校驗失敗: {e}"
                logger.warning(
                    f"chunk {chunk.chunk_id} 第 {attempt}/{self.max_retries} 次 schema 校驗失敗，準備重試: {e}"
                )
                self._sleep_before_retry(attempt)
                continue

            # keep=false 候選不得進入正式圖譜，但保留完整稽核資訊。
            all_candidates = list(result.concepts)
            chunk_evidence_key = _evidence_key(chunk.text)
            for candidate in all_candidates:
                evidence_key = _evidence_key(candidate.source_evidence)
                if (
                    candidate.keep
                    and (
                        len(evidence_key) < 4
                        or evidence_key not in chunk_evidence_key
                    )
                ):
                    candidate.keep = False
                    candidate.keep_reason = (
                        candidate.keep_reason.rstrip()
                        + "｜程式淘汰：source_evidence 無法在目前 Chunk 原文中定位。"
                    )
            accepted = [
                c
                for c in all_candidates
                if c.keep and c.normalize_name() not in excluded_names
            ]
            rejected = [
                c
                for c in all_candidates
                if not c.keep and c.normalize_name() not in excluded_names
            ]
            result.concepts = accepted
            result.rejected_concepts.extend(rejected)

            # 補齊正式與淘汰候選的來源資訊。
            for c in [*result.concepts, *result.rejected_concepts]:
                c.chunk_id = chunk.chunk_id
                c.chapter_id = chunk.chapter_id
                c.chapter_order = chunk.chapter_order
                c.book_id = chunk.book_id
                c.name = c.normalize_name()

            # 過濾無效邊：兩端必須是正式概念或前一輪已接受的正式概念。
            names = {c.name for c in result.concepts} | excluded_names
            result.subtype_edges = [
                e for e in result.subtype_edges
                if e.parent in names and e.child in names and e.parent != e.child
            ]
            result.local_prerequisites = [
                e for e in result.local_prerequisites
                if e.prereq in names and e.target in names and e.prereq != e.target
            ]
            return result

        logger.error(
            f"chunk {chunk.chunk_id} 重試 {self.max_retries} 次仍失敗，已記錄到 failed_chunks.jsonl。"
        )
        self._record_failed_chunk(chunk, last_error, last_raw)
        return ExtractionResult()

    def _sleep_before_retry(self, attempt: int) -> None:
        """最後一次失敗不需要 sleep；其餘使用指數退避降低 429 壓力。"""
        if attempt < self.max_retries and self.retry_delay > 0:
            time.sleep(self.retry_delay * (2 ** (attempt - 1)))

    def _record_failed_chunk(self, chunk: Chunk, reason: str, raw: str = "") -> None:
        """把最終仍失敗的 chunk 記錄下來，方便之後單獨補跑或人工檢查。"""
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "book_id": chunk.book_id,
            "chapter_id": chunk.chapter_id,
            "chapter_title": chunk.chapter_title,
            "chapter_order": chunk.chapter_order,
            "chunk_id": chunk.chunk_id,
            "reason": reason,
            "text_preview": (chunk.text or "")[:1200],
            "raw_response_preview": (raw or "")[:2000],
        }
        self.failed_chunks.append(record)

        try:
            settings.output_dir.mkdir(parents=True, exist_ok=True)
            fail_path = settings.output_dir / "failed_chunks.jsonl"
            with fail_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"寫入 failed_chunks.jsonl 失敗: {e}")

    def extract_all(
        self, chunks: List[Chunk]
    ) -> Tuple[List[Concept], List[SubtypeEdge], List[LocalPrerequisiteEdge]]:
        all_concepts: List[Concept] = []
        all_subtypes: List[SubtypeEdge] = []
        all_prereqs: List[LocalPrerequisiteEdge] = []
        for ck in tqdm(chunks, desc="抽取知識點 + 子類 + 先輩"):
            r = self.extract_from_chunk(ck)
            all_concepts.extend(r.concepts)
            all_subtypes.extend(r.subtype_edges)
            all_prereqs.extend(r.local_prerequisites)
        if self.failed_chunks:
            logger.warning(
                f"本次共有 {len(self.failed_chunks)} 個 chunk 多次重試後仍失敗，"
                f"已記錄於 {settings.output_dir / 'failed_chunks.jsonl'}"
            )
        logger.info(
            f"共抽出 {len(all_concepts)} 條原始概念 / "
            f"{len(all_subtypes)} 條子類邊 / "
            f"{len(all_prereqs)} 條同片段先輩邊"
        )
        return all_concepts, all_subtypes, all_prereqs

    # ----------------------------------------------------------------------
    @staticmethod
    def _parse_json_safe(raw: str) -> Optional[dict]:
        if not raw:
            return None
        m = re.search(r"```(?:json)?\s*(.+?)\s*```", raw, re.S)
        if m:
            raw = m.group(1)
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    return None
        return None


# ==========================================================================
# 去重 + 跨書本合併同名概念 (一階段)
# ==========================================================================
def deduplicate_concepts(concepts: List[Concept]) -> List[Concept]:
    """以正規化 name 為 key 合併同名概念。
    aliases 取聯集；definition 取較長；category/parent 取最具體者；
    book_ids 累積成 list (跨書共現)。"""
    bucket: Dict[str, Concept] = {}
    book_ids_per: Dict[str, set] = {}

    for c in concepts:
        key = _normalize_key(c.name)
        if not key:
            continue
        if key not in bucket:
            bucket[key] = c.model_copy(deep=True)
            book_ids_per[key] = set()
        else:
            cur = bucket[key]
            # aliases 聯集 (含舊名)
            merged_aliases = set(cur.aliases or [])
            merged_aliases.update(c.aliases or [])
            if c.name != cur.name:
                merged_aliases.add(c.name)
            cur.aliases = sorted(a for a in merged_aliases if a and a != cur.name)
            # definition 取較長
            if len(c.definition or "") > len(cur.definition or ""):
                cur.definition = c.definition
            # category 取較具體
            if cur.category in ("其他", "") and c.category not in ("其他", ""):
                cur.category = c.category
            # parent
            if not cur.parent and c.parent:
                cur.parent = c.parent
            # is_fine_grained 取 OR
            cur.is_fine_grained = bool(cur.is_fine_grained or c.is_fine_grained)
            # 稽核欄位保留信心較高的來源判定。
            if c.confidence > cur.confidence:
                cur.keep_reason = c.keep_reason
                cur.confidence = c.confidence
                cur.source_evidence = c.source_evidence
        if c.book_id:
            book_ids_per[key].add(c.book_id)

    # 把 book_ids 寫進 model 的 extra (透過 aliases 暫存？不行，獨立屬性)
    # 改用一個外部 dict 回傳給 writer，不污染 model。
    # 這邊把 book_ids 編碼進 c.aliases 的尾端會混淆，所以用 setattr。
    deduped: List[Concept] = []
    for key, c in bucket.items():
        # Pydantic v2: 透過 model_dump + 附加欄位
        c.__dict__["_book_ids"] = sorted(book_ids_per.get(key, set()))
        deduped.append(c)

    logger.info(f"去重後共 {len(deduped)} 個獨立知識點")
    return deduped


def get_concept_book_ids(c: Concept) -> List[str]:
    return list(c.__dict__.get("_book_ids", []) or [])


def _normalize_key(name: str) -> str:
    return (name or "").strip().lower().replace(" ", "")


# ==========================================================================
# 子類邊 / 先輩邊去重
# ==========================================================================
def deduplicate_subtype_edges(edges: List[SubtypeEdge]) -> List[SubtypeEdge]:
    seen = set()
    out: List[SubtypeEdge] = []
    for e in edges:
        key = (_normalize_key(e.parent), _normalize_key(e.child))
        if key in seen or key[0] == key[1]:
            continue
        seen.add(key)
        out.append(e)
    logger.info(f"子類邊去重: {len(edges)} → {len(out)}")
    return out


def deduplicate_local_prerequisites(
    edges: List[LocalPrerequisiteEdge],
) -> List[LocalPrerequisiteEdge]:
    """同方向 pair 取 confidence 最高那筆；若同 pair 兩個方向都有，取信心較高者。"""
    best: Dict[tuple, LocalPrerequisiteEdge] = {}
    for e in edges:
        key = (_normalize_key(e.prereq), _normalize_key(e.target))
        rev_key = (key[1], key[0])
        if key[0] == key[1]:
            continue
        # 與反向比較
        if rev_key in best:
            if e.confidence > best[rev_key].confidence:
                del best[rev_key]
                best[key] = e
            continue
        if key not in best or e.confidence > best[key].confidence:
            best[key] = e
    logger.info(f"先輩邊去重 (含反向衝突仲裁): {len(edges)} → {len(best)}")
    return list(best.values())
