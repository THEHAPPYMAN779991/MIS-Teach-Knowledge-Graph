"""KCEP Phase 2: Knowledge Concept Validation

對每個候選概念跑 8 條規則評分，>= 6 分才保留為正式 Concept。
同時判定 Importance = Core / Supporting / Mention。

用法:
    from kg_builder.validate_concepts import validate_candidate_concepts

    validated = validate_candidate_concepts(
        raw_concepts,          # Stage 2 抽出的候選
        chunks,                # 對應教材 chunks
        checkpoint_path,
        resume=True,
    )
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from utils.gemini_client import init_gemini


VALIDATION_SYSTEM_PROMPT = """你是「計算機教材知識圖譜」的概念驗證專家。

任務:對每個候選概念,判定是否應保留為 GraphRAG 的正式知識點。

# 8 條判定規則 (各 0 或 1 分)
1. formal_definition: 是否有正式定義? (例: Semaphore=1, Integer=0)
2. main_content: 是否是本章教材主要教授內容? (例: CPU_Scheduling=1, Initial_Value=0)
3. student_needs_learn: 學生是否需要學習? (例: Critical_Section=1, Example_P1=0)
4. independent_teaching_meaning: 是否具有獨立教學意義? (例: Deadlock=1, Blue_Process=0)
5. exam_material: 是否可能形成考題? (例: Semaphore=1, Integer=0)
6. graph_relation: 是否可建立 Graph Relation? (例: Semaphore-USED_FOR-Sync=1, Integer 無合理 rel=0)
7. not_descriptive_only: 是否非只是描述另一概念? (例: Semaphore=1, Integer_Variable 是 Semaphore 定義的一部分=0)
8. cross_chunk_reuse: 是否具跨 Chunk 重用價值? (例: Semaphore 在 20 chunk 出現=1, Example_2 只一次=0)

# 分數判定
- total >= 6: keep=true (進 Neo4j 正式節點)
- total 4-5: keep=candidate_review (放待審)
- total <= 3: keep=false (踢除或降級為 Mention)

# Importance 分層 (只對 keep=true)
- Core: 至少符合以下 2 條
  - 出現在章節標題 (chapter_title)
  - 有 glossary 定義
  - 跨 >= 5 chunk 出現
  - 可建 >= 3 條 relation
- Supporting: keep=true 但不符 Core 條件
- Mention: keep=false 但仍有名字識別價值 (存 Chunk.keywords, 不進 Neo4j 節點)

# 輸出格式 (JSON array, 每個候選一筆)
[
  {
    "name": "Semaphore",
    "scores": {
      "formal_definition": 1,
      "main_content": 1,
      "student_needs_learn": 1,
      "independent_teaching_meaning": 1,
      "exam_material": 1,
      "graph_relation": 1,
      "not_descriptive_only": 1,
      "cross_chunk_reuse": 1
    },
    "total_score": 8,
    "keep": true,
    "importance": "Core",
    "importance_reasons": ["chapter_title_mention", "cross_chunk_reuse"],
    "reject_reason": ""
  },
  {
    "name": "Integer",
    "scores": {"formal_definition": 0, "main_content": 0, ...},
    "total_score": 1,
    "keep": false,
    "importance": "Mention",
    "reject_reason": "純資料型別,是 Semaphore 定義的一部分,非獨立教學單位"
  }
]

只輸出 JSON,不要 markdown code block。"""


VALIDATION_USER_TEMPLATE = """本章節: {chapter_title}
教材片段 (供參考):
\"\"\"
{chunk_text}
\"\"\"

以下是 Stage 1 抽出的 {n_candidates} 個候選概念,請逐一判定:

{candidates_list}

回傳 JSON array,順序需與候選列表一致。"""


@dataclass
class ValidationResult:
    name: str
    scores: Dict[str, int] = field(default_factory=dict)
    total_score: int = 0
    keep: bool = False
    importance: str = "Mention"
    importance_reasons: List[str] = field(default_factory=list)
    reject_reason: str = ""


def _extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    """從 Gemini 回應抽 JSON array"""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\[.+\])\s*```", text, re.S)
    if m:
        text = m.group(1)
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _format_candidates_for_prompt(candidates: List[Dict[str, Any]]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        name = c.get("name", "")
        definition = (c.get("definition") or "")[:100]
        category = c.get("category") or ""
        lines.append(f"{i}. {name} [{category}]")
        if definition:
            lines.append(f"   定義: {definition}")
    return "\n".join(lines)


def _validate_batch(
    model,
    chunk_text: str,
    chapter_title: str,
    candidates: List[Dict[str, Any]],
) -> List[ValidationResult]:
    """對一批候選送 Gemini 驗證"""
    prompt = VALIDATION_SYSTEM_PROMPT + "\n\n" + VALIDATION_USER_TEMPLATE.format(
        chapter_title=chapter_title,
        chunk_text=chunk_text[:3000],  # 限制 chunk 長度避免爆 token
        n_candidates=len(candidates),
        candidates_list=_format_candidates_for_prompt(candidates),
    )
    try:
        response = model.generate_content(prompt)
        raw = str(getattr(response, "text", "") or "")
    except Exception as e:
        print(f"[validate] Gemini 失敗: {e}")
        return [ValidationResult(name=c.get("name", ""), keep=True, importance="Supporting")
                for c in candidates]  # fallback: 全保留

    parsed = _extract_json_array(raw)
    if not parsed or len(parsed) < len(candidates):
        return [ValidationResult(name=c.get("name", ""), keep=True, importance="Supporting")
                for c in candidates]  # fallback

    results = []
    for i, c in enumerate(candidates):
        name = c.get("name", "")
        v = parsed[i] if i < len(parsed) else {}
        # 對齊 name (Gemini 有時會改名)
        if v.get("name", "").lower() != name.lower():
            # 名字不一致,取 candidate 的原名
            v["name"] = name
        results.append(ValidationResult(
            name=name,
            scores=v.get("scores") or {},
            total_score=int(v.get("total_score") or 0),
            keep=bool(v.get("keep")),
            importance=str(v.get("importance") or "Mention"),
            importance_reasons=list(v.get("importance_reasons") or []),
            reject_reason=str(v.get("reject_reason") or ""),
        ))
    return results


def _group_candidates_by_chunk(candidates: List[Any]) -> Dict[str, List[Any]]:
    """把候選按所屬 chunk 分組"""
    groups: Dict[str, List[Any]] = {}
    for c in candidates:
        chunk_ids = getattr(c, "_source_chunk_ids", None)
        if chunk_ids:
            for cid in chunk_ids:
                groups.setdefault(cid, []).append(c)
        else:
            groups.setdefault("_orphan", []).append(c)
    return groups


def validate_candidate_concepts(
    candidates: List[Dict[str, Any]],
    chunks: List[Any],
    checkpoint: Path,
    resume: bool = True,
    workers: int = 3,
    batch_size: int = 10,
    model_name: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """跑 Phase 2 驗證,回傳 (kept_concepts, mention_concepts, rejected_concepts)

    kept_concepts: keep=true 的,含 importance 分層
    mention_concepts: keep=false 但保留為 Mention (存 Chunk.keywords)
    rejected_concepts: 完全排除,含 reject_reason
    """
    if not candidates:
        return [], [], []

    # 建 chunk_id -> chunk_text 索引
    chunk_index = {c.chunk_id: (c.text, c.chapter_title) for c in chunks}

    # 建 concept_name -> concept_dict 索引 (原始候選)
    name_to_candidate: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        if isinstance(c, dict):
            name = c.get("name")
        else:
            name = getattr(c, "name", None)
            c = c.model_dump() if hasattr(c, "model_dump") else dict(c.__dict__)
        if name:
            name_to_candidate.setdefault(name, c)

    unique_candidates = list(name_to_candidate.values())
    print(f"[validate] 收到 {len(candidates)} 個候選,去重後 {len(unique_candidates)} 個獨立名字")

    # 選一個代表性 chunk 給 Gemini 判斷 (取 candidate 第一個 source_chunk)
    # 若沒 source_chunk,用第一個 chunk fallback
    fallback_chunk_id = chunks[0].chunk_id if chunks else ""
    fallback_text, fallback_title = chunk_index.get(fallback_chunk_id, ("", ""))

    model = init_gemini(
        model_name=model_name or settings.gemini_model,
        generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
    )

    # 分批驗證
    all_results: Dict[str, ValidationResult] = {}
    # 依 source_chunk 分組批次
    batches: List[Tuple[str, List[Dict[str, Any]]]] = []
    for i in range(0, len(unique_candidates), batch_size):
        batches.append((fallback_chunk_id, unique_candidates[i:i + batch_size]))

    print(f"[validate] 共 {len(batches)} 個驗證批次,workers={workers}")

    def _process_batch(batch_idx: int, chunk_id: str, batch_candidates: List[Dict[str, Any]]) -> Tuple[int, List[ValidationResult]]:
        text, title = chunk_index.get(chunk_id, (fallback_text, fallback_title))
        results = _validate_batch(model, text, title, batch_candidates)
        return batch_idx, results

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_process_batch, i, chunk_id, batch): i
            for i, (chunk_id, batch) in enumerate(batches)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                _, results = fut.result()
                for r in results:
                    all_results[r.name] = r
                print(f"  [validate {idx + 1}/{len(batches)}] done ({len(results)} results)")
            except Exception as e:
                print(f"  [validate {idx + 1}/{len(batches)}] failed: {e}")

    # 分類
    kept = []
    mention = []
    rejected = []
    for candidate in unique_candidates:
        name = candidate.get("name")
        result = all_results.get(name)
        if not result:
            # 沒驗證結果 fallback keep (保守)
            candidate["importance"] = "Supporting"
            candidate["validation_score"] = -1
            kept.append(candidate)
            continue
        candidate["validation_score"] = result.total_score
        candidate["validation_scores_detail"] = result.scores
        candidate["importance"] = result.importance
        candidate["importance_reasons"] = result.importance_reasons
        candidate["validation_keep"] = result.keep
        candidate["validation_reject_reason"] = result.reject_reason
        if result.keep and result.importance in ("Core", "Supporting"):
            kept.append(candidate)
        elif result.importance == "Mention":
            mention.append(candidate)
        else:
            rejected.append(candidate)

    # 落地 checkpoint
    try:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("w", encoding="utf-8") as f:
            for c in kept + mention + rejected:
                f.write(json.dumps({
                    "name": c.get("name"),
                    "keep": c.get("validation_keep"),
                    "importance": c.get("importance"),
                    "score": c.get("validation_score"),
                    "reject_reason": c.get("validation_reject_reason", ""),
                }, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[validate] checkpoint 寫入失敗: {e}")

    print(f"[validate] 完成: kept={len(kept)} (Core+Supporting), mention={len(mention)}, rejected={len(rejected)}")
    return kept, mention, rejected
