"""跨書本概念合併。

用途：當第二本（第三本…）書本被 ingest 後，圖中可能出現
「行程 / 進程」、「二元樹 / 二叉樹」這類同義異名概念。
本模組會：
1. 從 Neo4j 把所有 Concept 取出，計算 (name, definition) 的 embedding
2. 用 cosine 相似度找出 top-N 最相似的候選 pair (跨書本/跨章節)
3. 把候選 pair 一批批送 Gemini 判斷 same / subset / different
4. 對 same 的：呼叫 Neo4jWriter.merge_concept_alias 合併
5. 對 subset 的：建立 HAS_SUBTYPE 邊
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from config import settings
from graphrag.embedder import GeminiEmbedder
from kg_builder.neo4j_writer import Neo4jWriter
from kg_builder.prompts import CROSS_BOOK_MERGE_SYSTEM, CROSS_BOOK_MERGE_USER
from utils.cypher_queries import LINK_HAS_SUBTYPE
from utils.gemini_client import init_gemini, generate_with_rotation
from utils.logger import get_logger

logger = get_logger(__name__)


# ==========================================================================
# Schemas
# ==========================================================================
class MergeDecision(BaseModel):
    a: str
    b: str
    relation: str  # same | subset | different
    canonical: Optional[str] = None
    child: Optional[str] = None
    parent: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class MergeResponse(BaseModel):
    decisions: List[MergeDecision] = Field(default_factory=list)


@dataclass
class MergeStats:
    candidates: int = 0
    merged_same: int = 0
    added_subtype: int = 0
    skipped_different: int = 0


# ==========================================================================
# Merger
# ==========================================================================
class CrossBookMerger:
    def __init__(
        self,
        writer: Optional[Neo4jWriter] = None,
        embedder: Optional[GeminiEmbedder] = None,
        model_name: Optional[str] = None,
        sim_threshold: float = 0.86,
        confidence_threshold: float = 0.85,
        candidates_per_concept: int = 5,
        batch_size: int = 10,
    ):
        self.writer = writer  # 可由外部傳入；若 None 會自建
        self.embedder = embedder or GeminiEmbedder()
        self.model_name = model_name or settings.gemini_model
        self.sim_threshold = sim_threshold
        self.confidence_threshold = confidence_threshold
        self.candidates_per_concept = candidates_per_concept
        self.batch_size = batch_size

        # 統一走 utils.gemini_client，合併判斷需要嚴格的 JSON + 0.0 temperature
        self.model = init_gemini(
            model_name=self.model_name,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        )

    # ----------------------------------------------------------------------
    def run(self, only_cross_book: bool = True, dry_run: bool = False) -> MergeStats:
        owned_writer = False
        if self.writer is None:
            self.writer = Neo4jWriter()
            self.writer.connect()
            owned_writer = True

        try:
            concepts = self._load_concepts()
            logger.info(f"載入 {len(concepts)} 個 Concept")
            if len(concepts) < 2:
                return MergeStats()

            # MOD 新增: 先跑 collapsed-key fast merge（不用 Gemini）
            # 例如 "VirtualMemory" vs "virtualmemory" vs "Virtual_Memory"
            # 這種只有大小寫/底線/空格差異的直接合併，不浪費 Gemini call
            fast_merged = self._fast_collapsed_key_merge(concepts, dry_run)
            if fast_merged:
                logger.info(f"⚡ Fast merge 合併 {fast_merged} 個 collapsed-key 相同的節點")
                # 重新載入合併後的 concepts
                concepts = self._load_concepts()

            pairs = self._build_candidate_pairs(concepts, only_cross_book)
            logger.info(f"候選 pair 數: {len(pairs)}")
            if not pairs:
                return MergeStats(candidates=0, merged_same=fast_merged)

            decisions = self._call_llm_in_batches(pairs, concepts)
            stats = MergeStats(candidates=len(pairs), merged_same=fast_merged)
            self._apply_decisions(decisions, stats, dry_run)
            return stats
        finally:
            if owned_writer and self.writer is not None:
                self.writer.close()

    def _fast_collapsed_key_merge(self, concepts: List[dict], dry_run: bool) -> int:
        """MOD: 對 collapsed-key（大小寫/底線/空格全去掉）完全相同的概念直接合併。
        不呼叫 Gemini，0 成本，適合處理 VirtualMemory / virtual memory / Virtual_Memory。
        """
        groups: Dict[str, List[str]] = {}
        for c in concepts:
            name = c.get("name") or ""
            key = re.sub(r'[^A-Za-z0-9]', '', name).lower()
            if not key:
                continue
            groups.setdefault(key, []).append(name)
        merged_count = 0
        for key, names in groups.items():
            if len(names) < 2:
                continue
            # 選 canonical: 內部大寫最多、長度最長
            def score(n: str) -> Tuple[int, int]:
                internal_caps = sum(1 for i in range(1, len(n)) if n[i].isupper())
                return (internal_caps, len(n))
            canonical = max(names, key=score)
            others = [n for n in names if n != canonical]
            for other in others:
                logger.info(f"⚡ Fast merge: {other} -> {canonical} (collapsed_key={key})")
                if not dry_run:
                    try:
                        self.writer._run(
                            "MATCH (keep:Concept {name: $keep}), (drop:Concept {name: $drop}) "
                            "CALL apoc.refactor.mergeNodes([keep, drop], "
                            "{properties:'combine', mergeRels:true}) YIELD node RETURN node",
                            keep=canonical, drop=other
                        )
                        merged_count += 1
                    except Exception as e:
                        logger.warning(f"Fast merge {other}->{canonical} 失敗: {e}")
        return merged_count

    # ----------------------------------------------------------------------
    def _load_concepts(self) -> List[dict]:
        rows = self.writer._run(
            "MATCH (c:Concept) "
            "RETURN c.name AS name, c.definition AS definition, "
            "c.category AS category, c.bookIds AS bookIds"
        )
        return rows

    def _build_candidate_pairs(
        self,
        concepts: List[dict],
        only_cross_book: bool,
    ) -> List[Tuple[str, str, float]]:
        # 計算 embedding
        names = [c["name"] for c in concepts]
        texts = [
            f"{c['name']}: {(c.get('definition') or '')[:200]}" for c in concepts
        ]
        vecs = self.embedder.embed_documents(texts)

        # 簡單 O(N^2) 餘弦 (對 N≤幾千夠用)
        import math
        def norm(v): return math.sqrt(sum(x * x for x in v)) or 1.0
        norms = [norm(v) for v in vecs]
        bookids_per = {c["name"]: set(c.get("bookIds") or []) for c in concepts}
        pairs: List[Tuple[str, str, float]] = []

        for i, j in combinations(range(len(concepts)), 2):
            ni, nj = names[i], names[j]
            if ni == nj:
                continue
            if only_cross_book:
                bi, bj = bookids_per.get(ni, set()), bookids_per.get(nj, set())
                if bi and bj and bi.isdisjoint(bj) is False:
                    # 完全同書 → 已在書內合併，這裡只看跨書
                    if not bi - bj and not bj - bi:
                        continue
            sim = sum(a * b for a, b in zip(vecs[i], vecs[j])) / (norms[i] * norms[j])
            if sim >= self.sim_threshold:
                pairs.append((ni, nj, sim))

        # 對每個概念只保留前 K 個最相似的鄰居 (避免爆炸)
        by_concept: Dict[str, List[Tuple[str, str, float]]] = {}
        for a, b, s in sorted(pairs, key=lambda x: -x[2]):
            by_concept.setdefault(a, []).append((a, b, s))
            by_concept.setdefault(b, []).append((a, b, s))
        kept: List[Tuple[str, str, float]] = []
        seen = set()
        for a, lst in by_concept.items():
            for trip in lst[: self.candidates_per_concept]:
                key = tuple(sorted([trip[0], trip[1]]))
                if key in seen:
                    continue
                seen.add(key)
                kept.append(trip)
        return kept

    # ----------------------------------------------------------------------
    def _call_llm_in_batches(
        self,
        pairs: List[Tuple[str, str, float]],
        concepts: List[dict],
    ) -> List[MergeDecision]:
        defs = {c["name"]: c.get("definition") or "" for c in concepts}
        out: List[MergeDecision] = []
        for start in tqdm(range(0, len(pairs), self.batch_size), desc="跨書 LLM 比對"):
            batch = pairs[start : start + self.batch_size]
            cand_block = "\n".join(
                f"({i+1}) A=「{a}」: {defs.get(a,'')[:120]}\n"
                f"     B=「{b}」: {defs.get(b,'')[:120]}\n"
                f"     embedding 相似度: {s:.3f}"
                for i, (a, b, s) in enumerate(batch)
            )
            prompt = (
                CROSS_BOOK_MERGE_SYSTEM + "\n\n"
                + CROSS_BOOK_MERGE_USER.format(candidates=cand_block)
            )
            try:
                # 走多 key 自動輪替
                raw = generate_with_rotation(self.model, prompt) or ""
            except Exception as e:
                logger.warning(f"Gemini 呼叫失敗: {e}")
                continue
            data = self._parse_json_safe(raw)
            if data is None:
                continue
            try:
                resp_obj = MergeResponse.model_validate(data)
                out.extend(resp_obj.decisions)
            except ValidationError as e:
                logger.warning(f"MergeResponse 驗證失敗: {e}")
        return out

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

    # ----------------------------------------------------------------------
    def _apply_decisions(
        self,
        decisions: List[MergeDecision],
        stats: MergeStats,
        dry_run: bool,
    ) -> None:
        results_log: List[dict] = []
        for d in decisions:
            log = d.model_dump()
            results_log.append(log)
            if d.confidence < self.confidence_threshold:
                stats.skipped_different += 1
                continue
            if d.relation == "same":
                # 決定 canonical 與 alias
                canonical = d.canonical or (d.a if len(d.a) <= len(d.b) else d.b)
                alias = d.b if canonical == d.a else d.a
                if dry_run:
                    logger.info(f"[DRY] 將合併 [{alias}] → [{canonical}]: {d.reason}")
                else:
                    assert self.writer is not None
                    self.writer.merge_concept_alias(alias, canonical)
                stats.merged_same += 1
            elif d.relation == "subset":
                p = d.parent or d.a
                c = d.child or d.b
                if dry_run:
                    logger.info(f"[DRY] 加 HAS_SUBTYPE [{p}] → [{c}]")
                else:
                    assert self.writer is not None
                    try:
                        self.writer._run(LINK_HAS_SUBTYPE,
                                         parent=p, child=c, source="cross_book_merge")
                    except Exception as e:
                        logger.warning(f"加 HAS_SUBTYPE 失敗 {p}->{c}: {e}")
                stats.added_subtype += 1
            else:
                stats.skipped_different += 1

        # 把判決寫到 outputs 方便人工複核
        try:
            settings.ensure_dirs()
            out_path = settings.output_dir / "cross_book_merge_decisions.json"
            out_path.write_text(
                json.dumps(results_log, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(f"判決紀錄寫到 {out_path}")
        except Exception as e:
            logger.warning(f"寫判決紀錄失敗: {e}")
