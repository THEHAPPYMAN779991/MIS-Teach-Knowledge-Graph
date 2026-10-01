"""使用 Gemini 推論「先輩關係 (prerequisite)」並做 DAG 環路檢查。

策略：
1. 候選 pair 產生：對所有概念兩兩配對代價太高，採用以下啟發式減少候選：
   (a) 章節順序差距 ≤ K (預設 3)，且
   (b) 兩概念的 category 相關，或 (c) 在 chunk 中曾共現
2. 將候選 pair 分批送 Gemini 判斷
3. 用 networkx 建 DAG，新增邊前先檢查是否成環
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import google.generativeai as genai
import networkx as nx
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from config import settings
from kg_builder.concept_extractor import Concept
from kg_builder.prompts import PREREQUISITE_SYSTEM, PREREQUISITE_USER
from utils.logger import get_logger

logger = get_logger(__name__)


# ----- Schema -----
class PrerequisiteResult(BaseModel):
    a: str
    b: str
    relation: str  # A_is_prereq_of_B | B_is_prereq_of_A | no_relation
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class PrerequisiteResponse(BaseModel):
    results: List[PrerequisiteResult] = Field(default_factory=list)


class PrerequisiteEdge(BaseModel):
    """寫入 Neo4j 用的邊。"""
    prereq: str
    target: str
    confidence: float
    reason: str
    source: str = "llm_inference"


# ----- Analyzer -----
class PrerequisiteAnalyzer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        chapter_window: int = 3,
        confidence_threshold: float = 0.7,
    ):
        self.api_key = api_key or settings.google_api_key
        if not self.api_key:
            raise RuntimeError("缺少 GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)
        self.model_name = model_name or settings.gemini_model
        self.batch_size = batch_size or settings.prerequisite_batch_size
        self.chapter_window = chapter_window
        self.confidence_threshold = confidence_threshold
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        )

    # ------------------------------------------------------------------
    def analyze(
        self,
        concepts: List[Concept],
        cooccurrence: Optional[Dict[str, Set[str]]] = None,
    ) -> List[PrerequisiteEdge]:
        if len(concepts) < 2:
            return []

        candidate_pairs = self._build_candidate_pairs(concepts, cooccurrence or {})
        logger.info(f"先輩關係候選 pair 數: {len(candidate_pairs)}")

        concept_map = {c.name: c for c in concepts}
        edges: List[PrerequisiteEdge] = []

        # 分批呼叫 LLM
        for batch_start in tqdm(
            range(0, len(candidate_pairs), self.batch_size),
            desc="LLM 推論先輩關係",
        ):
            batch = candidate_pairs[batch_start : batch_start + self.batch_size]
            results = self._call_llm(batch, concept_map)
            for r in results:
                if r.confidence < self.confidence_threshold:
                    continue
                if r.relation == "A_is_prereq_of_B":
                    edges.append(
                        PrerequisiteEdge(
                            prereq=r.a, target=r.b,
                            confidence=r.confidence, reason=r.reason,
                        )
                    )
                elif r.relation == "B_is_prereq_of_A":
                    edges.append(
                        PrerequisiteEdge(
                            prereq=r.b, target=r.a,
                            confidence=r.confidence, reason=r.reason,
                        )
                    )

        logger.info(f"LLM 提出邊數 (信心 ≥ {self.confidence_threshold}): {len(edges)}")
        edges = self._enforce_dag(edges)
        logger.info(f"DAG 檢查後保留邊數: {len(edges)}")
        return edges

    # ------------------------------------------------------------------
    def _build_candidate_pairs(
        self,
        concepts: List[Concept],
        cooccurrence: Dict[str, Set[str]],
    ) -> List[Tuple[str, str]]:
        """以章節距離 + 共現為啟發式條件產生候選 pair。"""
        pairs: Set[Tuple[str, str]] = set()
        for c1, c2 in combinations(concepts, 2):
            n1, n2 = c1.name, c2.name
            if n1 == n2:
                continue
            order_diff = abs((c1.chapter_order or 0) - (c2.chapter_order or 0))
            cooc = (
                n2 in cooccurrence.get(n1, set())
                or n1 in cooccurrence.get(n2, set())
            )
            same_cat = (c1.category and c2.category and c1.category == c2.category)
            if order_diff <= self.chapter_window or cooc or same_cat:
                key = tuple(sorted([n1, n2]))
                pairs.add(key)
        return list(pairs)

    def _call_llm(
        self,
        pairs: List[Tuple[str, str]],
        concept_map: Dict[str, Concept],
    ) -> List[PrerequisiteResult]:
        # 組概念定義區塊 (只包含本批用到的概念)
        used_names = {n for p in pairs for n in p}
        defs_block = "\n".join(
            f"- {n}: {(concept_map[n].definition or '(無定義)')[:160]}"
            for n in sorted(used_names) if n in concept_map
        )
        pairs_block = "\n".join(f"({i+1}) A=「{a}」 vs B=「{b}」" for i, (a, b) in enumerate(pairs))

        prompt = (
            PREREQUISITE_SYSTEM
            + "\n\n"
            + PREREQUISITE_USER.format(
                concept_definitions=defs_block,
                pairs=pairs_block,
            )
        )
        try:
            resp = self.model.generate_content(prompt)
            raw = resp.text or ""
        except Exception as e:
            logger.warning(f"Gemini 呼叫失敗: {e}")
            return []

        data = self._parse_json_safe(raw)
        if data is None:
            logger.warning(f"先輩關係批次 JSON 解析失敗: {raw[:120]}...")
            return []
        try:
            return PrerequisiteResponse.model_validate(data).results
        except ValidationError as e:
            logger.warning(f"PrerequisiteResponse schema 校驗失敗: {e}")
            return []

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

    # ------------------------------------------------------------------
    def _enforce_dag(self, edges: List[PrerequisiteEdge]) -> List[PrerequisiteEdge]:
        """以信心由高到低逐一加入邊；會造成環的就丟掉。"""
        edges_sorted = sorted(edges, key=lambda e: -e.confidence)
        g = nx.DiGraph()
        kept: List[PrerequisiteEdge] = []
        dropped = 0
        for e in edges_sorted:
            g.add_node(e.prereq)
            g.add_node(e.target)
            g.add_edge(e.prereq, e.target)
            if not nx.is_directed_acyclic_graph(g):
                g.remove_edge(e.prereq, e.target)
                dropped += 1
                continue
            kept.append(e)
        if dropped:
            logger.info(f"環路檢查移除 {dropped} 條邊")
        return kept


# ------------------------------------------------------------------
def build_cooccurrence(concepts: List[Concept]) -> Dict[str, Set[str]]:
    """同 chunk 共現矩陣，作為候選 pair 的弱訊號。"""
    chunk_to_names: Dict[str, Set[str]] = defaultdict(set)
    for c in concepts:
        if c.chunk_id:
            chunk_to_names[c.chunk_id].add(c.name)
    cooc: Dict[str, Set[str]] = defaultdict(set)
    for names in chunk_to_names.values():
        for a in names:
            for b in names:
                if a != b:
                    cooc[a].add(b)
    return cooc
