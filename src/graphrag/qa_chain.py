"""GraphRAG QA chain：把 RetrievalResult 餵給 Gemini 回答學生問題。"""
from __future__ import annotations

from typing import Optional

from config import settings
from graphrag.retriever import GraphRetriever, RetrievalResult
from kg_builder.prompts import GRAPHRAG_QA_SYSTEM, GRAPHRAG_QA_USER
from utils.gemini_client import init_gemini, generate_with_rotation
from utils.logger import get_logger

logger = get_logger(__name__)


class GraphRAGChain:
    def __init__(
        self,
        retriever: Optional[GraphRetriever] = None,
        model_name: Optional[str] = None,
    ):
        self.retriever = retriever or GraphRetriever()
        self.model_name = model_name or settings.gemini_model
        # 統一走 utils.gemini_client，QA 階段用較低 temperature 確保答案穩定
        self.model = init_gemini(
            model_name=self.model_name,
            generation_config={"temperature": 0.2},
        )

    def __enter__(self):
        self.retriever.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.retriever.__exit__(exc_type, exc, tb)

    def ask(self, question: str, top_k: int = 5) -> dict:
        result = self.retriever.retrieve(question, top_k=top_k)
        prompt = self._format_prompt(result)
        try:
            # 走多 key 自動輪替
            answer = generate_with_rotation(self.model, prompt) or ""
        except Exception as e:
            logger.error(f"Gemini 回答失敗: {e}")
            answer = "(LLM 呼叫失敗)"
        return {
            "question": question,
            "answer": answer.strip(),
            "seed_concepts": [s["name"] for s in result.seed_concepts],
            # ✨ 新增：把每個 seed 展開拿到的補充資料一起吐出來
            # （子類、父類、教材片段、book_ids 等）
            # MIS 側 enhance_prompt_with_knowledge 讀 result["expanded"] 統計
            # expanded_count，以前這格永遠是 0 就是因為這裡沒吐
            "expanded": result.expanded,
            "prereq_chains": result.prereq_chains,
            "descendant_chains": result.descendant_chains,
        }

    def _format_prompt(self, r: RetrievalResult) -> str:
        concepts_lines = []
        for s in r.seed_concepts:
            concepts_lines.append(
                f"- 【{s['name']}】({s.get('category','其他')}, sim={s.get('score',0):.3f}): "
                f"{(s.get('definition') or '(無定義)')[:200]}"
            )
        concepts_block = "\n".join(concepts_lines) or "(無)"

        chunk_lines = []
        for ex in r.expanded:
            for ck in (ex.get("sample_chunks") or [])[:2]:
                if ck:
                    chunk_lines.append(f"[{ex['name']}] {ck[:400]}")
        chunks_block = "\n---\n".join(chunk_lines) or "(無)"

        prereq_lines = []
        for chain in r.prereq_chains:
            ancestors = chain.get("ancestors") or []
            if not ancestors:
                continue
            names = " ← ".join(a["name"] for a in ancestors[:8])
            prereq_lines.append(f"{chain['concept']} ← {names}")
        prereq_block = "\n".join(prereq_lines) or "(無)"

        desc_lines = []
        for chain in r.descendant_chains:
            descendants = chain.get("descendants") or []
            if not descendants:
                continue
            names = " → ".join(d["name"] for d in descendants[:8])
            desc_lines.append(f"{chain['concept']} → {names}")
        descendant_block = "\n".join(desc_lines) or "(無)"

        subtype_lines = []
        for ex in r.expanded:
            subs = ex.get("subtypes") or []
            parents = ex.get("parents") or []
            if subs:
                subtype_lines.append(
                    f"{ex['name']} 的細分: " + ", ".join(subs[:8])
                )
            if parents:
                subtype_lines.append(
                    f"{ex['name']} 的父概念: " + ", ".join(parents[:4])
                )
        subtype_block = "\n".join(subtype_lines) or "(無)"

        # 把 SYSTEM + USER 拼成完整 prompt 給 generate_with_rotation
        # （沿用 kg_builder.prompts 定義好的兩段常數；GRAPHRAG_QA_USER 需要 6 個欄位）
        user_prompt = GRAPHRAG_QA_USER.format(
            question=r.query,
            concepts_block=concepts_block,
            chunks_block=chunks_block,
            prereq_block=prereq_block,
            descendant_block=descendant_block,
            subtype_block=subtype_block,
        )
        return f"{GRAPHRAG_QA_SYSTEM}\n\n{user_prompt}"
