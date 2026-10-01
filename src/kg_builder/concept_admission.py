"""Concept Admission — 決定 Candidate 是否升級成正式 Concept

7 條規則,對於一個「Chunk 中出現但 taxonomy 沒有」的候選詞:

    1. formal_definition   : 教材有正式定義
    2. main_content        : 為章節核心內容
    3. exam_material       : 能形成考題
    4. graph_relation      : 可建立 >= 2 條 relation
    5. not_variable_role   : 非變數/範例角色/實作細節
    6. not_figure_table    : 非 Figure/Table/Chart 名
    7. cross_chunk_reuse   : 跨 chunk 出現 >= 3 次 (或章節標題)

任一失敗 → 不升級,可選擇存 Chunk.keywords 或直接排除。

用法:
    from kg_builder.concept_admission import ConceptAdmission

    admission = ConceptAdmission(taxonomy=taxonomy)
    verdict = admission.evaluate(
        candidate_name="Memory Fence",
        occurrences=[{"chunk_id": "...", "text": "...", "context_type": "definition"}],
        relations_hint=[("Memory Fence", "USED_FOR", "Memory Consistency")],
    )
    # verdict.pass_all, verdict.scores, verdict.reject_reasons
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# 常見變數命名 pattern (從論文 Part 03 觀察)
VARIABLE_NAME_PATTERNS = [
    re.compile(r"^[a-z][A-Za-z]*[0-9]$"),           # thread1, register2
    re.compile(r"^[A-Z][a-z]+[Vv]ariable$"),        # TurnVariable, LockVariable
    re.compile(r"^[A-Z][a-z]+Array$"),              # FlagArray, WaitingArray
    re.compile(r"^Flag_?[a-zA-Z0-9]+$"),            # Flag_i, FlagI
    re.compile(r"^Struct(ure)?Of.+"),               # StructureOfProcessPi...
]

# 圖名/表名 pattern
FIGURE_TABLE_PATTERNS = [
    re.compile(r"^Figure\d"),
    re.compile(r"^Fig[_\.]?\d"),
    re.compile(r"^Table\d"),
    re.compile(r"^Chart\d"),
    re.compile(r".*(Diagram|Illustration|Schematic)$"),
]

# 「範例角色」pattern — Process/Thread + 描述性字
ROLE_PATTERNS = [
    re.compile(r"^(Producer|Consumer)(Process|Thread|Application)$"),
    re.compile(r"^(Reader|Writer)(Process|Thread)$"),
    re.compile(r"^(Client|Server)Process$"),
    re.compile(r"^Example[A-Z]"),
]

# 「實作細節」pattern — 過長的組合字
IMPLEMENTATION_PATTERNS = [
    re.compile(r".{40,}"),  # 太長 (40+ char) 通常是描述
    re.compile(r".*(Data|Info)Structure$"),   # KernelDataStructure
    re.compile(r".*Manipulation(Function)?$"),  # AtomicVariableManipulationFunction
    re.compile(r".*ForCriticalSection$"),
    re.compile(r".*InPetersonSolution$"),
]


@dataclass
class AdmissionVerdict:
    """單一候選的判定結果"""

    name: str
    scores: Dict[str, int] = field(default_factory=dict)  # 每條規則 0/1
    reject_reasons: List[str] = field(default_factory=list)
    pass_all: bool = False
    pass_count: int = 0
    verdict: str = "reject"  # accept / demote_to_mention / reject


def _rule_formal_definition(occurrences: List[Dict[str, Any]]) -> Tuple[int, str]:
    """規則 1: 是否有正式定義

    Signal: context_type 中含 'definition' 或文字含 'is defined as', 'refers to'
    """
    def_signals = [
        "is defined as", "refers to", "means", "denotes",
        "is a ", "is an ", "are defined", "is called",
    ]
    for occ in occurrences:
        ctx_type = str(occ.get("context_type") or "").lower()
        text = str(occ.get("text") or "").lower()
        if "definition" in ctx_type:
            return 1, ""
        if any(s in text for s in def_signals):
            return 1, ""
    return 0, "無正式定義 (未見 'is defined as' 或 definition context)"


def _rule_main_content(
    name: str, occurrences: List[Dict[str, Any]]
) -> Tuple[int, str]:
    """規則 2: 是否是章節核心

    Signal: 出現在 section_title / chapter_title 或多次出現在同章的多個 chunk
    """
    has_title = any(
        str(o.get("context_type") or "").lower()
        in ("section_title", "chapter_title", "heading")
        for o in occurrences
    )
    if has_title:
        return 1, ""
    if len(occurrences) >= 5:
        return 1, ""
    return 0, f"未在標題出現且 occurrence 不足 (只有 {len(occurrences)})"


def _rule_exam_material(name: str, occurrences: List[Dict[str, Any]]) -> Tuple[int, str]:
    """規則 3: 能形成考題

    啟發式: 名字是明確可問「What is X?」的名詞短語,不是變數/角色
    """
    if any(p.match(name) for p in VARIABLE_NAME_PATTERNS):
        return 0, "似變數名"
    if any(p.match(name) for p in ROLE_PATTERNS):
        return 0, "似範例角色"
    return 1, ""


def _rule_graph_relation(
    relations_hint: List[Tuple[str, str, str]],
) -> Tuple[int, str]:
    """規則 4: 可建立 >= 2 條 relation

    relations_hint: [(source, rel_type, target), ...]
    """
    if len(relations_hint) >= 2:
        return 1, ""
    return 0, f"relation 不足 (只有 {len(relations_hint)})"


def _rule_not_variable_role(name: str) -> Tuple[int, str]:
    """規則 5: 非變數/角色/實作細節"""
    for p in VARIABLE_NAME_PATTERNS:
        if p.match(name):
            return 0, f"符合變數命名 pattern: {p.pattern}"
    for p in ROLE_PATTERNS:
        if p.match(name):
            return 0, f"符合範例角色 pattern: {p.pattern}"
    for p in IMPLEMENTATION_PATTERNS:
        if p.match(name):
            return 0, f"符合實作細節 pattern: {p.pattern}"
    return 1, ""


def _rule_not_figure_table(name: str) -> Tuple[int, str]:
    """規則 6: 非 Figure/Table 名"""
    for p in FIGURE_TABLE_PATTERNS:
        if p.match(name):
            return 0, f"符合 Figure/Table pattern: {p.pattern}"
    return 1, ""


def _rule_cross_chunk_reuse(
    occurrences: List[Dict[str, Any]], min_chunks: int = 3
) -> Tuple[int, str]:
    """規則 7: 跨 chunk 出現 >= 3 次"""
    unique_chunks = {o.get("chunk_id") for o in occurrences if o.get("chunk_id")}
    if len(unique_chunks) >= min_chunks:
        return 1, ""
    return 0, f"跨 chunk 不足 (只有 {len(unique_chunks)}, 需 >= {min_chunks})"


class ConceptAdmission:
    """封裝 admission 判定"""

    def __init__(
        self,
        taxonomy: Optional[Dict[str, Any]] = None,
        min_pass: int = 5,
        demote_to_mention_at: int = 3,
    ):
        """
        min_pass                : >= 幾條規則通過 → accept (預設 5/7)
        demote_to_mention_at    : >= 幾條規則通過 → demote 為 Mention (存 keywords)
        """
        self.taxonomy = taxonomy or {}
        self.min_pass = min_pass
        self.demote_to_mention_at = demote_to_mention_at
        # 建 taxonomy name -> concept 索引 (含 alias)
        self._known_names: Set[str] = set()
        for st in (self.taxonomy.get("subtrees") or []):
            for c in (st.get("concepts") or []):
                self._known_names.add(c.get("name", "").lower())
                for a in (c.get("aliases") or []):
                    self._known_names.add(str(a).lower())

    def is_known(self, name: str) -> bool:
        return name.lower() in self._known_names

    def evaluate(
        self,
        candidate_name: str,
        occurrences: List[Dict[str, Any]],
        relations_hint: Optional[List[Tuple[str, str, str]]] = None,
    ) -> AdmissionVerdict:
        """對一個 candidate 跑 7 條規則"""
        rels = relations_hint or []
        rules = [
            ("formal_definition", *_rule_formal_definition(occurrences)),
            ("main_content", *_rule_main_content(candidate_name, occurrences)),
            ("exam_material", *_rule_exam_material(candidate_name, occurrences)),
            ("graph_relation", *_rule_graph_relation(rels)),
            ("not_variable_role", *_rule_not_variable_role(candidate_name)),
            ("not_figure_table", *_rule_not_figure_table(candidate_name)),
            ("cross_chunk_reuse", *_rule_cross_chunk_reuse(occurrences)),
        ]

        verdict = AdmissionVerdict(name=candidate_name)
        for rule_name, score, reason in rules:
            verdict.scores[rule_name] = score
            if score == 0 and reason:
                verdict.reject_reasons.append(f"{rule_name}: {reason}")

        verdict.pass_count = sum(verdict.scores.values())
        verdict.pass_all = verdict.pass_count == 7

        # 判定
        # Hard reject: 命名 pattern 觸發 (rule 5/6) 直接踢
        hard_kill = (
            verdict.scores.get("not_variable_role", 1) == 0
            or verdict.scores.get("not_figure_table", 1) == 0
        )
        if hard_kill:
            verdict.verdict = "reject"
        elif verdict.pass_count >= self.min_pass:
            verdict.verdict = "accept"
        elif verdict.pass_count >= self.demote_to_mention_at:
            verdict.verdict = "demote_to_mention"
        else:
            verdict.verdict = "reject"
        return verdict


# ============================================================
# CLI: 對現有 concepts.json 跑一次 admission 檢查 (用來 QA 舊資料)
# ============================================================
if __name__ == "__main__":
    import argparse
    import json
    from collections import Counter
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--concepts", "-c", required=True, help="concepts.json (from KCEP)")
    p.add_argument("--chunks", "-k", required=True, help="chunks.jsonl")
    p.add_argument("--relations", "-r", required=True, help="typed_relations.json")
    p.add_argument("--taxonomy", "-t", default=None, help="taxonomy.yaml (optional)")
    p.add_argument("--output", "-o", default="admission_report.json")
    args = p.parse_args()

    concepts = json.loads(Path(args.concepts).read_text(encoding="utf-8"))
    if isinstance(concepts, dict):
        concepts = concepts.get("concepts") or []
    relations = json.loads(Path(args.relations).read_text(encoding="utf-8"))
    if isinstance(relations, dict):
        relations = relations.get("relations") or []

    # 建 concept name -> occurrences
    name_to_occ: Dict[str, List[Dict[str, Any]]] = {}
    for c in concepts:
        name = c.get("name") or c.get("canonical") or ""
        occs = []
        for cid in (c.get("source_chunks") or []):
            occs.append({"chunk_id": cid, "text": c.get("definition", ""), "context_type": ""})
        name_to_occ[name] = occs

    # 建 relations index
    name_to_rels: Dict[str, List[Tuple[str, str, str]]] = {}
    for r in relations:
        s, t, rel = str(r.get("source") or ""), str(r.get("target") or ""), str(r.get("relation") or "RELATED")
        name_to_rels.setdefault(s, []).append((s, rel, t))
        name_to_rels.setdefault(t, []).append((s, rel, t))

    # 載 taxonomy (可選)
    taxonomy = None
    if args.taxonomy:
        import yaml
        taxonomy = yaml.safe_load(Path(args.taxonomy).read_text(encoding="utf-8"))

    admission = ConceptAdmission(taxonomy=taxonomy)
    results = []
    verdict_counter: Counter = Counter()
    for c in concepts:
        name = c.get("name") or c.get("canonical") or ""
        if not name:
            continue
        v = admission.evaluate(
            candidate_name=name,
            occurrences=name_to_occ.get(name, []),
            relations_hint=name_to_rels.get(name, []),
        )
        verdict_counter[v.verdict] += 1
        results.append({
            "name": name,
            "verdict": v.verdict,
            "pass_count": v.pass_count,
            "scores": v.scores,
            "reject_reasons": v.reject_reasons,
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "total": len(results),
        "verdict_summary": dict(verdict_counter),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"總計: {len(results)}")
    for k, v in verdict_counter.most_common():
        print(f"  {k}: {v}")
    print(f"報告寫出: {out}")
