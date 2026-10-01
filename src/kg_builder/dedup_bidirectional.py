"""kg_builder/dedup_bidirectional.py — 共用模組

抽取自 fix_bidirectional_relations.py, 供 pipeline 各階段自動呼叫,
確保未來每章匯入都會自動清理雙向 asymmetric relation bug。

用法:
    from kg_builder.dedup_bidirectional import dedup_bidirectional_relations

    cleaned_rels, dropped_rels, decisions = dedup_bidirectional_relations(
        relations, concepts, mentions
    )
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


# 這些 relation type 是 asymmetric, 只能單向
ASYMMETRIC_RELS = {
    "HAS_SUBTYPE", "PART_OF", "IMPLEMENTS", "USED_FOR",
    "REQUIRES", "CAUSES", "EXAMPLE_OF",
}

# 通用「父類 keyword」— 含這些字通常是廣義概念
GENERIC_PARENT_TOKENS = {
    "loader", "interface", "interpreter", "manager", "system", "service",
    "model", "structure", "kernel", "process", "primitive", "tool",
    "algorithm", "method", "approach", "mechanism", "protocol", "layer",
    "module", "component", "device", "resource",
}


# concept_type generality — 分數越高越 "廣義" (應該當 parent)
# 用來判 HAS_SUBTYPE / PART_OF / EXAMPLE_OF 的方向
_TYPE_GENERALITY_KEYWORDS = [
    # 極廣義 (10)
    (["framework", "architecture", "discipline", "paradigm"], 10),
    # 分類/類別性 (8-9)
    (["_category", "_type", "_family", "_group"], 8),  # 後綴匹配
    (["model", "approach", "strategy", "principle"], 8),
    (["classification", "taxonomy"], 8),
    # 中等 (5-6)
    (["service", "mechanism", "interface"], 6),
    (["system_program", "system_software", "runtime"], 6),
    (["design_concept", "design_principle", "design_decision", "design_goal"], 6),
    (["concept", "abstraction", "layer"], 5),
    # 具體 (3-4)
    (["implementation_strategy", "linking_mechanism", "compilation_strategy"], 3),
    (["metadata", "artifact", "structure"], 3),
    # 預設 = 1 (最具體)
]


def type_generality(concept_type: str) -> int:
    """判 concept_type 的廣義程度. 高分 = 傾向當 parent"""
    if not concept_type:
        return 1
    ct = concept_type.lower().strip()

    # 完全匹配優先
    for keywords, score in _TYPE_GENERALITY_KEYWORDS:
        for kw in keywords:
            if kw.startswith("_"):
                # 後綴匹配 (如 _category)
                if ct.endswith(kw):
                    return score
            elif ct == kw or f"_{kw}" in ct or f"{kw}_" in ct:
                return score
    # 名字含關鍵字 (fallback)
    for kw in ("category", "type", "family", "framework", "architecture"):
        if kw in ct:
            return 7
    return 1


def _section_tuple(sec: str) -> Tuple[int, ...]:
    if not sec:
        return (99,)
    try:
        return tuple(int(p) for p in sec.split("."))
    except ValueError:
        return (99,)


def _quote_signal(quotes: List[str], src_name: str, tgt_name: str, rel: str) -> float:
    """從 quotes 抽 pattern hints. 每種 rel 的語意 pattern 不同"""
    s = src_name.lower().strip()
    t = tgt_name.lower().strip()
    s_re = re.escape(s)
    t_re = re.escape(t)
    score = 0.0
    for q in quotes:
        ql = str(q or "").lower()

        if rel == "HAS_SUBTYPE":
            for pat in [
                rf"\b{t_re}\s+is\s+(?:a|an|one\s+of\s+the)\s+(?:(?:type|kind|form|specific|specialized|example)\s+of\s+)?{s_re}\b",
                rf"\b{t_re},?\s+(?:which|that)\s+is\s+(?:a|an)\s+{s_re}\b",
                rf"\b{s_re}\s+(?:such\s+as|including|like|e\.g\.?|for\s+example)\s+[^.]*?{t_re}\b",
            ]:
                if re.search(pat, ql):
                    score += 3.0
            for pat in [
                rf"\b{s_re}\s+is\s+(?:a|an|one\s+of\s+the)\s+(?:(?:type|kind|form|specific|example)\s+of\s+)?{t_re}\b",
                rf"\b{t_re}\s+(?:such\s+as|including|like|e\.g\.?)\s+[^.]*?{s_re}\b",
            ]:
                if re.search(pat, ql):
                    score -= 3.0

        elif rel == "PART_OF":
            for pat in [
                rf"\b{s_re}\s+is\s+(?:a\s+)?(?:part|component|element|piece|module)\s+of\s+{t_re}\b",
                rf"\b{s_re}\s+is\s+contained\s+in\s+{t_re}\b",
                rf"\b{t_re}\s+(?:contains|consists\s+of|includes|is\s+composed\s+of|comprises)\s+[^.]*?{s_re}\b",
                rf"\b{t_re}\s+provides\s+{s_re}\b",
                rf"\b{s_re}\s+within\s+{t_re}\b",
            ]:
                if re.search(pat, ql):
                    score += 3.0
            for pat in [
                rf"\b{t_re}\s+is\s+(?:a\s+)?(?:part|component|element|piece|module)\s+of\s+{s_re}\b",
                rf"\b{s_re}\s+(?:contains|consists\s+of|includes|comprises)\s+[^.]*?{t_re}\b",
            ]:
                if re.search(pat, ql):
                    score -= 3.0

        elif rel == "EXAMPLE_OF":
            for pat in [
                rf"\b{s_re}\s+is\s+(?:a|an)\s+example\s+of\s+{t_re}\b",
                rf"\b{t_re},?\s+(?:such\s+as|for\s+example|e\.g\.?)\s+[^.]*?{s_re}\b",
            ]:
                if re.search(pat, ql):
                    score += 3.0

        elif rel == "IMPLEMENTS":
            for pat in [
                rf"\b{s_re}\s+implements?\s+{t_re}\b",
                rf"\b{s_re}\s+(?:is\s+an?\s+)?implementation\s+of\s+{t_re}\b",
                rf"\b{s_re}\s+provides?\s+{t_re}\s+interface\b",
            ]:
                if re.search(pat, ql):
                    score += 3.0

        elif rel == "USED_FOR":
            for pat in [
                rf"\b{s_re}\s+(?:is\s+)?used\s+(?:for|to)\s+{t_re}\b",
                rf"\b{s_re}\s+enables?\s+{t_re}\b",
                rf"use\s+{s_re}\s+to\s+{t_re}\b",
            ]:
                if re.search(pat, ql):
                    score += 3.0

        elif rel == "REQUIRES":
            for pat in [
                rf"\b{s_re}\s+requires?\s+{t_re}\b",
                rf"\b{s_re}\s+(?:needs?|depends?\s+on)\s+{t_re}\b",
            ]:
                if re.search(pat, ql):
                    score += 3.0

        elif rel == "CAUSES":
            for pat in [
                rf"\b{s_re}\s+causes?\s+{t_re}\b",
                rf"\b{s_re}\s+leads?\s+to\s+{t_re}\b",
                rf"\b{s_re}\s+results?\s+in\s+{t_re}\b",
            ]:
                if re.search(pat, ql):
                    score += 3.0
    return score


def _score_direction(
    src_id: str, tgt_id: str, rel: str,
    concepts_by_id: Dict[str, Dict[str, Any]],
    mention_counts: Dict[str, int],
    fwd_quotes: List[str], rev_quotes: List[str],
) -> float:
    s = concepts_by_id.get(src_id, {})
    t = concepts_by_id.get(tgt_id, {})
    s_name = str(s.get("name", "")).lower()
    t_name = str(t.get("name", "")).lower()
    s_def = " ".join(s.get("definitions") or []).lower()
    t_def = " ".join(t.get("definitions") or []).lower()
    s_type = str(s.get("concept_type", "")).lower()
    t_type = str(t.get("concept_type", "")).lower()

    score = _quote_signal(fwd_quotes + rev_quotes, s_name, t_name, rel) * 2

    if rel == "HAS_SUBTYPE":
        broader_name, broader_def, broader_id, broader_type = s_name, s_def, src_id, s_type
        narrower_name, narrower_def, narrower_id, narrower_type = t_name, t_def, tgt_id, t_type
    elif rel in ("PART_OF", "EXAMPLE_OF"):
        broader_name, broader_def, broader_id, broader_type = t_name, t_def, tgt_id, t_type
        narrower_name, narrower_def, narrower_id, narrower_type = s_name, s_def, src_id, s_type
    else:
        return score

    # ⭐ Signal 0.5: concept_type generality (最強語意信號, 權重 x3)
    # 若 broader 的 type_generality > narrower 的, 方向正確 → 加大分
    b_gen = type_generality(broader_type)
    n_gen = type_generality(narrower_type)
    if b_gen > n_gen:
        score += (b_gen - n_gen) * 3.0    # e.g. category(8) - process_management(1) = 7 → +21
    elif n_gen > b_gen:
        score -= (n_gen - b_gen) * 3.0    # 反向懲罰

    if broader_name and narrower_name and broader_name != narrower_name:
        if re.search(rf"\b{re.escape(broader_name)}\b", narrower_name):
            score += 3.0
        elif re.search(rf"\b{re.escape(narrower_name)}\b", broader_name):
            score -= 3.0

    if broader_name and narrower_def and re.search(rf"\b{re.escape(broader_name)}\b", narrower_def):
        score += 2.0
    if narrower_name and broader_def and re.search(rf"\b{re.escape(narrower_name)}\b", broader_def):
        score -= 1.0

    bm = mention_counts.get(broader_id, 0)
    nm = mention_counts.get(narrower_id, 0)
    if bm >= nm * 1.5 and bm > 0:
        score += 1.0
    elif nm >= bm * 1.5 and nm > 0:
        score -= 1.0

    if rel == "HAS_SUBTYPE":
        b_sec = _section_tuple(s.get("section", ""))
        n_sec = _section_tuple(t.get("section", ""))
        if b_sec < n_sec:
            score += 0.5
        elif b_sec > n_sec:
            score -= 0.5

    b_tokens = set(re.findall(r"[a-z]+", broader_name))
    n_tokens = set(re.findall(r"[a-z]+", narrower_name))
    b_generic = b_tokens & GENERIC_PARENT_TOKENS
    n_generic = n_tokens & GENERIC_PARENT_TOKENS
    if b_generic and not n_generic:
        score += 1.0
    elif n_generic and not b_generic:
        score -= 1.0
    if len(n_tokens) > len(b_tokens):
        score += 0.5
    elif len(b_tokens) > len(n_tokens):
        score -= 0.5

    return score


def dedup_bidirectional_relations(
    relations: List[Dict[str, Any]],
    concepts: List[Dict[str, Any]],
    mentions: List[Dict[str, Any]],
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """對 relations 做雙向去重.

    Args:
        relations: [{source_kp, target_kp, relation, quote, chunk_id, ...}]
        concepts:  [{id, name, section, definitions, ...}]
        mentions:  [{kp_id, chunk_id, ...}]

    Returns:
        (fixed_relations, dropped_relations, decisions)
    """
    concepts_by_id = {c["id"]: c for c in concepts}

    # 統計 mention 次數 (unique chunk)
    mention_counts: Counter = Counter()
    seen_pair: Set[Tuple[str, str]] = set()
    for m in mentions:
        pair = (m.get("kp_id", ""), m.get("chunk_id", ""))
        if pair in seen_pair:
            continue
        seen_pair.add(pair)
        mention_counts[m.get("kp_id", "")] += 1

    # 找雙向 pair
    by_pair: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in relations:
        rel = r.get("relation")
        if rel not in ASYMMETRIC_RELS:
            continue
        s, t = r.get("source_kp"), r.get("target_kp")
        if not s or not t:
            continue
        key = (min(s, t), max(s, t), rel)
        by_pair[key].append(r)

    bidirectional: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for k, v in by_pair.items():
        if len(v) < 2:
            continue
        dirs = {(r["source_kp"], r["target_kp"]) for r in v}
        if len(dirs) >= 2:
            bidirectional[k] = v

    # 解析
    drops: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    id2name = {c["id"]: c["name"] for c in concepts}

    for key, rels in bidirectional.items():
        dirs_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for r in rels:
            dirs_map[(r["source_kp"], r["target_kp"])].append(r)

        scored = []
        for (s, t), items in dirs_map.items():
            fwd_q = [r.get("quote", "") for r in dirs_map.get((s, t), [])]
            rev_q = [r.get("quote", "") for r in dirs_map.get((t, s), [])]
            sc = _score_direction(s, t, items[0]["relation"],
                                  concepts_by_id, mention_counts, fwd_q, rev_q)
            scored.append((sc, s, t, items))

        scored.sort(key=lambda x: -x[0])
        winner = scored[0]
        for _, s, t, items in scored[1:]:
            drops.extend(items)

        w = winner[3][0]
        decisions.append({
            "keep": f"{id2name.get(w['source_kp'])} --[{w['relation']}]--> "
                    f"{id2name.get(w['target_kp'])}",
            "drop": [
                f"{id2name.get(d['source_kp'])} --[{d['relation']}]--> "
                f"{id2name.get(d['target_kp'])}"
                for _, s, t, items in scored[1:] for d in items
            ],
            "score_diff": round(winner[0] - (scored[1][0] if len(scored) > 1 else 0), 2),
        })
        if verbose:
            print(f"  ✅ KEEP:  {decisions[-1]['keep']}  (Δ={decisions[-1]['score_diff']})")
            for x in decisions[-1]["drop"]:
                print(f"  ❌ DROP:  {x}")

    drop_ids = {id(d) for d in drops}
    fixed = [r for r in relations if id(r) not in drop_ids]

    # 進一步: 移除 3+ 節點循環 (只對 HAS_SUBTYPE / PART_OF)
    fixed, cycle_drops = _remove_multi_node_cycles(fixed, verbose=verbose)
    drops.extend(cycle_drops)

    return fixed, drops, decisions


# ============================================================
# 3+ 節點循環偵測
# ============================================================
def _remove_multi_node_cycles(
    relations: List[Dict[str, Any]],
    hierarchy_rels: Tuple[str, ...] = ("HAS_SUBTYPE", "PART_OF"),
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """對 HAS_SUBTYPE / PART_OF 建 DiGraph, 找 simple_cycles, 移除最弱邊

    "最弱邊" 定義: confidence 最低; 沒 confidence 則 quote 最短
    """
    try:
        import networkx as nx
    except ImportError:
        print("[cycle] networkx 未安裝, 跳過多節點循環偵測")
        return relations, []

    drops: List[Dict[str, Any]] = []
    for rel_type in hierarchy_rels:
        rels_of_type = [r for r in relations if r.get("relation") == rel_type]
        if len(rels_of_type) < 3:
            continue
        # 用 (src, tgt) → list of rels 索引 (可能有多筆同向)
        edge_index: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for r in rels_of_type:
            edge_index[(r["source_kp"], r["target_kp"])].append(r)

        G = nx.DiGraph()
        for (s, t) in edge_index:
            G.add_edge(s, t)

        # 找所有 simple cycle (小的先破)
        removed_this_type = 0
        while True:
            try:
                cycles = list(nx.simple_cycles(G))
            except Exception:
                break
            if not cycles:
                break
            # 挑最短循環
            cycles.sort(key=len)
            cycle = cycles[0]
            # 找該循環中最弱邊 (confidence 最低)
            cycle_edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

            def edge_strength(edge):
                rs = edge_index.get(edge, [])
                if not rs:
                    return 0.0
                confs = [float(r.get("confidence") or 0) for r in rs]
                max_conf = max(confs) if confs else 0.0
                max_quote = max((len(r.get("quote") or "") for r in rs), default=0)
                return max_conf + max_quote / 10000  # tiebreak by quote len

            weakest = min(cycle_edges, key=edge_strength)
            # 刪該邊所有 rels
            for r in edge_index[weakest]:
                drops.append(r)
            edge_index[weakest] = []
            G.remove_edge(*weakest)
            removed_this_type += 1
            if verbose:
                print(f"  ⚠️ {rel_type} cycle {len(cycle)}-node: {' → '.join(cycle)}")
                print(f"     刪除最弱邊: {weakest[0]} → {weakest[1]}")

        if removed_this_type > 0 and verbose:
            print(f"  [cycle] {rel_type} 破 {removed_this_type} 條循環")

    drop_ids = {id(d) for d in drops}
    fixed = [r for r in relations if id(r) not in drop_ids]
    return fixed, drops
