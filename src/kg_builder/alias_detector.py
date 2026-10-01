"""kg_builder/alias_detector.py — 從 chunk 文字自動偵測別名 pair

偵測 pattern:
    "X, also known as Y"
    "X, also called Y"
    "X (also known as Y)"
    "X (or Y)"                    (e.g., "command-line interface, or command interpreter")
    "X, or Y"
    "X (Y)"                       (簡短括弧, 需 Y 也在 KP 清單才算)
    "X or Y"                      (需 X 和 Y 都在 KP 清單)

回傳:
    aliases = [(kp_id_a, kp_id_b, evidence_chunk_id, evidence_quote), ...]

用法:
    from kg_builder.alias_detector import detect_aliases

    aliases = detect_aliases(chunks, concepts)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


# 別名 pattern (Y 是 X 的別名)
ALIAS_PATTERNS = [
    # X, also known as Y
    r"([A-Z][A-Za-z\-\s]{2,60}?),?\s+also\s+known\s+as\s+(?:the\s+)?([A-Z][A-Za-z\-\s]{2,60})",
    # X, also called Y
    r"([A-Z][A-Za-z\-\s]{2,60}?),?\s+also\s+called\s+(?:the\s+)?([A-Z][A-Za-z\-\s]{2,60})",
    # X (aka Y) / X (also known as Y)
    r"([A-Z][A-Za-z\-\s]{2,60}?)\s*\((?:aka|a\.k\.a\.|also\s+known\s+as|also\s+called)\s+([A-Z][A-Za-z\-\s]{2,60})\)",
    # X, or Y
    r"([A-Z][A-Za-z\-\s]{2,60}?),\s+or\s+([A-Z][A-Za-z\-\s]{2,60})",
    # X (Y) — 只在 Y 短且 X/Y 都在 KP 清單時
    r"([A-Z][A-Za-z\-]{2,40}(?:\s+[A-Z][A-Za-z\-]{2,40}){0,3})\s+\(([A-Z][A-Za-z]{1,20})\)",
]
_COMPILED = [re.compile(p) for p in ALIAS_PATTERNS]


def _normalize(s: str) -> str:
    """去空白 + 小寫, 用於比對"""
    return re.sub(r"\s+", " ", s).strip().lower()


def _looks_like_acronym(value: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]", "", value)
    return bool(compact) and len(compact) <= 12 and compact.upper() == compact


def detect_aliases(
    chunks: List[Dict[str, Any]],
    concepts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """從 chunk 文字找 KP-KP 的別名 pair.

    只回傳兩個 name 都在 KP 清單裡的 pair (避免抓到非 KP 詞)。
    """
    # KP 名字 → id (加正規化)
    name_to_id: Dict[str, str] = {}
    id_to_concept: Dict[str, Dict[str, Any]] = {}
    for c in concepts:
        name_to_id[_normalize(c["name"])] = c["id"]
        id_to_concept[c["id"]] = c

    aliases: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for chunk in chunks:
        text = chunk.get("text", "")
        if not text:
            continue
        cid = chunk.get("chunk_id", "")
        for compiled in _COMPILED:
            for m in compiled.finditer(text):
                a_raw, b_raw = m.group(1).strip(), m.group(2).strip()
                a_norm = _normalize(a_raw)
                b_norm = _normalize(b_raw)
                if a_norm == b_norm:
                    continue
                a_id = name_to_id.get(a_norm)
                b_id = name_to_id.get(b_norm)
                if not a_id or not b_id:
                    continue
                a_concept = id_to_concept[a_id]
                b_concept = id_to_concept[b_id]
                a_name = str(a_concept.get("name") or a_raw)
                b_name = str(b_concept.get("name") or b_raw)
                a_core = a_concept.get("importance") == "Core"
                b_core = b_concept.get("importance") == "Core"
                # Prefer the reviewed Core entry; otherwise prefer the expanded
                # non-acronym label. "Interrupt Service Routine (ISR)" must not
                # canonicalize to the shorter acronym.
                choose_a = (
                    (a_core and not b_core)
                    or (
                        a_core == b_core
                        and _looks_like_acronym(b_name)
                        and not _looks_like_acronym(a_name)
                    )
                    or (
                        a_core == b_core
                        and _looks_like_acronym(a_name) == _looks_like_acronym(b_name)
                        and len(a_norm) >= len(b_norm)
                    )
                )
                if choose_a:
                    canonical_id, alias_id = a_id, b_id
                    canonical_name, alias_name = a_name, b_name
                else:
                    canonical_id, alias_id = b_id, a_id
                    canonical_name, alias_name = b_name, a_name
                key = tuple(sorted([canonical_id, alias_id]))
                if key not in aliases:
                    # 抓 quote (前後各 40 字)
                    start = max(0, m.start() - 40)
                    end = min(len(text), m.end() + 40)
                    quote = text[start:end].replace("\n", " ").strip()
                    aliases[key] = {
                        "canonical_kp": canonical_id,
                        "canonical_name": canonical_name,
                        "alias_kp": alias_id,
                        "alias_name": alias_name,
                        "chunk_id": cid,
                        "quote": quote[:300],
                    }
    return list(aliases.values())


def apply_aliases_to_relations(
    relations: List[Dict[str, Any]],
    aliases: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """把 relations 中的 alias_kp 全部替換成 canonical_kp

    順帶去掉「A → B」若 A 和 B 是別名的 relation (因為它們是同一概念)
    """
    # 建映射 alias_id → canonical_id
    alias_to_canon: Dict[str, str] = {}
    for a in aliases:
        alias_to_canon[a["alias_kp"]] = a["canonical_kp"]

    if not alias_to_canon:
        return relations

    fixed = []
    for r in relations:
        s = alias_to_canon.get(r.get("source_kp", ""), r.get("source_kp"))
        t = alias_to_canon.get(r.get("target_kp", ""), r.get("target_kp"))
        # 若 alias 替換後變自循環, 丟
        if s == t:
            continue
        new_r = dict(r)
        new_r["source_kp"] = s
        new_r["target_kp"] = t
        fixed.append(new_r)
    return fixed


def apply_aliases_to_mentions(
    mentions: List[Dict[str, Any]],
    aliases: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """mentions 的 kp_id 也替換成 canonical"""
    alias_to_canon = {a["alias_kp"]: a["canonical_kp"] for a in aliases}
    if not alias_to_canon:
        return mentions
    return [{
        **m,
        "kp_id": alias_to_canon.get(m.get("kp_id", ""), m.get("kp_id")),
    } for m in mentions]


def drop_alias_kps(
    concepts: List[Dict[str, Any]],
    aliases: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """從 concepts 移除 alias KP (併入 canonical). 回傳 (剩下的 concepts, canonical→[aliases] 對照)"""
    alias_ids = {a["alias_kp"] for a in aliases}
    canon_aliases: Dict[str, List[str]] = {}
    for a in aliases:
        canon_aliases.setdefault(a["canonical_kp"], []).append(a["alias_name"])

    concept_by_id = {str(c.get("id")): dict(c) for c in concepts}
    for alias in aliases:
        canonical = concept_by_id.get(str(alias.get("canonical_kp")))
        alias_concept = concept_by_id.get(str(alias.get("alias_kp")))
        if canonical is None or alias_concept is None:
            continue
        definitions = [
            *(canonical.get("definitions") or []),
            *(alias_concept.get("definitions") or []),
        ]
        canonical["definitions"] = list(dict.fromkeys(definitions))
        canon_aliases.setdefault(str(alias.get("canonical_kp")), []).extend(
            alias_concept.get("aliases") or []
        )
        evidence_by_key = {}
        for item in [
            *(canonical.get("evidence") or []),
            *(alias_concept.get("evidence") or []),
        ]:
            key = (
                item.get("chunk_id"),
                item.get("quote"),
                item.get("type"),
            )
            evidence_by_key[key] = item
        if evidence_by_key:
            canonical["evidence"] = list(evidence_by_key.values())
            canonical["chunk_count"] = len({
                item.get("chunk_id") for item in canonical["evidence"]
                if item.get("chunk_id")
            })
        concept_by_id[str(alias.get("canonical_kp"))] = canonical

    remaining = []
    for original in concepts:
        c = concept_by_id[str(original.get("id"))]
        if c["id"] in alias_ids:
            continue
        # 加入 aliases 屬性
        c["aliases"] = list(dict.fromkeys(
            [*(c.get("aliases") or []), *canon_aliases.get(c["id"], [])]
        ))
        remaining.append(c)
    return remaining, canon_aliases
