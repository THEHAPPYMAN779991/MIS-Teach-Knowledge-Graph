# -*- coding: utf-8 -*-
"""Relation typer for GraphRAG JSON outputs.

v8 目的：
- 延續 v7：保留 concepts / subtype_edges / prereq_edges，額外產生 typed_relations。
- 修正 v7 常見誤判：Buffer -> BoundedBuffer 不再判成 PART_OF；Process -> AddressSpace 不再判成 IS_A。
- 新增 API_MEMBER_OF / HAS_COMPONENT。
- 低信心 RELATED_TO 不混入正式 typed_relations，改放 review_only_relations.json。
- typed_relations 依 (source, target, relation) 去重，不因 origin 不同重複。
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

Relation = Dict[str, Any]
Record = Dict[str, Any]


def _norm_text(s: Any) -> str:
    s = str(s or "").strip()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()


def _tokens(name: str) -> List[str]:
    n = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name or ""))
    n = re.sub(r"[^A-Za-z0-9]+", " ", n)
    return [t.lower() for t in n.split() if t]


def _contains_any(text: str, patterns: Iterable[str]) -> bool:
    t = _norm_text(text)
    return any(p.lower() in t for p in patterns)


def _concept_map(concepts: List[Record]) -> Dict[str, Record]:
    return {str(c.get("name")): c for c in concepts if c.get("name")}


def _definition(concepts_by_name: Dict[str, Record], name: str) -> str:
    c = concepts_by_name.get(name) or {}
    return str(c.get("definition") or "")


def _aliases(concepts_by_name: Dict[str, Record], name: str) -> List[str]:
    c = concepts_by_name.get(name) or {}
    return [str(x) for x in (c.get("aliases") or [])]


# -----------------------------
# classification data
# -----------------------------

# parent -> child 表示 parent 底下有 child；relation 名稱是語意類型，不是 Cypher 方向的英文句子。
KNOWN_CATEGORY_PARENTS = {
    "Process", "OperatingSystem", "Buffer", "Pipe", "Message", "Mailbox", "Socket", "Port",
    "Thread", "Command", "Queue", "InterprocessCommunication", "Communication", "MessagePassing",
    "SharedMemory", "ProcessClassification", "InitialProcess", "ProcessTermination", "ProcessCreation",
    "ProcessExecutionModel", "ParentChildExecutionFlow", "ProcessResource", "MessageSize", "Buffering",
    "InterprocessCommunicationMethod", "DirectCommunication", "IndirectCommunication", "ClientServerArchitecture",
    "RemoteProcedureCall", "MachMessageSendingOption", "MachMessageSendingOptions", "PortRight",
    "MachPortRight", "CommunicationPort", "PrivateCommunicationPort", "FileDescriptor",
}

# 這些 parent 底下的 child 常常是組成/欄位，不是子類。
STRUCTURE_PARENT_KEYWORDS = (
    "Structure", "Struct", "ControlBlock", "Header", "Record", "Object", "Descriptor",
    "Table", "Layout", "Block", "Packet", "Frame",
)

MECHANISM_PARENT_KEYWORDS = (
    "System", "Facility", "Mechanism", "Architecture", "Framework", "Hierarchy", "Classification",
    "Communication", "MessagePassing", "ProcedureCall", "Channel", "Port", "Pipe", "ContextSwitch",
    "Swapping", "Scheduling", "ProcessState", "MemoryLayout",
)

FIELD_LIKE_CHILDREN = {
    "ProcessHandle", "ThreadHandle", "ProgramCounter", "CentralProcessingUnitRegister",
    "CentralProcessingUnitSchedulingInformation", "MemoryManagementInformation", "AccountingInformation",
    "InputOutputStatusInformation", "OpenFileList", "MemoryLimit", "ProcessNumber", "ProcessState",
    "TaskStructure", "ReadEnd", "WriteEnd", "ReadHandle", "WriteHandle", "ReadEndFileDescriptor",
    "WriteEndFileDescriptor", "ReadEndOfPipe", "WriteEndOfPipe", "InPointer", "OutPointer",
    "MapSharedFlag", "ExitStatus", "PortNumber", "InternetProtocolAddress", "ByteOrder", "DataType",
    "MostRecentCommandExecution", "HistoryBuffer", "OutputRedirectionOperator", "InputRedirectionOperator",
}

COMPONENT_LIKE_CHILD_KEYWORDS = (
    "Space", "Region", "Segment", "Section", "Area", "Handle", "Descriptor", "Pointer", "Counter",
    "Register", "Information", "Status", "Limit", "Number", "List", "Table", "Entry", "Header",
    "Port", "Stub", "Daemon", "Buffer", "Object", "Queue", "Stream", "Reader", "Writer",
)

CALLABLE_CHILD_KEYWORDS = (
    "Function", "SystemCall", "Primitive", "Operation", "Command", "Method", "Call",
)

API_PARENT_KEYWORDS = (
    "API", "ApplicationProgrammingInterface", "Interface", "SystemCall", "Function", "Primitive", "Operation", "Command",
)

STATE_TRANSITIONS = {
    ("NewState", "ReadyState"),
    ("ReadyState", "RunningState"),
    ("RunningState", "WaitingState"),
    ("WaitingState", "ReadyState"),
    ("RunningState", "TerminatedState"),
    ("RunningState", "ReadyState"),
}

# 明確修正常見誤判：parent -> child
FORCED_STRUCTURE_RELATIONS: Dict[Tuple[str, str], Tuple[str, str, float]] = {
    ("Process", "AddressSpace"): ("HAS_COMPONENT", "a process has an address space; address space is not a subtype of process", 0.94),
    ("SharedMemory", "SharedMemoryRegion"): ("HAS_COMPONENT", "shared memory has/uses a shared memory region", 0.90),
    ("SharedMemory", "SharedMemorySegment"): ("HAS_COMPONENT", "shared memory has/uses a shared memory segment", 0.90),
    ("ProcessInformationStructure", "ProcessHandle"): ("HAS_FIELD", "PROCESS_INFORMATION contains hProcess", 0.95),
    ("ProcessInformationStructure", "ThreadHandle"): ("HAS_FIELD", "PROCESS_INFORMATION contains hThread", 0.95),
    ("SocketAddress", "InternetProtocolAddress"): ("HAS_COMPONENT", "socket address contains IP address", 0.92),
    ("SocketAddress", "PortNumber"): ("HAS_COMPONENT", "socket address contains port number", 0.92),
    ("Program", "MainFunction"): ("HAS_COMPONENT", "program contains main function entry point", 0.82),
}

FORCED_IS_A_RELATIONS: Dict[Tuple[str, str], Tuple[str, str, float]] = {
    ("Buffer", "BoundedBuffer"): ("IS_A", "bounded buffer is a kind of buffer", 0.94),
    ("Buffer", "UnboundedBuffer"): ("IS_A", "unbounded buffer is a kind of buffer", 0.94),
    ("Pipe", "OrdinaryPipe"): ("IS_A", "ordinary pipe is a kind of pipe", 0.94),
    ("Pipe", "NamedPipe"): ("IS_A", "named pipe is a kind of pipe", 0.94),
    ("Pipe", "AnonymousPipe"): ("IS_A", "anonymous pipe is a kind of pipe", 0.92),
    ("Message", "SimpleMessage"): ("IS_A", "simple message is a kind of message", 0.90),
    ("Message", "ComplexMessage"): ("IS_A", "complex message is a kind of message", 0.90),
    ("Socket", "ClientSocket"): ("IS_A", "client socket is a kind of socket", 0.90),
    ("Socket", "ServerSocket"): ("IS_A", "server socket is a kind of socket", 0.90),
}


def _looks_callable(name: str) -> bool:
    return any(k in name for k in CALLABLE_CHILD_KEYWORDS) or name.lower().endswith(("call", "function", "method"))


def _looks_api_parent(name: str) -> bool:
    return any(k in name for k in API_PARENT_KEYWORDS)


def _looks_structure_parent(name: str) -> bool:
    return any(k in name for k in STRUCTURE_PARENT_KEYWORDS)


def _looks_mechanism_parent(name: str) -> bool:
    return any(k in name for k in MECHANISM_PARENT_KEYWORDS)


def _looks_component_child(name: str) -> bool:
    return name in FIELD_LIKE_CHILDREN or any(k in name for k in COMPONENT_LIKE_CHILD_KEYWORDS)


# -----------------------------
# relation classification
# -----------------------------

def classify_structure_edge(parent: str, child: str, concepts_by_name: Dict[str, Record]) -> Tuple[str, str, float, bool]:
    """Classify a legacy subtype edge into a typed relation.

    Returns: relation, note, confidence, review_only
    """
    if (parent, child) in STATE_TRANSITIONS:
        return "STATE_TRANSITION", "matches a known process-state transition", 0.96, False

    if (parent, child) in FORCED_STRUCTURE_RELATIONS:
        rel, note, conf = FORCED_STRUCTURE_RELATIONS[(parent, child)]
        return rel, note, conf, False
    if (parent, child) in FORCED_IS_A_RELATIONS:
        rel, note, conf = FORCED_IS_A_RELATIONS[(parent, child)]
        return rel, note, conf, False

    p_tokens = set(_tokens(parent))
    c_tokens = set(_tokens(child))
    p_def = _definition(concepts_by_name, parent)
    c_def = _definition(concepts_by_name, child)
    combined_child_text = " ".join([child, c_def] + _aliases(concepts_by_name, child))

    # API / Interface membership. Includes functions and data structures/constants declared by an API.
    if _looks_api_parent(parent):
        if _looks_callable(child):
            return "API_MEMBER_OF", "parent API/interface/system-call family contains this callable member", 0.90, False
        if _looks_structure_parent(child) or _looks_component_child(child):
            return "API_MEMBER_OF", "parent API/interface contains this data structure/constant/member", 0.86, False

    # Concrete structures/control blocks contain fields/members.
    if _looks_structure_parent(parent):
        if _looks_component_child(child) or child in FIELD_LIKE_CHILDREN:
            return "HAS_FIELD", "parent data structure/control block contains child as field/member", 0.92, False
        return "HAS_COMPONENT", "parent structure contains child as component/detail", 0.82, False

    # Type/specialization first, before component rules, to avoid Buffer -> BoundedBuffer becoming HAS_COMPONENT.
    if parent in KNOWN_CATEGORY_PARENTS:
        if _contains_any(combined_child_text, ["kind of", "type of", "specific type", "specific instance", "是一種", "具體類型", "特定類型"]):
            return "IS_A", "child definition/aliases describe a kind/type/instance of parent", 0.88, False
        if p_tokens and p_tokens.issubset(c_tokens) and parent != child:
            return "IS_A", "child name lexically specializes parent category", 0.84, False
        if child.endswith(parent) or child.startswith(parent):
            return "IS_A", "child name specializes parent category", 0.82, False
        # common OS categories whose children are normally types/roles
        if parent in {"Process", "OperatingSystem", "Pipe", "Socket", "Mailbox", "Message", "Buffer", "InterprocessCommunication", "ProcessClassification", "ProcessResource"}:
            # Do not classify obvious components as IS_A.
            if _looks_component_child(child) and parent not in {"Buffer", "Pipe", "Socket", "Mailbox", "Message"}:
                return "HAS_COMPONENT", "child is a component/property of parent rather than a subtype", 0.80, False
            return "IS_A", "parent is a known category and child is a specific kind/role/model", 0.82, False

    # Mechanism/system/layout contains components, options, operations.
    if _looks_mechanism_parent(parent):
        if _looks_callable(child):
            return "API_MEMBER_OF", "mechanism/facility exposes this operation/primitive", 0.82, False
        return "HAS_COMPONENT", "mechanism/system/layout contains child as component/detail/option", 0.80, False

    # Lexical specialization fallback.
    if p_tokens and p_tokens.issubset(c_tokens) and parent != child:
        return "IS_A", "child name lexically specializes parent name", 0.76, False

    if _contains_any(c_def, [f"is a {parent}", f"is an {parent}", f"是一種 {parent}", "是一種", "specific instance", "特定類型", "具體類型"]):
        return "IS_A", "child definition describes itself as a type/instance of parent", 0.78, False

    return "RELATED_TO", "legacy edge could not be safely classified; kept for manual review only", 0.50, True


def classify_prereq_edge(edge: Record, concepts_by_name: Dict[str, Record]) -> Tuple[str, str, float, bool]:
    prereq = str(edge.get("prereq") or "")
    target = str(edge.get("target") or "")
    reason = str(edge.get("reason") or "")

    if (prereq, target) in STATE_TRANSITIONS:
        return "STATE_TRANSITION", "this pair is a state transition, not a learning prerequisite", 0.96, False

    # Use stronger semantic classes when the reason is clearly not learning-order.
    if _contains_any(reason, ["creates", "created by", "建立", "創建", "產生", "結果", "導致", "return value", "回傳值", "responsible for"]):
        return "CREATES_OR_RESULTS_IN", "reason describes creation/result/return value rather than pure learning dependency", 0.76, False
    if _contains_any(reason, ["parameter", "參數", "需要", "使用", "uses", "used by", "呼叫", "called", "invoke", "passed", "waits for", "等待"]):
        return "USES", "reason describes use/parameter/call/wait relationship; not a pure prerequisite", 0.72, False
    if _contains_any(reason, ["member", "field", "contains", "包含", "成員", "欄位"]):
        return "HAS_FIELD", "reason describes field/member containment", 0.78, False
    if _contains_any(reason, ["組成", "component", "part of", "核心組件", "一部分"]):
        return "HAS_COMPONENT", "reason describes component containment rather than prerequisite", 0.76, False
    if _contains_any(reason, ["is a", "是一種", "具體類型", "specific instance", "specific type", "特定類型"]):
        return "IS_A", "reason describes type relation rather than learning dependency", 0.76, False

    return "PREREQ_OF", "kept as learning prerequisite", float(edge.get("confidence") or 0.75), False


def build_typed_relations(concepts: List[Record], subtype_edges: List[Record], prereq_edges: List[Record]) -> Tuple[List[Relation], Dict[str, Any], List[Relation]]:
    concepts_by_name = _concept_map(concepts)
    typed_by_key: Dict[Tuple[str, str, str], Relation] = {}
    review_only: List[Relation] = []

    def add_rel(source: str, target: str, relation: str, origin: str, confidence: float, note: str,
                original: Optional[Record] = None, review_flag: bool = False):
        if not source or not target or source not in concepts_by_name or target not in concepts_by_name:
            return
        base: Relation = {
            "source": source,
            "target": target,
            "relation": relation,
            "origin": origin,
            "confidence": round(float(confidence), 3),
            "note": note,
        }
        if original:
            if original.get("reason"):
                base["original_reason"] = original.get("reason")
            if original.get("source"):
                base["original_source"] = original.get("source")

        # Low confidence / RELATED_TO is review-only, not a formal typed relation.
        if review_flag or relation == "RELATED_TO" or confidence < 0.60:
            base["review_only"] = True
            review_only.append(base)
            return

        key = (source, target, relation)
        existing = typed_by_key.get(key)
        if existing:
            origins = set(str(x) for x in existing.get("origins", [existing.get("origin")]) if x)
            origins.add(origin)
            existing["origins"] = sorted(origins)
            existing["origin"] = "+".join(sorted(origins))
            existing["confidence"] = max(existing.get("confidence", 0), round(float(confidence), 3))
            if original and original.get("reason"):
                rs = existing.setdefault("merged_reasons", [])
                if original.get("reason") not in rs:
                    rs.append(original.get("reason"))
            return

        base["origins"] = [origin]
        typed_by_key[key] = base

    for e in subtype_edges:
        p, c = str(e.get("parent") or ""), str(e.get("child") or "")
        rel, note, conf, review = classify_structure_edge(p, c, concepts_by_name)
        add_rel(p, c, rel, "subtype_edges", conf, note, e, review_flag=review)

    for e in prereq_edges:
        p, t = str(e.get("prereq") or ""), str(e.get("target") or "")
        rel, note, conf, review = classify_prereq_edge(e, concepts_by_name)
        add_rel(p, t, rel, "prereq_edges", conf, note, e, review_flag=review)

    typed = list(typed_by_key.values())
    typed.sort(key=lambda r: (r["relation"], r["source"], r["target"]))

    pair_to_rels = defaultdict(set)
    for r in typed:
        pair_to_rels[(r["source"], r["target"])].add(r["relation"])
    dual_pairs = [
        {"source": s, "target": t, "relations": sorted(list(rels))}
        for (s, t), rels in pair_to_rels.items()
        if len(rels) > 1
    ]

    counts = {rel: sum(1 for r in typed if r["relation"] == rel) for rel in {r["relation"] for r in typed}}
    report = {
        "typed_relation_version": "v8",
        "total_typed_relations": len(typed),
        "relation_type_counts": dict(sorted(counts.items())),
        "review_only_relations_count": len(review_only),
        "review_only_sample": review_only[:200],
        "dual_relation_pairs_count": len(dual_pairs),
        "dual_relation_pairs": dual_pairs[:300],
        "notes": [
            "RELATED_TO and low-confidence relations are excluded from formal typed_relations.json and written to review_only_relations.json.",
            "Deduplication key is (source, target, relation), so identical triples from subtype/prereq are merged.",
        ],
    }
    return typed, report, review_only


# -----------------------------
# reports
# -----------------------------

def build_alias_conflict_report(concepts: List[Record]) -> Dict[str, Any]:
    alias_to_names: Dict[str, set] = defaultdict(set)
    raw_aliases: Dict[str, set] = defaultdict(set)

    for c in concepts:
        name = str(c.get("name") or "")
        aliases = list(c.get("aliases") or []) + [name]
        for a in aliases:
            key = _norm_text(a)
            if not key or len(key) <= 1:
                continue
            alias_to_names[key].add(name)
            raw_aliases[key].add(str(a))

    conflicts = []
    for key, names in alias_to_names.items():
        if len(names) <= 1:
            continue
        names_list = sorted(names)
        risk = "review_required"
        if len(names_list) == 2 and any(n.lower() == key.replace(" ", "").lower() for n in names_list):
            risk = "likely_duplicate"
        if any(s in " ".join(names_list) for s in ["Mechanism", "System", "Communication", "Function", "Structure", "Model"]):
            risk = "different_abstraction_level_possible"
        conflicts.append({
            "normalized_alias": key,
            "raw_aliases": sorted(raw_aliases[key]),
            "concepts": names_list,
            "risk": risk,
            "recommendation": "review only; do not auto-merge unless definitions and edge roles are equivalent",
        })

    conflicts.sort(key=lambda x: (-len(x["concepts"]), x["normalized_alias"]))
    return {
        "alias_conflict_report_version": "v8",
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
    }


def build_derived_concept_report(concepts: List[Record]) -> Dict[str, Any]:
    derived_markers = (
        "Flow", "Model", "Property", "Technique", "Mechanism", "Hierarchy", "Classification",
        "Scheme", "Strategy", "Pattern", "Paradigm", "Possibility", "Requirement", "Constraint",
        "Ranking", "Architecture", "Facility", "Option",
    )
    candidates = []
    for c in concepts:
        name = str(c.get("name") or "")
        definition = str(c.get("definition") or "")
        aliases = c.get("aliases") or []
        marker_hit = [m for m in derived_markers if name.endswith(m) or m in name]
        if marker_hit:
            candidates.append({
                "name": name,
                "markers": marker_hit,
                "definition": definition[:220],
                "aliases": aliases[:8],
                "recommendation": "mark as derived_concept unless the exact term appears in PDF text/figure caption",
            })
    return {
        "derived_concept_report_version": "v8",
        "candidate_count": len(candidates),
        "candidates": candidates[:500],
    }


def generate_relation_outputs(concepts: List[Record], subtype_edges: List[Record], prereq_edges: List[Record]) -> Dict[str, Any]:
    typed, relation_quality, review_only = build_typed_relations(concepts, subtype_edges, prereq_edges)
    alias_report = build_alias_conflict_report(concepts)
    derived_report = build_derived_concept_report(concepts)
    return {
        "typed_relations": typed,
        "relation_quality_report": relation_quality,
        "review_only_relations": review_only,
        "alias_conflict_report": alias_report,
        "derived_concept_report": derived_report,
    }


def write_relation_outputs(output_dir: Path, book_id: str, concepts: List[Record], subtype_edges: List[Record], prereq_edges: List[Record]) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = generate_relation_outputs(concepts, subtype_edges, prereq_edges)
    (output_dir / f"{book_id}_typed_relations.json").write_text(json.dumps(outputs["typed_relations"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / f"{book_id}_relation_quality_report.json").write_text(json.dumps(outputs["relation_quality_report"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / f"{book_id}_review_only_relations.json").write_text(json.dumps(outputs["review_only_relations"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / f"{book_id}_alias_conflict_report.json").write_text(json.dumps(outputs["alias_conflict_report"], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / f"{book_id}_derived_concept_report.json").write_text(json.dumps(outputs["derived_concept_report"], ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs
