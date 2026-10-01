"""Coverage guard for GraphRAG JSON extraction.

This module complements LLM extraction with deterministic, conservative coverage checks.
It is designed to reduce missed textbook facts such as chapter objectives, figure/diagram
terms, process memory layout, PCB fields, scheduling queues, and code/API workflow items.

Key design:
- It never calls an LLM.
- It does not try to infer arbitrary semantics.
- It adds only concepts/edges that are strongly signaled by source text patterns.
- It is safe for Neo4j import because every added edge endpoint is also added as a concept.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from kg_builder.concept_name_normalizer import normalize_graph_json_records
except Exception:  # pragma: no cover
    normalize_graph_json_records = None  # type: ignore

Concept = Dict[str, Any]
Edge = Dict[str, Any]

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _clean_text(s: Any) -> str:
    return str(s or "").replace("\x0c", " ")


def text_from_chunks(chunks: Optional[Iterable[Any]] = None, raw_text: Optional[str] = None) -> str:
    """Best-effort text collector for project Chunk objects or plain strings."""
    parts: List[str] = []
    if raw_text:
        parts.append(str(raw_text))
    if chunks:
        for ch in chunks:
            if isinstance(ch, str):
                parts.append(ch)
                continue
            for attr in ("text", "content", "page_content", "chunk_text", "raw_text"):
                value = getattr(ch, attr, None)
                if value:
                    parts.append(str(value))
                    break
            else:
                if isinstance(ch, dict):
                    for key in ("text", "content", "page_content", "chunk_text", "raw_text"):
                        if ch.get(key):
                            parts.append(str(ch[key]))
                            break
    return "\n".join(_clean_text(p) for p in parts if p)


def text_from_pdf_path(pdf_path: Path) -> str:
    """Read PDF text without depending on project internals."""
    try:
        import fitz  # PyMuPDF
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("需要 PyMuPDF / fitz 才能從 PDF 做 coverage enhance") from exc
    doc = fitz.open(str(pdf_path))
    try:
        return "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _contains(text: str, *needles: str) -> bool:
    lower = text.lower()
    return all(n.lower() in lower for n in needles)


def _any_contains(text: str, needles: Sequence[str]) -> bool:
    lower = text.lower()
    return any(n.lower() in lower for n in needles)


def _concept(
    name: str,
    definition: str,
    aliases: Optional[List[str]] = None,
    category: str = "作業系統",
    parent: Optional[str] = None,
    is_fine_grained: bool = True,
    book_id: Optional[str] = None,
) -> Concept:
    rec: Concept = {
        "name": name,
        "definition": definition,
        "aliases": aliases or [re.sub(r"(?<!^)(?=[A-Z])", " ", name)],
        "category": category,
        "parent": parent,
        "is_fine_grained": is_fine_grained,
    }
    if book_id:
        rec["book_id"] = book_id
        rec["book_ids"] = [book_id]
    rec["source"] = "coverage_guard"
    return rec


def _sub(parent: str, child: str) -> Edge:
    return {"parent": parent, "child": child, "source": "coverage_guard"}


def _pre(prereq: str, target: str, reason: str, confidence: float = 0.90) -> Edge:
    return {
        "prereq": prereq,
        "target": target,
        "confidence": confidence,
        "reason": reason,
        "source": "coverage_guard",
    }


def _concept_names(concepts: Iterable[Concept]) -> set[str]:
    return {str(c.get("name")) for c in concepts if c.get("name")}


def _add_unique_concept(concepts: List[Concept], c: Concept, added: List[str]) -> None:
    names = _concept_names(concepts)
    if c["name"] not in names:
        concepts.append(c)
        added.append(c["name"])
    else:
        # Merge aliases / book_ids into existing concept without overwriting definition.
        for old in concepts:
            if old.get("name") == c["name"]:
                aliases = []
                for x in list(old.get("aliases") or []) + list(c.get("aliases") or []):
                    if x and x not in aliases:
                        aliases.append(x)
                old["aliases"] = aliases
                if not old.get("parent") and c.get("parent"):
                    old["parent"] = c.get("parent")
                break


def _add_unique_sub(edges: List[Edge], e: Edge, added: List[str]) -> None:
    key = (e.get("parent"), e.get("child"))
    if not key[0] or not key[1] or key[0] == key[1]:
        return
    if key not in {(x.get("parent"), x.get("child")) for x in edges}:
        edges.append(e)
        added.append(f"{key[0]} -> {key[1]}")


def _add_unique_pre(edges: List[Edge], e: Edge, added: List[str]) -> None:
    key = (e.get("prereq"), e.get("target"))
    if not key[0] or not key[1] or key[0] == key[1]:
        return
    if key not in {(x.get("prereq"), x.get("target")) for x in edges}:
        edges.append(e)
        added.append(f"{key[0]} -> {key[1]}")


# ---------------------------------------------------------------------------
# Deterministic pattern rules
# ---------------------------------------------------------------------------

def _rule_process_memory_layout(text: str, book_id: Optional[str]) -> Tuple[List[Concept], List[Edge], List[Edge]]:
    if not (_contains(text, "text section") and _contains(text, "heap section") and _contains(text, "stack section")):
        return [], [], []
    cs = [
        _concept("ProcessMemoryLayout", "ProcessMemoryLayout describes the memory sections of a process, commonly including text, data, heap, and stack regions.", ["Process Memory Layout", "Layout of a process in memory", "process memory layout", "行程記憶體配置"], parent="Process", is_fine_grained=False, book_id=book_id),
        _concept("TextSection", "TextSection is the fixed portion of a process memory layout that stores executable program code.", ["Text Section", "text section", "executable code", "程式碼區段"], parent="ProcessMemoryLayout", book_id=book_id),
        _concept("DataSection", "DataSection is the fixed portion of a process memory layout that stores global variables and static program data.", ["Data Section", "data section", "global variables", "資料區段"], parent="ProcessMemoryLayout", book_id=book_id),
        _concept("HeapSection", "HeapSection is the dynamically allocated memory region of a process that grows and shrinks during program execution.", ["Heap Section", "heap section", "heap", "堆積區段"], parent="ProcessMemoryLayout", book_id=book_id),
        _concept("StackSection", "StackSection is the process memory region used for temporary function-call data such as parameters, local variables, and return addresses.", ["Stack Section", "stack section", "stack", "堆疊區段"], parent="ProcessMemoryLayout", book_id=book_id),
        _concept("ExecutableCode", "ExecutableCode is the machine-executable instruction content stored in the text section of a process.", ["Executable Code", "executable code"], parent="TextSection", book_id=book_id),
        _concept("GlobalVariable", "GlobalVariable is program data stored in the data section and accessible beyond a single local function scope.", ["Global Variable", "global variables", "全域變數"], parent="DataSection", book_id=book_id),
        _concept("DynamicMemoryAllocation", "DynamicMemoryAllocation is the run-time allocation and release of memory, usually represented by growth and shrinkage of the heap section.", ["Dynamic Memory Allocation", "dynamically allocated", "動態記憶體配置"], parent="HeapSection", book_id=book_id),
        _concept("ActivationRecord", "ActivationRecord is a stack frame pushed during a function call that contains function parameters, local variables, and return address data.", ["Activation Record", "activation record", "stack frame", "活化紀錄"], parent="StackSection", book_id=book_id),
        _concept("FunctionParameter", "FunctionParameter is an input value passed to a function call and commonly stored within an activation record on the stack.", ["Function Parameter", "function parameters", "參數"], parent="ActivationRecord", book_id=book_id),
        _concept("ReturnAddress", "ReturnAddress is the address stored during a function call that indicates where execution resumes after the function returns.", ["Return Address", "return address", "返回位址"], parent="ActivationRecord", book_id=book_id),
        _concept("LocalVariable", "LocalVariable is a variable scoped to a function or block and commonly stored as temporary data in a stack activation record.", ["Local Variable", "local variables", "區域變數"], parent="ActivationRecord", book_id=book_id),
        _concept("ExecutableFile", "ExecutableFile is a passive file containing program instructions that becomes part of a process when loaded into memory.", ["Executable File", "executable file", "可執行檔"], parent="Program", book_id=book_id),
        _concept("ProgramCounter", "ProgramCounter is a processor register value indicating the address of the next instruction to execute for a process.", ["Program Counter", "program counter", "PC", "程式計數器"], parent=None, book_id=book_id),
    ]
    subs = [_sub("ProcessMemoryLayout", x) for x in ["TextSection", "DataSection", "HeapSection", "StackSection"]]
    subs += [_sub("TextSection", "ExecutableCode"), _sub("DataSection", "GlobalVariable"), _sub("HeapSection", "DynamicMemoryAllocation"), _sub("StackSection", "ActivationRecord")]
    pres = [_pre("Process", "ProcessMemoryLayout", "A process must be understood before its memory layout sections can be interpreted.")]
    for x in ["TextSection", "DataSection", "HeapSection", "StackSection"]:
        pres.append(_pre("ProcessMemoryLayout", x, f"{x} is one section of the process memory layout."))
    pres += [
        _pre("TextSection", "ExecutableCode", "Executable code is stored in the text section."),
        _pre("DataSection", "GlobalVariable", "Global variables are stored in the data section."),
        _pre("HeapSection", "DynamicMemoryAllocation", "Dynamic allocation is represented by heap growth and shrinkage."),
        _pre("StackSection", "ActivationRecord", "Function calls push activation records onto the stack."),
    ]
    return cs, subs, pres


def _rule_process_states(text: str, book_id: Optional[str]) -> Tuple[List[Concept], List[Edge], List[Edge]]:
    if not (_contains(text, "process state") and _contains(text, "new") and _contains(text, "running") and _contains(text, "terminated")):
        return [], [], []
    states = ["NewState", "ReadyState", "RunningState", "WaitingState", "TerminatedState"]
    cs = [_concept("ProcessState", "ProcessState is the current execution condition of a process, such as new, ready, running, waiting, or terminated.", ["Process State", "process state", "行程狀態"], parent="Process", is_fine_grained=False, book_id=book_id)]
    defs = {
        "NewState": "NewState is the process state in which a process is being created.",
        "ReadyState": "ReadyState is the process state in which a process is waiting to be assigned to a processor.",
        "RunningState": "RunningState is the process state in which instructions are being executed.",
        "WaitingState": "WaitingState is the process state in which a process is waiting for an event such as input/output completion.",
        "TerminatedState": "TerminatedState is the process state in which a process has finished execution.",
    }
    aliases = {
        "NewState": ["New", "new state", "新增狀態"],
        "ReadyState": ["Ready", "ready state", "就緒狀態"],
        "RunningState": ["Running", "running state", "執行狀態"],
        "WaitingState": ["Waiting", "waiting state", "等待狀態"],
        "TerminatedState": ["Terminated", "terminated state", "終止狀態"],
    }
    cs += [_concept(s, defs[s], aliases[s], parent="ProcessState", book_id=book_id) for s in states]
    subs = [_sub("ProcessState", s) for s in states]
    pres = [_pre("Process", "ProcessState", "A process must be understood before its state model can be interpreted.")]
    # State-transition facts are represented as prerequisite-like flow hints because the current schema has no generic FLOW edge.
    pres += [
        _pre("NewState", "ReadyState", "A newly created process is admitted into the ready state in the process-state diagram.", 0.85),
        _pre("ReadyState", "RunningState", "Scheduler dispatch moves a ready process into the running state.", 0.85),
        _pre("RunningState", "WaitingState", "A running process may wait for input/output or another event.", 0.85),
        _pre("WaitingState", "ReadyState", "Completion of input/output or an event returns a waiting process to the ready state.", 0.85),
        _pre("RunningState", "TerminatedState", "A running process reaches termination when it exits.", 0.85),
    ]
    return cs, subs, pres


def _rule_pcb(text: str, book_id: Optional[str]) -> Tuple[List[Concept], List[Edge], List[Edge]]:
    if not (_contains(text, "process control block") or _contains(text, "task control block") or _contains(text, "task_struct")):
        return [], [], []
    fields = [
        ("ProcessControlBlock", "ProcessControlBlock is the operating-system data structure that stores all information needed to represent, start, or restart a process.", ["Process Control Block", "PCB", "task control block", "Task Control Block", "行程控制區塊"], None, False),
        ("CentralProcessingUnitRegister", "CentralProcessingUnitRegister is a CPU register whose value may be saved in a process control block during interrupts and context switches.", ["CPU Register", "CPU registers", "processor registers", "暫存器"], "ProcessControlBlock", True),
        ("CentralProcessingUnitSchedulingInformation", "CentralProcessingUnitSchedulingInformation stores scheduling-related data such as process priority, queue pointers, and scheduling parameters.", ["CPU Scheduling Information", "CPU-scheduling information", "scheduling information"], "ProcessControlBlock", True),
        ("MemoryManagementInformation", "MemoryManagementInformation stores process memory management data such as base and limit registers, page tables, or segment tables.", ["Memory Management Information", "memory-management information"], "ProcessControlBlock", True),
        ("AccountingInformation", "AccountingInformation stores usage and accounting data such as CPU time, real time, time limits, account numbers, and job or process numbers.", ["Accounting Information", "accounting information"], "ProcessControlBlock", True),
        ("InputOutputStatusInformation", "InputOutputStatusInformation stores process input/output status such as allocated devices and open file lists.", ["I/O Status Information", "Input Output Status Information", "I/O status information"], "ProcessControlBlock", True),
        ("OpenFileList", "OpenFileList is the list of files opened by or allocated to a process, often stored in a process control block or related process structure.", ["Open File List", "list of open files", "開啟檔案清單"], "InputOutputStatusInformation", True),
        ("MemoryLimit", "MemoryLimit is a memory boundary value stored as part of process management information.", ["Memory Limit", "memory limits"], "MemoryManagementInformation", True),
        ("ProcessNumber", "ProcessNumber is an accounting or identification number associated with a process or job.", ["Process Number", "process number", "job number"], "ProcessControlBlock", True),
        ("TaskStructure", "TaskStructure is the Linux kernel task_struct data structure used to represent active processes.", ["task_struct", "task struct", "Task Structure", "Linux task_struct"], "ProcessControlBlock", True),
    ]
    cs = [_concept(n, d, a, parent=p, is_fine_grained=fg, book_id=book_id) for n, d, a, p, fg in fields]
    # ProgramCounter / ProcessState may have been added by other rules, but include the PCB relationship here.
    cs += [
        _concept("ProgramCounter", "ProgramCounter is a processor register value indicating the address of the next instruction to execute for a process.", ["Program Counter", "program counter", "PC"], parent="ProcessControlBlock", book_id=book_id),
        _concept("ProcessState", "ProcessState is the current execution condition of a process, such as new, ready, running, waiting, or terminated.", ["Process State", "process state"], parent="ProcessControlBlock", book_id=book_id),
    ]
    children = ["ProcessState", "ProgramCounter", "CentralProcessingUnitRegister", "CentralProcessingUnitSchedulingInformation", "MemoryManagementInformation", "AccountingInformation", "InputOutputStatusInformation", "OpenFileList", "MemoryLimit", "ProcessNumber", "TaskStructure"]
    subs = [_sub("ProcessControlBlock", x) for x in children]
    pres = [_pre("Process", "ProcessControlBlock", "A process must be understood before the data structure representing it can be understood.")]
    for x in children:
        pres.append(_pre("ProcessControlBlock", x, f"{x} is information stored in or associated with the process control block.", 0.88))
    return cs, subs, pres


def _rule_scheduling(text: str, book_id: Optional[str]) -> Tuple[List[Concept], List[Edge], List[Edge]]:
    if not (_contains(text, "process scheduling") or _contains(text, "ready queue") or _contains(text, "context switch")):
        return [], [], []
    items = [
        ("ProcessScheduling", "ProcessScheduling is the operating-system activity of selecting processes for execution to support multiprogramming and time sharing.", ["Process Scheduling", "process scheduling"], None, False),
        ("Multiprogramming", "Multiprogramming is the objective of keeping some process running at all times to maximize CPU utilization.", ["multiprogramming", "多道程式設計"], "ProcessScheduling", True),
        ("TimeSharing", "TimeSharing is the objective of switching a CPU core among processes frequently enough to support interactive use.", ["time sharing", "分時"], "ProcessScheduling", True),
        ("ProcessScheduler", "ProcessScheduler is the operating-system component that selects an available process for execution on a processor core.", ["process scheduler", "scheduler", "行程排程器"], "ProcessScheduling", True),
        ("CentralProcessingUnitScheduler", "CentralProcessingUnitScheduler selects a process from the ready queue and allocates a CPU core to it.", ["CPU Scheduler", "CPU scheduler", "Central Processing Unit Scheduler"], "ProcessScheduler", True),
        ("ReadyQueue", "ReadyQueue is the queue of processes that are ready and waiting to execute on a CPU core.", ["Ready Queue", "ready queue", "就緒佇列"], "ProcessScheduling", True),
        ("WaitQueue", "WaitQueue is a queue containing processes that are waiting for a particular event such as input/output completion.", ["Wait Queue", "wait queue", "等待佇列"], "ProcessScheduling", True),
        ("InputOutputBoundProcess", "InputOutputBoundProcess is a process that spends more time performing input/output than computation.", ["I/O-bound Process", "I/O bound process", "Input Output Bound Process"], "ProcessScheduling", True),
        ("CentralProcessingUnitBoundProcess", "CentralProcessingUnitBoundProcess is a process that performs more computation and generates input/output requests less frequently.", ["CPU-bound Process", "CPU bound process", "Central Processing Unit Bound Process"], "ProcessScheduling", True),
        ("DegreeOfMultiprogramming", "DegreeOfMultiprogramming is the number of processes currently in memory.", ["degree of multiprogramming", "多道程式程度"], "Multiprogramming", True),
        ("Swapping", "Swapping is the scheduling-related memory technique of moving a process out of memory to disk and later bringing it back.", ["swapping", "swap out", "swap in", "交換"], "ProcessScheduling", True),
        ("SwappedOutProcess", "SwappedOutProcess is a process whose current state has been saved from memory to disk.", ["swapped out", "swapped-out process"], "Swapping", True),
        ("SwappedInProcess", "SwappedInProcess is a process restored from disk back to memory so execution can continue.", ["swapped in", "swapped-in process"], "Swapping", True),
        ("ContextSwitch", "ContextSwitch is the operating-system action of saving the state of one process and restoring the state of another process on a CPU core.", ["Context Switch", "context switch", "context-switch"], "ProcessScheduling", True),
        ("StateSave", "StateSave is the operation of saving the current CPU or process state before switching execution context.", ["State Save", "state save", "save state"], "ContextSwitch", True),
        ("StateRestore", "StateRestore is the operation of reloading a previously saved CPU or process state to resume execution.", ["State Restore", "state restore", "restore state"], "ContextSwitch", True),
        ("ContextSwitchTime", "ContextSwitchTime is pure overhead time spent performing a context switch rather than useful application work.", ["Context Switch Time", "context-switch time", "context switch overhead"], "ContextSwitch", True),
        ("TimeSlice", "TimeSlice is the limited interval of CPU time after which a running process may be interrupted and returned to the ready queue.", ["Time Slice", "time slice", "time slice expired"], "ProcessScheduling", True),
    ]
    cs = [_concept(n, d, a, parent=p, is_fine_grained=fg, book_id=book_id) for n, d, a, p, fg in items]
    subs = [_sub(p, n) for n, _, _, p, _ in items if p]
    pres = [_pre("Process", "ProcessScheduling", "Scheduling is defined over processes that are managed by the operating system.")]
    pres += [
        _pre("ProcessScheduling", "ReadyQueue", "The ready queue is part of process scheduling."),
        _pre("ProcessScheduling", "WaitQueue", "Wait queues are part of process scheduling."),
        _pre("ReadyQueue", "CentralProcessingUnitScheduler", "The CPU scheduler selects from the ready queue."),
        _pre("ProcessScheduling", "ContextSwitch", "Context switches occur as the scheduler changes which process runs."),
        _pre("ContextSwitch", "StateSave", "A context switch requires saving process state."),
        _pre("ContextSwitch", "StateRestore", "A context switch requires restoring another process state."),
        _pre("ContextSwitch", "ContextSwitchTime", "Context-switch time is the overhead caused by switching context."),
    ]
    return cs, subs, pres


def _rule_threads(text: str, book_id: Optional[str]) -> Tuple[List[Concept], List[Edge], List[Edge]]:
    if not (_contains(text, "threads") and _contains(text, "single thread of execution")):
        return [], [], []
    cs = [
        _concept("Thread", "Thread is a thread of execution within a process, allowing a process to perform one or more tasks concurrently.", ["Thread", "threads", "執行緒"], parent="Process", book_id=book_id),
        _concept("SingleThreadOfExecution", "SingleThreadOfExecution is a process execution model in which only one sequence of instructions runs at a time.", ["single thread of execution", "single thread"], parent="Thread", book_id=book_id),
        _concept("MultithreadedProcess", "MultithreadedProcess is a process containing multiple threads of execution so it can perform more than one task at a time.", ["multithreaded process", "multiple threads of execution"], parent="Process", book_id=book_id),
        _concept("MulticoreSystem", "MulticoreSystem is a computer system with multiple CPU cores, enabling multiple threads or processes to run in parallel.", ["multicore system", "multiple cores", "多核心系統"], parent=None, book_id=book_id),
    ]
    subs = [_sub("Process", "Thread"), _sub("Process", "MultithreadedProcess"), _sub("Thread", "SingleThreadOfExecution")]
    pres = [_pre("Process", "Thread", "Threads are introduced as an extension of the process concept."), _pre("Thread", "MultithreadedProcess", "A multithreaded process contains multiple threads.")]
    return cs, subs, pres


def _rule_code_identifiers(text: str, book_id: Optional[str]) -> Tuple[List[Concept], List[Edge], List[Edge]]:
    cs: List[Concept] = []
    subs: List[Edge] = []
    pres: List[Edge] = []
    code_map = [
        ("shm_open", "SharedMemoryOpenFunction", "SharedMemoryOpenFunction is a PortableOperatingSystemInterface shared-memory function that opens or creates a shared-memory object.", ["shm_open", "shm_open()"], "PortableOperatingSystemInterfaceSharedMemoryApplicationProgrammingInterface"),
        ("ftruncate", "FtruncateFunction", "FtruncateFunction sets the size of a file or shared-memory object in POSIX shared-memory programming.", ["ftruncate", "ftruncate()"], "PortableOperatingSystemInterfaceSharedMemoryApplicationProgrammingInterface"),
        ("mmap", "MemoryMapFunction", "MemoryMapFunction maps a file or shared-memory object into a process address space.", ["mmap", "mmap()"], "PortableOperatingSystemInterfaceSharedMemoryApplicationProgrammingInterface"),
        ("shm_unlink", "SharedMemoryUnlinkFunction", "SharedMemoryUnlinkFunction removes a POSIX shared-memory object name.", ["shm_unlink", "shm_unlink()"], "PortableOperatingSystemInterfaceSharedMemoryApplicationProgrammingInterface"),
        ("pipe(", "PipeFunction", "PipeFunction creates a pipe for interprocess communication between related processes.", ["pipe", "pipe()"], "Pipe"),
        ("CreatePipe", "CreatePipeFunction", "CreatePipeFunction is a Windows function that creates an anonymous pipe.", ["CreatePipe", "CreatePipe()"], "Pipe"),
        ("CloseHandle", "CloseHandleFunction", "CloseHandleFunction closes an open Windows object handle to release system resources.", ["CloseHandle", "CloseHandle()"], "WindowsApplicationProgrammingInterface"),
    ]
    for needle, name, definition, aliases, parent in code_map:
        if needle.lower() in text.lower():
            cs.append(_concept(name, definition, aliases, parent=parent, book_id=book_id))
            cs.append(_concept(parent, f"{parent} is a related parent concept for {name}.", [re.sub(r"(?<!^)(?=[A-Z])", " ", parent)], is_fine_grained=False, book_id=book_id))
            subs.append(_sub(parent, name))
            pres.append(_pre(parent, name, f"{name} is used within or as part of {parent}."))
    return cs, subs, pres


RULES = [_rule_process_memory_layout, _rule_process_states, _rule_pcb, _rule_scheduling, _rule_threads, _rule_code_identifiers]


def enhance_graph_from_text(
    concepts: Iterable[Concept],
    subtype_edges: Iterable[Edge],
    prereq_edges: Iterable[Edge],
    text: str,
    book_id: Optional[str] = None,
) -> Tuple[List[Concept], List[Edge], List[Edge], Dict[str, Any]]:
    """Add high-confidence missing concepts and edges based on source text."""
    fixed_concepts = [deepcopy(c) for c in concepts]
    fixed_subtypes = [deepcopy(e) for e in subtype_edges]
    fixed_prereqs = [deepcopy(e) for e in prereq_edges]
    added_concepts: List[str] = []
    added_subtypes: List[str] = []
    added_prereqs: List[str] = []

    text = _clean_text(text)
    triggered: List[str] = []
    for rule in RULES:
        cs, subs, pres = rule(text, book_id)
        if cs or subs or pres:
            triggered.append(rule.__name__)
        for c in cs:
            _add_unique_concept(fixed_concepts, c, added_concepts)
        for e in subs:
            # Ensure endpoints exist.
            for n in [e.get("parent"), e.get("child")]:
                if n and n not in _concept_names(fixed_concepts):
                    _add_unique_concept(fixed_concepts, _concept(str(n), f"{n} was added as an endpoint required by coverage guard.", book_id=book_id), added_concepts)
            _add_unique_sub(fixed_subtypes, e, added_subtypes)
        for e in pres:
            for n in [e.get("prereq"), e.get("target")]:
                if n and n not in _concept_names(fixed_concepts):
                    _add_unique_concept(fixed_concepts, _concept(str(n), f"{n} was added as an endpoint required by coverage guard.", book_id=book_id), added_concepts)
            _add_unique_pre(fixed_prereqs, e, added_prereqs)

    if normalize_graph_json_records:
        fixed_concepts, fixed_subtypes, fixed_prereqs, _ = normalize_graph_json_records(fixed_concepts, fixed_subtypes, fixed_prereqs)

    names = _concept_names(fixed_concepts)
    missing_subtype = [e for e in fixed_subtypes if e.get("parent") not in names or e.get("child") not in names]
    missing_prereq = [e for e in fixed_prereqs if e.get("prereq") not in names or e.get("target") not in names]
    report = {
        "coverage_guard_version": "v6",
        "triggered_rules": triggered,
        "added_concepts_count": len(set(added_concepts)),
        "added_subtype_edges_count": len(set(added_subtypes)),
        "added_prereq_edges_count": len(set(added_prereqs)),
        "added_concepts": sorted(set(added_concepts)),
        "added_subtype_edges": sorted(set(added_subtypes)),
        "added_prereq_edges": sorted(set(added_prereqs)),
        "missing_subtype_endpoints_after_enhance": len(missing_subtype),
        "missing_prereq_endpoints_after_enhance": len(missing_prereq),
    }
    return fixed_concepts, fixed_subtypes, fixed_prereqs, report


def enhance_graph_from_chunks(
    concepts: Iterable[Concept],
    subtype_edges: Iterable[Edge],
    prereq_edges: Iterable[Edge],
    chunks: Optional[Iterable[Any]] = None,
    raw_text: Optional[str] = None,
    book_id: Optional[str] = None,
) -> Tuple[List[Concept], List[Edge], List[Edge], Dict[str, Any]]:
    return enhance_graph_from_text(concepts, subtype_edges, prereq_edges, text_from_chunks(chunks, raw_text), book_id=book_id)
