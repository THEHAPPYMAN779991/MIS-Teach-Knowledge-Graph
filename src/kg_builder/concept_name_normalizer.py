"""Normalize extracted GraphRAG concept names to English PascalCase.

目的：
1. concept.name 一律改成完整英文 PascalCase，例如 IPC -> InterprocessCommunication。
2. 縮寫、中文、英文空格寫法保留在 aliases。
3. concepts[].parent、subtype_edges[].parent/child、prereq_edges[].prereq/target 同步改名，避免 Neo4j 建邊失敗。
4. 僅做「形式正規化」與「保守同義合併」：大小寫、空白、標點、明確縮寫展開可以合併；詞性/本質/應用場景不同的近義詞不可硬合併。

重要原則：
- 這個模組面向所有教材與所有 JSON，不針對單一詞彙特例硬解。
- ProcessID / ProcessIdentifier / ProcessIdentification 只是示例；任何 XID、XIdentifier、XIdentification、XFunction、XMechanism、XSystem、XObject 等，只要本質不同，都應保留不同節點。
- 不呼叫 LLM；不會猜測所有中文術語的英文全名，所以仍建議同時修改 prompts.py，讓 LLM 直接輸出英文標準名。
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+")

# 直接指定：常見縮寫、中文術語、混合術語 → 標準 PascalCase 名稱
DIRECT_NAME_MAP: Dict[str, str] = {
    # OS / IPC
    "IPC": "InterprocessCommunication",
    "ipc": "InterprocessCommunication",
    "Inter-process Communication": "InterprocessCommunication",
    "Interprocess Communication": "InterprocessCommunication",
    "行程間通訊": "InterprocessCommunication",
    "進程間通訊": "InterprocessCommunication",
    "程序間通訊": "InterprocessCommunication",
    "共享記憶體": "SharedMemory",
    "共享內存": "SharedMemory",
    "Shared Memory": "SharedMemory",
    "共享記憶體區段": "SharedMemorySegment",
    "Shared Memory Segment": "SharedMemorySegment",
    "共享記憶體物件": "SharedMemoryObject",
    "Shared Memory Object": "SharedMemoryObject",
    "POSIX 共享記憶體 API": "PortableOperatingSystemInterfaceSharedMemoryApplicationProgrammingInterface",
    "POSIX Shared Memory API": "PortableOperatingSystemInterfaceSharedMemoryApplicationProgrammingInterface",
    "POSIX shared memory API": "PortableOperatingSystemInterfaceSharedMemoryApplicationProgrammingInterface",
    "POSIX": "PortableOperatingSystemInterface",

    # PID family: keep related-but-different concepts separate.
    # ProcessID        = concrete numeric/code identifier assigned to a process, e.g. PID 1402.
    # ProcessIdentifier = generic identifier field/object/type used to identify a process.
    # ProcessIdentification = the act/mechanism/process of identifying a process.
    # Do NOT merge these three into one node.
    "PID": "ProcessID",
    "Pid": "ProcessID",
    "pid": "ProcessID",
    "pid_t": "ProcessID",
    "Process ID": "ProcessID",
    "Process Id": "ProcessID",
    "process ID": "ProcessID",
    "process id": "ProcessID",
    "ProcessID": "ProcessID",
    "ProcessId": "ProcessID",
    "ProcessIdentifier": "ProcessIdentifier",
    "Process Identifier": "ProcessIdentifier",
    "process identifier": "ProcessIdentifier",
    "ProcessIdentification": "ProcessIdentification",
    "Process Identification": "ProcessIdentification",
    "process identification": "ProcessIdentification",
    "ProcessIdentificationNumber": "ProcessID",

    # Init / initial process are the same concept in OS textbooks.
    "InitProcess": "InitialProcess",
    "Init Process": "InitialProcess",
    "init process": "InitialProcess",
    "InitialProcess": "InitialProcess",
    "Initial Process": "InitialProcess",
    "initial process": "InitialProcess",

    # Windows API names that may be extracted with or without the word Function.
    "CreateProcess": "CreateProcessFunction",
    "CreateProcess()": "CreateProcessFunction",
    "CreateProcess Function": "CreateProcessFunction",
    "CreateProcessFunction": "CreateProcessFunction",
    "WaitForSingleObjectFunction": "WaitForSingleObject",
    "WaitForSingleObject": "WaitForSingleObject",
    "WaitForSingleObject()": "WaitForSingleObject",
    "ZeroMemoryFunction": "ZeroMemory",
    "ZeroMemory": "ZeroMemory",
    "ZeroMemory()": "ZeroMemory",

    # Mailbox spelling normalization.
    "MailBox": "Mailbox",
    "Mailbox": "Mailbox",
    "Mailboxes": "Mailbox",
    "mailbox": "Mailbox",
    "mailboxes": "Mailbox",

    # Common IPC naming variants. Keep mechanism/system/model terms separate from the base concept.
    # Example: MessagePassing, MessagePassingMechanism, and MessagePassingSystem may be related,
    # but they are not always identical. Do not collapse them here.
    "Message Passing": "MessagePassing",
    "message passing": "MessagePassing",
    "Shared Memory": "SharedMemory",
    "shared memory": "SharedMemory",

    # Common system-call variants.
    "ExeclpSystemCall": "ExecLPSystemCall",
    "ExecLpSystemCall": "ExecLPSystemCall",
    "ExecLPSystemCall": "ExecLPSystemCall",
    "execlp": "ExecLPSystemCall",
    "execlp()": "ExecLPSystemCall",

    # Common CS abbreviations
    "CPU": "CentralProcessingUnit",
    "GPU": "GraphicsProcessingUnit",
    "RAM": "RandomAccessMemory",
    "ROM": "ReadOnlyMemory",
    "OS": "OperatingSystem",
    "API": "ApplicationProgrammingInterface",
    "SQL": "StructuredQueryLanguage",
    "DBMS": "DatabaseManagementSystem",
    "TCP": "TransmissionControlProtocol",
    "UDP": "UserDatagramProtocol",
    "IP": "InternetProtocol",
    "HTTP": "HypertextTransferProtocol",
    "HTTPS": "HypertextTransferProtocolSecure",
    "DNS": "DomainNameSystem",
    "I/O": "InputOutput",
    "IO": "InputOutput",
    "XML": "ExtensibleMarkupLanguage",
    "HTML": "HypertextMarkupLanguage",
    "CSS": "CascadingStyleSheets",
    "URL": "UniformResourceLocator",
    "URI": "UniformResourceIdentifier",
    "JSON": "JavaScriptObjectNotation",

    # Big-O / complexity forms
    "O(N)": "LinearTimeComplexity",
    "O(n)": "LinearTimeComplexity",
    "O(N^2)": "QuadraticTimeComplexity",
    "O(n^2)": "QuadraticTimeComplexity",
    "O(N2)": "QuadraticTimeComplexity",
    "O(N^3)": "CubicTimeComplexity",
    "O(n^3)": "CubicTimeComplexity",
    "O(N log N)": "LinearithmicTimeComplexity",
    "O(n log n)": "LinearithmicTimeComplexity",
    "O(log N)": "LogarithmicTimeComplexity",
    "O(log n)": "LogarithmicTimeComplexity",
    "O(1)": "ConstantTimeComplexity",

    # Data structures / algorithms common Chinese terms
    "演算法": "Algorithm",
    "算法": "Algorithm",
    "演算法分析": "AlgorithmAnalysis",
    "演算法效率": "AlgorithmEfficiency",
    "效能分析": "PerformanceAnalysis",
    "最佳情況效能": "BestCasePerformance",
    "平均情況效能": "AverageCasePerformance",
    "最差情況效能": "WorstCasePerformance",
    "計算機資源": "ComputerResources",
    "時間複雜度": "TimeComplexity",
    "執行時間複雜度": "RunningTimeComplexity",
    "Time Complexity": "TimeComplexity",
    "time complexity": "TimeComplexity",
    "Running Time Complexity": "RunningTimeComplexity",
    "running time complexity": "RunningTimeComplexity",
    "Running Time": "RunningTime",
    "running time": "RunningTime",
    "執行時間分析": "RunningTimeAnalysis",
    "空間複雜度": "SpaceComplexity",
    "大O表示法": "BigONotation",
    "大 O 表示法": "BigONotation",
    "大O符號": "BigONotation",
    "大 O 符號": "BigONotation",
    "Big-O 符號": "BigONotation",
    "Big-Oh 符號": "BigONotation",
    "大Omega表示法": "BigOmegaNotation",
    "大Omega符號": "BigOmegaNotation",
    "Omega符號": "OmegaNotation",
    "大Theta表示法": "BigThetaNotation",
    "大Theta符號": "BigThetaNotation",
    "Theta符號": "ThetaNotation",
    "小o符號": "LittleONotation",
    "小 o 符號": "LittleONotation",
    "漸近符號": "AsymptoticNotation",
    "漸近分析": "AsymptoticAnalysis",
    "漸進分析": "AsymptoticAnalysis",
    "函數成長率": "FunctionGrowthRate",
    "成長率": "GrowthRate",
    "增長率": "GrowthRate",
    "對數成長率": "LogarithmicGrowthRate",
    "線性時間複雜度": "LinearTimeComplexity",
    "二次時間複雜度": "QuadraticTimeComplexity",
    "平方時間複雜度": "QuadraticTimeComplexity",
    "平方級時間複雜度": "QuadraticTimeComplexity",
    "立方級時間複雜度": "CubicTimeComplexity",
    "立方時間複雜度": "CubicTimeComplexity",
    "指數時間複雜度": "ExponentialTimeComplexity",
    "常數時間複雜度": "ConstantTimeComplexity",
    "極限": "Limit",
    "羅必達法則": "LHopitalsRule",
    "洛必達法則": "LHopitalsRule",
    "導數": "Derivative",
    "函數": "Function",
    "函式": "Function",
    "對數": "Logarithm",
    "冪函數": "PowerFunction",
    "排序演算法": "SortingAlgorithm",
    "排序算法": "SortingAlgorithm",
    "氣泡排序": "BubbleSort",
    "插入排序": "InsertionSort",
    "快速排序": "QuickSort",
    "合併排序": "MergeSort",
    "執行時間": "RunningTime",
    "平均執行時間": "AverageRunningTime",
    "最差執行時間": "WorstCaseRunningTime",
    "計算模型": "ModelOfComputation",
    "指令": "Instruction",
    "簡單指令": "SimpleInstruction",
    "演算法輸入": "AlgorithmInput",
    "輸入大小": "InputSize",
    "程式": "Program",
    "實作效率不彰": "ImplementationInefficiency",
    "最大子序列和問題": "MaximumSubsequenceSumProblem",
    "最大子序列和": "MaximumSubsequenceSum",
    "最大連續子序列和演算法": "MaximumContiguousSubsequenceSumAlgorithm",
    "遞迴最大連續子序列和演算法": "RecursiveMaximumContiguousSubsequenceSumAlgorithm",
    "遞迴": "Recursion",
    "遞迴函數": "RecursiveFunction",
    "遞迴關係式": "RecurrenceRelation",
    "遞迴關係": "RecurrenceRelation",
    "遞迴呼叫": "RecursiveCall",
    "遞迴基底條件": "BaseCase",
    "基本情況": "BaseCase",
    "分治法": "DivideAndConquer",
    "陣列": "Array",
    "一維陣列": "OneDimensionalArray",
    "二維陣列": "TwoDimensionalArray",
    "多維陣列": "MultidimensionalArray",
    "動態陣列": "DynamicArray",
    "向量": "Vector",
    "子陣列": "Subarray",
    "迴圈": "Loop",
    "FOR 迴圈": "ForLoop",
    "for迴圈": "ForLoop",
    "巢狀迴圈": "NestedLoop",
    "迭代次數": "IterationCount",
    "陳述式": "Statement",
    "連續陳述式": "SequentialStatement",
}


# Correct casing / canonical spelling for names that commonly appear as lower-cased PascalCase
# after LLM extraction. These entries are checked before generic tokenization.
CANONICAL_CASE_MAP: Dict[str, str] = {
    # Windows / UNIX calls and structs
    "Createprocess": "CreateProcess",
    "CreateProcess": "CreateProcess",
    "Createprocessfunction": "CreateProcessFunction",
    "CreateProcessFunction": "CreateProcessFunction",
    "Waitforsingleobject": "WaitForSingleObject",
    "WaitForSingleObject": "WaitForSingleObject",
    "WaitForSingleObject()": "WaitForSingleObject",
    "Zeromemory": "ZeroMemory",
    "ZeroMemory": "ZeroMemory",
    "ZeroMemory()": "ZeroMemory",
    "StartupinfoStructure": "StartupInformationStructure",
    "StartupInfoStructure": "StartupInformationStructure",
    "Startupinformationstructure": "StartupInformationStructure",
    "StartupInformationStructure": "StartupInformationStructure",
    "STARTUPINFO": "StartupInformationStructure",
    "Processinformationstructure": "ProcessInformationStructure",
    "ProcessInformationStructure": "ProcessInformationStructure",
    "PROCESS_INFORMATION": "ProcessInformationStructure",
    "Processexecutionflow": "ProcessExecutionFlow",
    "ProcessExecutionFlow": "ProcessExecutionFlow",
    "Concurrentprocessexecution": "ConcurrentProcessExecution",
    "ConcurrentProcessExecution": "ConcurrentProcessExecution",
    "Concurrentexecution": "ConcurrentExecution",
    "ConcurrentExecution": "ConcurrentExecution",
    "Parentwaitingforchildtermination": "ParentWaitingForChildTermination",
    "ParentWaitingForChildTermination": "ParentWaitingForChildTermination",
    "Processmanagementcommand": "ProcessManagementCommand",
    "ProcessManagementCommand": "ProcessManagementCommand",
    "Resourcepartitioning": "ResourcePartitioning",
    "ResourcePartitioning": "ResourcePartitioning",
    "Resourcesharing": "ResourceSharing",
    "ResourceSharing": "ResourceSharing",
    "Browserprocess": "BrowserProcess",
    "BrowserProcess": "BrowserProcess",
    "Webbrowser": "WebBrowser",
    "WebBrowser": "WebBrowser",
    "Chromebrowser": "ChromeBrowser",
    "ChromeBrowser": "ChromeBrowser",
    "RendererProcesses": "RendererProcess",
    "RendererProcess": "RendererProcess",
    "PlugInProcess": "PluginProcess",
    "PluginProcess": "PluginProcess",
    "Diskinputoutput": "DiskInputOutput",
    "DiskInputOutput": "DiskInputOutput",
    "Networkinputoutput": "NetworkInputOutput",
    "NetworkInputOutput": "NetworkInputOutput",
    # IPC / message passing
    "Messagepassingmechanism": "MessagePassingMechanism",
    "MessagePassingMechanism": "MessagePassingMechanism",
    "MessagePassingmechanism": "MessagePassingMechanism",
    "MessagePassingSystem": "MessagePassingSystem",
    "Messagepassingsystem": "MessagePassingSystem",
    "MessagePassingSystems": "MessagePassingSystem",
    "Messagepassingsystems": "MessagePassingSystem",
    "SharedMemorySystems": "SharedMemorySystem",
    "SharedMemorySystem": "SharedMemorySystem",
    "Portableoperatingsysteminterfacemessagequeue": "PortableOperatingSystemInterfaceMessageQueue",
    "PortableOperatingSystemInterfaceMessageQueue": "PortableOperatingSystemInterfaceMessageQueue",
    "Portableoperatingsysteminterfacesharedmemoryopenfunction": "PortableOperatingSystemInterfaceSharedMemoryOpenFunction",
    "PortableOperatingSystemInterfaceSharedMemoryOpenFunction": "PortableOperatingSystemInterfaceSharedMemoryOpenFunction",
    "Operatingsystemownedmailbox": "OperatingSystemOwnedMailbox",
    "OperatingSystemOwnedMailbox": "OperatingSystemOwnedMailbox",
    "Blockingsend": "BlockingSend",
    "Blockingreceive": "BlockingReceive",
    "Nonblockingsend": "NonblockingSend",
    "Nonblockingreceive": "NonblockingReceive",
    "Sendmessageoperation": "SendMessageOperation",
    "Receivemessageoperation": "ReceiveMessageOperation",
    "FixedSizedMessages": "FixedSizedMessage",
    "VariableSizedMessages": "VariableSizedMessage",
    "Machmessagetrap": "MachMessageTrap",
    "Machmessageoverwritetrap": "MachMessageOverwriteTrap",
    "Taskselfport": "TaskSelfPort",
    "Notifyport": "NotifyPort",
    "BootstrapPort": "BootstrapPort",
    "PortObject": "PortObject",
    "ConnectionPort": "ConnectionPort",
    "CommunicationPort": "CommunicationPort",
    "CommunicationChannel": "CommunicationChannel",
    "PrivateCommunicationPort": "PrivateCommunicationPort",
    "CallbackMechanism": "CallbackMechanism",
    "Messagequeuefullhandling": "MessageQueueFullHandling",
    "MessageQueueFullHandling": "MessageQueueFullHandling",
    "WaitIndefinitely": "WaitIndefinitely",
    "WaitAtMostNMilliseconds": "WaitAtMostNMilliseconds",
    # Process / Android states
    "ZStateProcess": "ZombieProcess",
    "ZombieProcess": "ZombieProcess",
    "ForegroundProcess": "ForegroundProcess",
    "VisibleProcess": "VisibleProcess",
    "ServiceProcess": "ServiceProcess",
    "BackgroundProcess": "BackgroundProcess",
    "EmptyProcess": "EmptyProcess",
    "IndependentProcess": "IndependentProcess",
    "CooperatingProcess": "CooperatingProcess",
    "ImportanceHierarchyOfProcesses": "ProcessImportanceHierarchy",
    # POSIX shared memory API full names
    "PortableOperatingSystemInterfaceSharedMemoryApplicationProgrammingInterface": "PortableOperatingSystemInterfaceSharedMemoryApplicationProgrammingInterface",
    "PortableOperatingSystemInterfaceSharedMemoryApi": "PortableOperatingSystemInterfaceSharedMemoryApplicationProgrammingInterface",
}

# Word-level canonicalization used by the segmenter. It fixes collapsed names such as
# Waitforsingleobject -> WaitForSingleObject even when no spaces or uppercase boundaries exist.
WORD_CANON: Dict[str, str] = {
    "a": "A", "an": "An", "and": "And", "api": "ApplicationProgrammingInterface",
    "application": "Application", "programming": "Programming", "interface": "Interface",
    "advanced": "Advanced", "local": "Local", "procedure": "Procedure", "call": "Call",
    "address": "Address", "space": "Space", "system": "System", "operating": "Operating",
    "windows": "Windows", "unix": "Unix", "linux": "Linux", "android": "Android", "ios": "Ios",
    "portable": "Portable", "posix": "PortableOperatingSystemInterface",
    "process": "Process", "processes": "Process", "program": "Program", "execution": "Execution",
    "flow": "Flow", "concurrent": "Concurrent", "parent": "Parent", "child": "Child",
    "children": "Children", "termination": "Termination", "waiting": "Waiting", "for": "For",
    "single": "Single", "object": "Object", "function": "Function", "create": "Create",
    "fork": "Fork", "exec": "Exec", "execlp": "Execlp", "exit": "Exit", "wait": "Wait",
    "zero": "Zero", "memory": "Memory", "startup": "Startup", "info": "Information",
    "information": "Information", "structure": "Structure", "identifier": "Identifier",
    "management": "Management", "command": "Command", "resource": "Resource", "resources": "Resources",
    "sharing": "Sharing", "partitioning": "Partitioning", "deallocation": "Deallocation",
    "cascading": "Cascading", "status": "Status", "state": "State", "table": "Table",
    "tree": "Tree", "v": "V", "init": "Init", "systemd": "Systemd", "model": "Model",
    "computation": "Computation", "input": "Input", "output": "Output", "io": "InputOutput",
    "disk": "Disk", "network": "Network", "web": "Web", "browser": "Browser", "chrome": "Chrome",
    "renderer": "Renderer", "renderers": "Renderer", "plugin": "Plugin", "plug": "Plug", "in": "In",
    "sandbox": "Sandbox", "multiprocess": "Multiprocess", "architecture": "Architecture",
    "interprocess": "Interprocess", "communication": "Communication", "shared": "Shared",
    "message": "Message", "passing": "Passing", "mechanism": "Mechanism", "method": "Method",
    "systems": "System", "region": "Region", "segment": "Segment", "producer": "Producer",
    "consumer": "Consumer", "buffer": "Buffer", "bounded": "Bounded", "unbounded": "Unbounded",
    "capacity": "Capacity", "zero": "Zero", "mailbox": "Mailbox", "mailboxes": "Mailbox",
    "queue": "Queue", "send": "Send", "receive": "Receive", "blocking": "Blocking",
    "nonblocking": "Nonblocking", "synchronous": "Synchronous", "asynchronous": "Asynchronous",
    "direct": "Direct", "indirect": "Indirect", "addressing": "Addressing", "symmetric": "Symmetric",
    "asymmetric": "Asymmetric", "owned": "Owned", "owner": "Owner", "mach": "Mach",
    "port": "Port", "rights": "Rights", "right": "Right", "header": "Header", "simple": "Simple",
    "complex": "Complex", "kernel": "Kernel", "virtual": "Virtual", "file": "File", "truncate": "Truncate",
    "map": "Map", "mapped": "Mapped", "open": "Open", "unlink": "Unlink", "mmap": "MemoryMap",
    "shm": "SharedMemory", "socket": "Socket", "sockets": "Socket", "pipe": "Pipe", "pipes": "Pipe",
    "thread": "Thread", "threads": "Thread", "sender": "Sender", "receiver": "Receiver",
    "item": "Item", "fixed": "Fixed", "sized": "Sized", "variable": "Variable",
    "at": "At", "most": "Most", "n": "N", "milliseconds": "Milliseconds",
}

# Prefer longer words during segmentation.
_SEG_WORDS = sorted(WORD_CANON.keys(), key=len, reverse=True)

# 用於英文 token 內的縮寫展開。只有完整 token 才會換，不會動 BigO 這種已成詞結果。
# Conservative equivalence map. Only merge terms that are truly the same concept.
# Do NOT merge near-synonyms that have different practical meanings.
# Example: ProcessID, ProcessIdentifier, and ProcessIdentification are related,
# but they are intentionally kept as separate nodes.
SEMANTIC_EQUIVALENCE_MAP: Dict[str, str] = {
    # PID family: exact spelling normalization only. Keep related concepts separate.
    "PID": "ProcessID",
    "Pid": "ProcessID",
    "pid": "ProcessID",
    "pid_t": "ProcessID",
    "Process ID": "ProcessID",
    "Process Id": "ProcessID",
    "process ID": "ProcessID",
    "process id": "ProcessID",
    "ProcessID": "ProcessID",
    "ProcessId": "ProcessID",
    "ProcessIdentifier": "ProcessIdentifier",
    "Process Identifier": "ProcessIdentifier",
    "process identifier": "ProcessIdentifier",
    "ProcessIdentification": "ProcessIdentification",
    "Process Identification": "ProcessIdentification",
    "process identification": "ProcessIdentification",
    "ProcessIdentificationNumber": "ProcessID",

    # Initial/init process variants in OS texts
    "InitProcess": "InitialProcess",
    "Init Process": "InitialProcess",
    "init process": "InitialProcess",
    "InitialProcess": "InitialProcess",
    "Initial Process": "InitialProcess",
    "initial process": "InitialProcess",

    # API function name variants
    "CreateProcess": "CreateProcessFunction",
    "CreateProcess()": "CreateProcessFunction",
    "CreateProcess Function": "CreateProcessFunction",
    "CreateProcessFunction": "CreateProcessFunction",
    "WaitForSingleObjectFunction": "WaitForSingleObject",
    "WaitForSingleObject": "WaitForSingleObject",
    "WaitForSingleObject()": "WaitForSingleObject",
    "ZeroMemoryFunction": "ZeroMemory",
    "ZeroMemory": "ZeroMemory",
    "ZeroMemory()": "ZeroMemory",

    # Spelling variants
    "MailBox": "Mailbox",
    "Mailbox": "Mailbox",
    "Mailboxes": "Mailbox",
    "mailbox": "Mailbox",
    "mailboxes": "Mailbox",

    # Common IPC model spelling variants. Keep base concept / mechanism / system distinct.
    "Message Passing": "MessagePassing",
    "message passing": "MessagePassing",
    "Message Passing Mechanism": "MessagePassingMechanism",
    "Message Passing System": "MessagePassingSystem",
    "Shared Memory": "SharedMemory",
    "shared memory": "SharedMemory",
    "Shared Memory System": "SharedMemorySystem",

    # Symbol-sensitive technical names. These must stay distinct when generic
    # punctuation cleanup is applied.
    "B-tree": "BTree",
    "B-Tree": "BTree",
    "B-Trees": "BTree",
    "B tree": "BTree",
    "B Tree": "BTree",
    "B\u6a39": "BTree",
    "B+ tree": "BPlusTree",
    "B+ Tree": "BPlusTree",
    "B+-tree": "BPlusTree",
    "B+\u6a39": "BPlusTree",
    "B*-tree": "BStarTree",
    "B*-trees": "BStarTree",
    "B* tree": "BStarTree",
    "B*\u6a39": "BStarTree",
    "C++": "CPlusPlusProgrammingLanguage",
    "C++ Programming Language": "CPlusPlusProgrammingLanguage",
    "C++ \u7a0b\u5f0f\u8a9e\u8a00": "CPlusPlusProgrammingLanguage",
    "operator[]": "SubscriptOperator",
    "Index Operator": "SubscriptOperator",
    "\u7d22\u5f15\u904b\u7b97\u5b50": "SubscriptOperator",

    # Repeated acronym expansion artifacts found in legacy OS chapter JSON.
    "PortableOperatingSystemInterfaceRealTimeSchedulingApplicationProgrammingInterface":
        "RealTimeSchedulingApplicationProgrammingInterface",
    "TransmissionControlProtocolInternetProtocolTransmissionControlProtocolInternetProtocolSuite":
        "TransmissionControlProtocolInternetProtocolSuite",
    "TDIApplicationProgrammingInterfaceTransportDriverInterfaceApplicationProgrammingInterface":
        "TransportDriverInterfaceApplicationProgrammingInterface",
}

ABBR_TOKEN_MAP: Dict[str, str] = {
    "CPU": "Central Processing Unit",
    "GPU": "Graphics Processing Unit",
    "RAM": "Random Access Memory",
    "ROM": "Read Only Memory",
    "OS": "Operating System",
    "API": "Application Programming Interface",
    "SQL": "Structured Query Language",
    "DBMS": "Database Management System",
    "TCP": "Transmission Control Protocol",
    "UDP": "User Datagram Protocol",
    "IP": "Internet Protocol",
    "HTTP": "Hypertext Transfer Protocol",
    "HTTPS": "Hypertext Transfer Protocol Secure",
    "DNS": "Domain Name System",
    "I/O": "Input Output",
    "IO": "Input Output",
    "POSIX": "Portable Operating System Interface",
    "PID": "Process ID",
    "XML": "Extensible Markup Language",
    "HTML": "Hypertext Markup Language",
    "CSS": "Cascading Style Sheets",
    "URL": "Uniform Resource Locator",
    "URI": "Uniform Resource Identifier",
    "JSON": "JavaScript Object Notation",
}


def _clean_key(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def _has_cjk(s: str) -> bool:
    return bool(CJK_RE.search(s or ""))


def _english_score(s: str) -> int:
    return len(re.findall(r"[A-Za-z]", s or ""))


def _is_mostly_acronym(s: str) -> bool:
    raw = re.sub(r"[^A-Za-z]", "", s or "")
    return bool(raw) and len(raw) <= 6 and raw.upper() == raw




def _has_internal_pascal_caps(s: str) -> bool:
    return bool(re.search(r"[a-z0-9][A-Z]", s or ""))


def _looks_like_pascal_name(s: str) -> bool:
    """True for names that already look like a valid English PascalCase concept name."""
    s = str(s or "").strip()
    return bool(re.fullmatch(r"[A-Z][A-Za-z0-9]*", s)) and not _is_mostly_acronym(s)


# Generic semantic-role suffixes. These suffixes often change the nature of a concept,
# so a form normalizer must not merge them merely because their prefix is similar.
ROLE_SUFFIXES = (
    "ID", "Id", "Identifier", "Identification", "Number", "Code", "Type",
    "Function", "Operation", "Procedure", "Call", "Structure", "Object",
    "Mechanism", "System", "Model", "Method", "Technique", "Policy",
    "Protocol", "Interface", "Process", "State", "Status", "Table", "Queue",
    "Buffer", "Region", "Segment", "Address", "Space", "Analysis", "Management",
)


def _role_suffix(name: str) -> str:
    name = _to_pascal_case_no_semantic(name) if "_to_pascal_case_no_semantic" in globals() else str(name or "")
    for suffix in sorted(ROLE_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix) and len(name) > len(suffix):
            return suffix
    return ""


def _same_surface_form(a: str, b: str) -> bool:
    return re.sub(r"[^A-Za-z0-9]", "", a or "").lower() == re.sub(r"[^A-Za-z0-9]", "", b or "").lower()

def _split_pascal(s: str) -> str:
    s = str(s or "").strip()
    # Handle acronym-to-word boundaries, normal PascalCase boundaries, and digit boundaries.
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _segment_word(token: str) -> List[str]:
    """Segment collapsed English token into known technical words.

    Examples: Waitforsingleobject -> [Wait, For, Single, Object]
              Processinformationstructure -> [Process, Information, Structure]
    """
    if not token:
        return []
    # Keep already known/acronym tokens first.
    raw = token.strip()
    if raw in CANONICAL_CASE_MAP:
        return [CANONICAL_CASE_MAP[raw]]
    upper = raw.upper()
    if upper in ABBR_TOKEN_MAP:
        return ABBR_TOKEN_MAP[upper].split()

    # Split real PascalCase before trying lower-case segmentation.
    split = _split_pascal(raw)
    parts = split.split()
    if len(parts) > 1:
        out: List[str] = []
        for p in parts:
            out.extend(_segment_word(p))
        return out

    low = raw.lower()
    # Dynamic programming: maximize covered characters, then minimize number of segments.
    n = len(low)
    best: List[Optional[List[str]]] = [None] * (n + 1)
    best[0] = []
    for i in range(n):
        if best[i] is None:
            continue
        for w in _SEG_WORDS:
            if low.startswith(w, i):
                j = i + len(w)
                cand = best[i] + [WORD_CANON[w]]
                if best[j] is None or len(cand) < len(best[j]):
                    best[j] = cand
    if best[n] is not None:
        return best[n] or []

    # Fallback: normal title case.
    if raw.isupper() and len(raw) <= 6:
        return [raw]
    return [raw[:1].upper() + raw[1:].lower()]


def _normalize_piece(piece: str) -> List[str]:
    if not piece:
        return []
    if piece in CANONICAL_CASE_MAP:
        return [CANONICAL_CASE_MAP[piece]]
    if piece in DIRECT_NAME_MAP:
        return [DIRECT_NAME_MAP[piece]]
    up = re.sub(r"[^A-Za-z/]", "", piece).upper()
    if up in ABBR_TOKEN_MAP:
        return ABBR_TOKEN_MAP[up].split()
    return _segment_word(piece)


def _to_pascal_case(term: str) -> str:
    term = _clean_key(term)
    if not term:
        return term

    # Exact mappings first. Semantic equivalence has highest priority because
    # it intentionally merges names that are individually valid but synonymous.
    if term in SEMANTIC_EQUIVALENCE_MAP:
        return SEMANTIC_EQUIVALENCE_MAP[term]
    if term in CANONICAL_CASE_MAP:
        return CANONICAL_CASE_MAP[term]
    if term in DIRECT_NAME_MAP:
        return DIRECT_NAME_MAP[term]

    # Expand common syntax and normalize punctuation into token boundaries.
    term = term.replace("I/O", " Input Output ")
    term = term.replace("input/output", " Input Output ")
    term = term.replace("Input/output", " Input Output ")
    term = re.sub(r"\bBig[-\s]?Oh\b", "Big O", term, flags=re.I)
    term = re.sub(r"\bBig[-\s]?Omega\b", "Big Omega", term, flags=re.I)
    term = re.sub(r"\bBig[-\s]?Theta\b", "Big Theta", term, flags=re.I)

    # Preserve correct internal capitalization by splitting before WORD_RE tokenization.
    split_term = _split_pascal(term)

    tokens: List[str] = []
    for tok in WORD_RE.findall(split_term):
        tokens.extend(_normalize_piece(tok))
    if not tokens:
        return term

    fixed: List[str] = []
    for t in tokens:
        if not t:
            continue
        if t in CANONICAL_CASE_MAP.values() or t in DIRECT_NAME_MAP.values():
            fixed.append(t)
        elif t.isdigit():
            fixed.append(t)
        elif len(t) == 1 and t.isupper():
            fixed.append(t)
        else:
            fixed.append(t[:1].upper() + t[1:])
    out = "".join(fixed)
    out = CANONICAL_CASE_MAP.get(out, out)
    # Final conservative pass: merge exact equivalents only, not related concepts.
    out = DIRECT_NAME_MAP.get(out, out)
    return SEMANTIC_EQUIVALENCE_MAP.get(out, out)


def _candidate_terms(name: str, aliases: Iterable[Any]) -> List[str]:
    seen = set()
    terms: List[str] = []
    for x in [name, *(aliases or [])]:
        t = _clean_key(x)
        if t and t not in seen:
            terms.append(t)
            seen.add(t)
    return terms


def canonical_name(name: str, aliases: Optional[Iterable[Any]] = None) -> str:
    """Return canonical English PascalCase concept name when possible.

    Design goal:
    - Fix form problems globally: Chinese/abbreviation/spacing/casing.
    - Do NOT merge near terms merely because they share a prefix.
      Example pattern for all domains: XID, XIdentifier, XIdentification,
      XFunction, XMechanism, XSystem, XObject may be related but are not
      automatically the same concept.
    """
    original = _clean_key(name)
    terms = _candidate_terms(original, aliases or [])

    # 1) Original name has priority over aliases when it already names a valid concept.
    # This prevents aliases such as "Process ID" from collapsing a concept named
    # ProcessIdentifier or ProcessIdentification into ProcessID.
    if original:
        if original in SEMANTIC_EQUIVALENCE_MAP:
            return SEMANTIC_EQUIVALENCE_MAP[original]
        if original in CANONICAL_CASE_MAP:
            v = CANONICAL_CASE_MAP[original]
            return SEMANTIC_EQUIVALENCE_MAP.get(v, v)
        if original in DIRECT_NAME_MAP:
            v = DIRECT_NAME_MAP[original]
            return SEMANTIC_EQUIVALENCE_MAP.get(v, v)
        if _looks_like_pascal_name(original) and _has_internal_pascal_caps(original):
            # Already a reasonable PascalCase technical term.
            return original

    # 2) If original is broken casing such as Waitforsingleobject, an alias with
    # better PascalCase can repair it. This is form repair, not semantic merging.
    english_terms = [t for t in terms if _english_score(t) > 0 and not _has_cjk(t)]
    non_acronym_english = [t for t in english_terms if not _is_mostly_acronym(t)]
    if non_acronym_english:
        def candidate_score(x: str) -> Tuple[int, int, int, int, int]:
            normalized = _to_pascal_case(x)
            exact_form_of_original = 1 if original and _same_surface_form(x, original) else 0
            has_internal_caps = 1 if re.search(r"[a-z][A-Z]", x) else 0
            is_mapped = 1 if x in CANONICAL_CASE_MAP or normalized in CANONICAL_CASE_MAP.values() else 0
            word_count = len(_split_pascal(normalized).split())
            return (is_mapped, exact_form_of_original, has_internal_caps, word_count, _english_score(x))

        # Only let alias win when original is CJK/acronym/broken-lowercase or lacks English.
        original_needs_repair = (
            not original
            or _has_cjk(original)
            or _is_mostly_acronym(original)
            or (not _has_internal_pascal_caps(original) and _looks_like_pascal_name(original))
            or _english_score(original) == 0
        )
        if original_needs_repair:
            best = sorted(non_acronym_english, key=candidate_score, reverse=True)[0]
            return _to_pascal_case(best)

    # 3) Acronym-only fallback: expand if known.
    for t in english_terms:
        upper = re.sub(r"[^A-Za-z/]", "", t).upper()
        if upper in ABBR_TOKEN_MAP:
            return _to_pascal_case(ABBR_TOKEN_MAP[upper])

    # 4) If original is English/mixed, PascalCase it without guessing semantic equivalence.
    if _english_score(original) > 0 and not _has_cjk(original):
        return _to_pascal_case(original)

    # 5) Last fallback: keep original to avoid data loss.
    return original

def _record(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return deepcopy(obj)
    if hasattr(obj, "model_dump"):
        d = obj.model_dump()
        # Preserve private book_ids used by the original project when present.
        try:
            if "book_ids" not in d and getattr(obj, "__dict__", None):
                d["book_ids"] = list(obj.__dict__.get("_book_ids", []))
        except Exception:
            pass
        return d
    if hasattr(obj, "dict"):
        return obj.dict()
    raise TypeError(f"Unsupported record type: {type(obj)!r}")


def _unique_keep_order(items: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        s = _clean_key(x)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _alias_candidates(old_name: str, aliases: Iterable[Any], new_name: str) -> List[str]:
    spaced = _split_pascal(new_name)
    return _unique_keep_order([old_name, spaced, *(aliases or [])])


def _merge_concepts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(a)
    # Longer definition usually has more information.
    if len(str(b.get("definition") or "")) > len(str(merged.get("definition") or "")):
        merged["definition"] = b.get("definition")
    merged["aliases"] = _unique_keep_order([*(merged.get("aliases") or []), *(b.get("aliases") or [])])
    if not merged.get("category") or merged.get("category") == "其他":
        merged["category"] = b.get("category") or merged.get("category")
    if not merged.get("parent") and b.get("parent"):
        merged["parent"] = b.get("parent")
    merged["is_fine_grained"] = bool(merged.get("is_fine_grained")) or bool(b.get("is_fine_grained"))
    if "book_ids" in merged or "book_ids" in b:
        merged["book_ids"] = _unique_keep_order([*(merged.get("book_ids") or []), *(b.get("book_ids") or [])])
    return merged


def normalize_graph_json_records(
    concepts: Iterable[Any],
    subtype_edges: Iterable[Any],
    prereq_edges: Iterable[Any],
    refined_prereq_edges: Optional[Iterable[Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Normalize concepts and all edge endpoints.

    Returns: (concepts, subtype_edges, prereq_edges, refined_prereq_edges)
    """
    concept_dicts = [_record(c) for c in concepts]
    subtype_dicts = [_record(e) for e in subtype_edges]
    prereq_dicts = [_record(e) for e in prereq_edges]
    refined_dicts = [_record(e) for e in (refined_prereq_edges or [])]

    # Build canonical mapping from names and aliases to canonical concept name.
    name_map: Dict[str, str] = {}
    for c in concept_dicts:
        old = _clean_key(c.get("name"))
        aliases = c.get("aliases") or []
        new = canonical_name(old, aliases)
        for term in _candidate_terms(old, aliases):
            # 若 alias 本身是已知標準術語，不要被其他概念覆蓋。
            # 例：執行時間 aliases 含「時間複雜度」，但「時間複雜度」應保留為 TimeComplexity。
            if term in SEMANTIC_EQUIVALENCE_MAP:
                name_map[term] = SEMANTIC_EQUIVALENCE_MAP[term]
            elif term in CANONICAL_CASE_MAP:
                v = CANONICAL_CASE_MAP[term]
                name_map[term] = SEMANTIC_EQUIVALENCE_MAP.get(v, v)
            elif term in DIRECT_NAME_MAP:
                v = DIRECT_NAME_MAP[term]
                name_map[term] = SEMANTIC_EQUIVALENCE_MAP.get(v, v)
            else:
                name_map.setdefault(term, new)
        # 原始 name 一定代表此概念本身，可以覆蓋 setdefault 結果。
        new = SEMANTIC_EQUIVALENCE_MAP.get(new, new)
        name_map[old] = new
        name_map.setdefault(new, new)

    # Case-insensitive / collapsed-name canonical map. This merges names such as
    # Processinformationstructure and ProcessInformationStructure before Neo4j MERGE.
    collapsed_to_canonical: Dict[str, str] = {}
    for v in list(name_map.values()):
        key = re.sub(r"[^A-Za-z0-9]", "", v).lower()
        if not key:
            continue
        # Prefer names with more internal uppercase boundaries and longer spellings.
        old_v = collapsed_to_canonical.get(key)
        if old_v is None or (len(_split_pascal(v).split()), len(v)) > (len(_split_pascal(old_v).split()), len(old_v)):
            collapsed_to_canonical[key] = v
    for k, v in list(name_map.items()):
        collapsed = re.sub(r"[^A-Za-z0-9]", "", v).lower()
        if collapsed in collapsed_to_canonical:
            name_map[k] = collapsed_to_canonical[collapsed]

    def map_endpoint(x: Any) -> str:
        s = _clean_key(x)
        if not s:
            return s
        if s in name_map:
            return name_map[s]
        # For endpoints not in concepts, still canonicalize so future matching is more consistent.
        return canonical_name(s, [])

    # Normalize concept records.
    by_name: Dict[str, Dict[str, Any]] = {}
    for c in concept_dicts:
        old_name = _clean_key(c.get("name"))
        new_name = name_map.get(old_name) or canonical_name(old_name, c.get("aliases") or [])
        new_c = deepcopy(c)
        new_c["name"] = new_name
        new_c["aliases"] = _alias_candidates(old_name, c.get("aliases") or [], new_name)
        if new_c.get("parent"):
            new_c["parent"] = map_endpoint(new_c.get("parent"))
        if new_name in by_name:
            by_name[new_name] = _merge_concepts(by_name[new_name], new_c)
        else:
            by_name[new_name] = new_c

    normalized_concepts = list(by_name.values())

    # Normalize subtype edges.
    subtype_seen = set()
    normalized_subtypes: List[Dict[str, Any]] = []
    for e in subtype_dicts:
        parent = map_endpoint(e.get("parent"))
        child = map_endpoint(e.get("child"))
        if not parent or not child or parent == child:
            continue
        key = (parent, child)
        if key in subtype_seen:
            continue
        subtype_seen.add(key)
        new_e = deepcopy(e)
        new_e["parent"] = parent
        new_e["child"] = child
        normalized_subtypes.append(new_e)

    # Normalize prereq edges; keep highest confidence duplicate.
    def normalize_prereq_list(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for e in edges:
            prereq = map_endpoint(e.get("prereq") or e.get("a"))
            target = map_endpoint(e.get("target") or e.get("b"))
            if not prereq or not target or prereq == target:
                continue
            new_e = deepcopy(e)
            if "prereq" in new_e or "target" in new_e:
                new_e["prereq"] = prereq
                new_e["target"] = target
            else:
                new_e["a"] = prereq
                new_e["b"] = target
            key = (prereq, target)
            old = by_key.get(key)
            if old is None or float(new_e.get("confidence") or 0) > float(old.get("confidence") or 0):
                by_key[key] = new_e
        return list(by_key.values())

    normalized_prereqs = normalize_prereq_list(prereq_dicts)
    normalized_refined = normalize_prereq_list(refined_dicts)

    return normalized_concepts, normalized_subtypes, normalized_prereqs, normalized_refined
