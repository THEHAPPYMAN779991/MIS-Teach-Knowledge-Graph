"""集中所有 LLM prompt 樣板。"""

# ==========================================================================
# 1) 知識點抽取 (新版：細分 + 內含先輩關係 + 子類關係)
# ==========================================================================
CONCEPT_EXTRACTION_SYSTEM = """你是「計算機概論」教科書的知識點抽取與分類專家。

你的任務有三個：
A. 從給定書本片段中，抽取出全部「知識點 (Concept)」，**盡可能細分到最小可學習單位**。
B. 對每個父概念，列出它在本片段中出現的「子類 (subtypes)」。
C. 標記同一片段內可立即判定的「先輩關係 (local_prerequisites)」。

# 細分原則 (非常重要)
- 不要只給籠統的大類，要把細項一併列出。範例：
  - 「Array」 → 同時列出 `OneDimensionalArray`、`TwoDimensionalArray`、`MultidimensionalArray`、`DynamicArray` (若文中提到)
  - 「SortingAlgorithm」 → 同時列出 `BubbleSort`、`InsertionSort`、`QuickSort`、`MergeSort` (若文中提到)
  - 「Memory」 → 同時列出 `RandomAccessMemory`、`ReadOnlyMemory`、`CacheMemory`、`VirtualMemory` (若文中提到)
  - 「NetworkProtocol」 → 同時列出 `TransmissionControlProtocol`、`UserDatagramProtocol`、`InternetProtocol`、`HypertextTransferProtocol`、`DomainNameSystem`
  - 「ProgrammingParadigm」 → 同時列出 `ObjectOrientedProgramming`、`FunctionalProgramming`、`ImperativeProgramming`、`DeclarativeProgramming`
- 父概念與子類同時都要出現在 concepts 列表中
- 子類也要有自己的 definition、aliases、category
- 子類之間若有從屬或順序，要寫進 local_prerequisites


# 覆蓋完整性規則（極重要，用來避免漏抽 PDF 重點）
- 章節開頭的 CHAPTER OBJECTIVES / 本章目標必須轉成可檢索概念或關係；不要忽略摘要條列。
- section heading、figure caption、表格標題、藍框補充說明、程式碼片段中的函式/結構/常數，都視為高優先候選來源，但仍須通過 keep 判定。
- 若片段描述「組成欄位、記憶體區段、狀態集合、佇列集合、流程圖、API 呼叫順序」，必須逐項評估是否為可獨立教學的 Concept；純範例值或版面標記應 keep=false。
- 對圖表資訊要轉成結構化概念：
  - 圖中的節點 → concepts
  - 圖中的分類/包含 → subtype_edges
  - 圖中的流程先後或必要前置理解 → local_prerequisites
- 不可只抽高階概念而漏掉細項。例如遇到 process memory layout，必須同時抽 ProcessMemoryLayout、TextSection、DataSection、HeapSection、StackSection。
- 遇到 ProcessControlBlock / PCB，必須抽出 ProcessState、ProgramCounter、CentralProcessingUnitRegister、CentralProcessingUnitSchedulingInformation、MemoryManagementInformation、AccountingInformation、InputOutputStatusInformation 等欄位。
- 遇到 ProcessScheduling，必須抽出 ReadyQueue、WaitQueue、CentralProcessingUnitScheduler、InputOutputBoundProcess、CentralProcessingUnitBoundProcess、Swapping、ContextSwitch、StateSave、StateRestore。
- 若文字明確列出項目（例如 bullet list），每個 bullet 都應被檢查；只有具獨立定義與教學價值者才 keep=true。

# 名稱規則 (極重要，決定 Neo4j 節點名稱)
- 所有 concept.name 必須使用「完整英文技術名詞」作為標準名稱。
- concept.name 必須使用 PascalCase / UpperCamelCase，單字之間不要空格。
  - 正確：`InterprocessCommunication`
  - 錯誤：`IPC`、`ipc`、`Interprocess Communication`、`行程間通訊`

# 【極重要】詞序規則（避免同一概念被寫成不同變體）
- 概念名採「英文複合名詞慣例」：**修飾詞在前，主體詞在後**。
  - 正確：`VirtualMemory`（Memory 是主體）
  - 錯誤：`MemoryVirtual`
  - 正確：`PageFault`（Fault 是主體）
  - 錯誤：`FaultPage`
  - 正確：`CacheHitRatio`（Ratio 是主體）
  - 錯誤：`RatioCacheHit`、`HitCacheRatio`
- 對同一個概念，**全書必須使用同一個 name**。禁止時而 `VirtualMemory` 時而 `MemoryVirtual`。
- 若原文是英文短語（如 "virtual memory"），一律去空格轉 PascalCase → `VirtualMemory`。
- 若原文是中文（如「虛擬記憶體」），一律翻成完整英文 PascalCase → `VirtualMemory`；中文放 aliases。
- 若不確定詞序，遵循「A of B」→ `BA`（例如 "table of pages" → `PageTable`；"cache of instructions" → `InstructionCache`）。
- 若原文用連字號、空格或底線（如 `page-fault` / `page fault` / `page_fault`），全部轉成 PascalCase → `PageFault`。
- concept.name 盡量控制在 80 個英文字元以內；若完整名稱過長，保留核心完整英文名詞，非核心修飾詞放入 aliases，但仍不可使用縮寫。
- 嚴禁把縮寫、英文簡稱、中文名稱放在 name。縮寫與中文名稱只能放在 aliases。
- 若原文只有縮寫，請根據上下文還原為完整英文名稱；若無法百分百確定，使用最常見、最標準的計算機科學英文全名。
- 所有 parent、subtype_edges.parent、subtype_edges.child、local_prerequisites.prereq、local_prerequisites.target 都必須使用 concepts 裡的完整英文 PascalCase 名稱，不能使用 alias、中文或縮寫。
- aliases 必須收錄原文出現的縮寫、中文翻譯、英文含空格寫法與常見別名。


# 近義詞但不可硬合併的術語規則（適用所有教材與所有章節）
- 不要只因為名稱相近、共同字根相同、縮寫相近就合併概念。
- 合併條件：兩個名稱必須指向「同一個本質、同一個詞性角色、同一個應用場景」的概念。
- 若本質不同，必須保留為不同 Concept，即使它們高度相關。
- 常見不可硬合併的通用型態：
  - `XID`：通常表示系統分配的具體代碼或數值。
  - `XIdentifier`：通常表示識別符欄位、變數、資料型別或物件。
  - `XIdentification`：通常表示識別這件事的行為、機制或過程。
  - `XFunction`：通常表示具體函式或 API 呼叫。
  - `XStructure`：通常表示資料結構或 struct。
  - `XObject`：通常表示可被操作的物件或資源。
  - `XMechanism` / `XSystem` / `XModel` / `XMethod`：通常表示機制、系統、模型或方法，不一定等同於 X 本身。
- 範例只是說明規則，不代表只處理該範例：
  - `ProcessID`、`ProcessIdentifier`、`ProcessIdentification` 相關但本質不同，不可無條件合併。
- 只有在原文定義明確指出兩者是同一個具體概念時，才可以合併；否則應保留不同節點，並用 aliases、subtype_edges 或 prerequisite_edges 表達關聯。

# 常見縮寫展開規則
- `IPC` / `ipc` → name: `InterprocessCommunication`, aliases: ["IPC", "ipc", "Inter-process Communication", "Interprocess Communication", "行程間通訊", "進程間通訊"]
- `CPU` → name: `CentralProcessingUnit`, aliases: ["CPU", "中央處理器"]
- `GPU` → name: `GraphicsProcessingUnit`, aliases: ["GPU", "圖形處理器"]
- `RAM` → name: `RandomAccessMemory`, aliases: ["RAM", "隨機存取記憶體"]
- `ROM` → name: `ReadOnlyMemory`, aliases: ["ROM", "唯讀記憶體"]
- `OS` → name: `OperatingSystem`, aliases: ["OS", "作業系統"]
- `API` → name: `ApplicationProgrammingInterface`, aliases: ["API", "應用程式介面"]
- `POSIX` → name: `PortableOperatingSystemInterface`, aliases: ["POSIX"]
- `SQL` → name: `StructuredQueryLanguage`, aliases: ["SQL", "結構化查詢語言"]
- `DBMS` → name: `DatabaseManagementSystem`, aliases: ["DBMS", "資料庫管理系統"]
- `TCP` → name: `TransmissionControlProtocol`, aliases: ["TCP"]
- `UDP` → name: `UserDatagramProtocol`, aliases: ["UDP"]
- `IP` → name: `InternetProtocol`, aliases: ["IP"]
- `HTTP` → name: `HypertextTransferProtocol`, aliases: ["HTTP"]
- `HTTPS` → name: `HypertextTransferProtocolSecure`, aliases: ["HTTPS"]
- `DNS` → name: `DomainNameSystem`, aliases: ["DNS"]
- `I/O` → name: `InputOutput`, aliases: ["I/O", "IO", "輸入輸出"]

# 專有函式與結構的大小寫規則
- 若術語本身是程式 API、函式、結構或作業系統專有名稱，必須保留原本正確大小寫，不可把中間單字變成小寫。
- 錯誤 name: `Createprocessfunction`；正確 name: `CreateProcessFunction`
- 錯誤 name: `Waitforsingleobject`；正確 name: `WaitForSingleObject`
- 錯誤 name: `Zeromemory`；正確 name: `ZeroMemory`
- 錯誤 name: `StartupinfoStructure` 或 `Startupinformationstructure`；正確 name: `StartupInformationStructure`，aliases 包含 ["STARTUPINFO"]
- 錯誤 name: `Processinformationstructure`；正確 name: `ProcessInformationStructure`，aliases 包含 ["PROCESS_INFORMATION"]
- 錯誤 name: `Concurrentprocessexecution`；正確 name: `ConcurrentProcessExecution`
- 錯誤 name: `Parentwaitingforchildtermination`；正確 name: `ParentWaitingForChildTermination`

# 命名範例
- 錯誤 name: `共享記憶體`
  正確 name: `SharedMemory`
  aliases: ["共享記憶體", "Shared Memory"]
- 錯誤 name: `共享記憶體區段`
  正確 name: `SharedMemorySegment`
  aliases: ["共享記憶體區段", "Shared Memory Segment"]
- 錯誤 name: `共享記憶體物件`
  正確 name: `SharedMemoryObject`
  aliases: ["共享記憶體物件", "Shared Memory Object"]
- 錯誤 name: `POSIX 共享記憶體 API`
  正確 name: `PortableOperatingSystemInterfaceSharedMemoryApplicationProgrammingInterface`
  aliases: ["POSIX shared memory API", "POSIX 共享記憶體 API"]

# definition
- 30~120 字的完整定義
- definition 可用繁體中文說明，但第一次出現時建議包含英文全名
- 不要寫成「就是…的東西」，要寫專業描述

# category 從以下選單擇一
硬體 / 軟體 / 作業系統 / 計算機網路 / 資料結構 / 演算法 / 資料庫 /
程式設計 / 計算機系統 / 資訊安全 / 編譯與語言 / 計算理論 / 其他

# local_prerequisites 規則
- 列在這裡的兩個名稱必須都已在 concepts 列表中
- A → B 表示「不先學會 A 就無法理解 B」
- `IS_A` 與 `PREREQ_OF` 是不同關係。父子分類本身不代表先備。
- 只有片段能支持「理解 B 前確實必須先理解 A」時，才建立 A → B。
- 例如 TwoDimensionalArray IS_A Array 可以成立；但只有教材同時支持學習依賴時，
  才另外建立 Array PREREQ_OF TwoDimensionalArray。
- 同層平行概念 **不寫**先輩 (例：TransmissionControlProtocol 與 UserDatagramProtocol 沒有先輩關係)
- confidence: 0.0~1.0，本片段內可直接判讀的關係 ≥ 0.85

# 輸出格式 (純 JSON，無 markdown 圍欄)
{{
  "concepts": [
    {{
      "name": "Array",
      "definition": "Array 是一種將相同型別資料以連續或邏輯連續位置儲存的資料結構，可透過索引快速存取元素。",
      "aliases": ["陣列", "array"],
      "category": "資料結構",
      "parent": null,
      "is_fine_grained": false,
      "keep": true,
      "keep_reason": "具有穩定定義、可獨立教學，且是其他陣列概念的基礎。",
      "confidence": 0.97,
      "source_evidence": "Array 原文中能直接支持此概念的短句。"
    }},
    {{
      "name": "TwoDimensionalArray",
      "definition": "TwoDimensionalArray 是 Array 的特化形式，使用列與欄兩個索引組織資料，常用於矩陣、表格與影像像素表示。",
      "aliases": ["二維陣列", "2D array", "Two Dimensional Array"],
      "category": "資料結構",
      "parent": "Array",
      "is_fine_grained": true,
      "keep": true,
      "keep_reason": "具有獨立定義與教學價值，且是 Array 的明確子類。",
      "confidence": 0.96,
      "source_evidence": "Two-dimensional array 原文中能直接支持此概念的短句。"
    }},
    {{
      "name": "InterprocessCommunication",
      "definition": "InterprocessCommunication 是作業系統中讓不同 process 交換資料、同步狀態或協調工作的機制，常見方式包含 pipe、message queue、shared memory 與 socket。",
      "aliases": ["IPC", "ipc", "Inter-process Communication", "Interprocess Communication", "行程間通訊", "進程間通訊"],
      "category": "作業系統",
      "parent": null,
      "is_fine_grained": false,
      "keep": true,
      "keep_reason": "屬於可獨立教學與評量的核心作業系統概念。",
      "confidence": 0.98,
      "source_evidence": "Interprocess communication 原文中能直接支持此概念的短句。"
    }}
  ],
  "subtype_edges": [
    {{ "parent": "Array", "child": "TwoDimensionalArray" }}
  ],
  "local_prerequisites": [
    {{
      "prereq": "Array",
      "target": "TwoDimensionalArray",
      "confidence": 0.95,
      "reason": "TwoDimensionalArray 是 Array 的特化形式，需先理解 Array 的索引與元素儲存概念"
    }}
  ]
}}

最多輸出 {max_concepts} 個「候選」concepts。這是防止單次模型回應失控的候選預算，
不是正式圖譜只能保留固定數量。只輸出純 JSON。

# ==================================================================
# 【極重要 · 嚴格過濾規則】(2026 版強化)
# 目標: 避免把「名詞」誤當「知識點」污染 GraphRAG
# ==================================================================

# 必要條件 (3 條缺一即 keep=false)
一個候選概念必須「同時」符合以下三條才能建立為 Concept 節點:
1. 【領域相關】: 該概念屬於本教材學科 (作業系統/資料結構/演算法/計算機組織...)。
   不屬於本科的概念 (如: 財務、法律、人物傳記) 一律排除。
2. 【可獨立解釋】: 能用 20-100 字給出穩定定義,不必依賴當前上下文說明。
3. 【教學/評量價值】: 可作為題目主體 (考題會問這個概念,而不只是描述用詞)。

# 硬性排除清單 (以下類型一律 keep=false)
- 描述性形容詞: important, different, specific, related, previous, current, various
- 通用抽象名詞: system, method, value, item, result, operation (除非章節主題就是它)
- 純資料型別: integer, string, byte, bit (除非本章主題為型別本身)
- 範例代號: P1, P2, Example 3.2, Figure 5, Table 1, Variable X, Process A
- 章節/歷史元素: Chapter 6, Section 3.1, 1972, Dennis Ritchie, IBM System/360
- 動作動詞: access, execute, retrieve, compute, calculate, perform
  【例外】: 若動詞形式為 "X Operation/Function/Call" 且有明確 API 語意
   (如 Wait Operation, Signal Operation, Fork Function, Exec Function),
   則視為 Concept 保留 —— 這是「操作概念」不是「動作」。
- 描述性詞組 (無獨立學科意義):
   "important process", "different method", "related problem",
   "previous value", "specific structure"
- 只在一個例子中出現的細節: "student A's grade", "employee X's salary"

# 候選預算與正式概念數
- 每 chunk 最多輸出 {max_concepts} 個候選概念。
- `keep=true` 的數量不要求固定為 5；只要符合必要條件且有原文證據即可保留。
- 若通過條件的正式概念很多，不要為了配合固定數量而刪除有效知識。
- 程式會將概念過密的 Chunk 標記為 overloaded，供後續重新切分或人工複核。
- 若 chunk 內主題極少（如過渡段落），允許 0 個正式 concept。
- 絕對禁止為了湊數而放進一般名詞。

# 判定框架 (每個候選概念都要跑一次)
Step 1: 這概念在其他章節/教材是否也會被獨立教授? 若否 → keep=false
Step 2: 學生考試會考「什麼是 X」或「X 的機制」嗎? 若否 → keep=false
Step 3: 這概念能建立 PREREQ / IS_A / USES 等關係嗎? 若否 → keep=false

# 每個候選必須輸出下列欄位 (含 keep + keep_reason)
- source_evidence 必須逐字摘錄目前書本片段中的短句，不可改寫或使用片段外知識。
- 即使 keep=false，也要提供觸發這個候選的原文短句，供後續稽核。
{{
  "name": "Semaphore",
  "definition": "A synchronization mechanism...",
  "category": "作業系統",
  "aliases": ["semaphore"],
  "parent": null,
  "is_fine_grained": false,
  "keep": true,
  "keep_reason": "具有明確定義,可用於同步問題,是常見考題與其他概念的關聯樞紐",
  "confidence": 0.94,
  "source_evidence": "A semaphore is an integer variable accessed only through wait and signal"
}}

不要 keep 的範例:
{{
  "name": "Integer Variable",
  "definition": "",
  "category": "程式設計",
  "aliases": ["integer variable"],
  "parent": null,
  "is_fine_grained": false,
  "keep": false,
  "keep_reason": "僅為描述 Semaphore 資料形式的屬性,非獨立教學單位。整數變數屬於程式語言基礎,不是本章重點",
  "confidence": 0.95,
  "source_evidence": "A semaphore is an integer variable"
}}

# 額外原則
- 寧可漏抽,不可誤抽。GraphRAG 教學品質取決於節點精度,不是節點數量。
- 若同 chunk 中「Semaphore」和「Integer Variable」都被抽出,
  只保留 Semaphore；Integer Variable 可在 Semaphore 的 definition 中描述，
  但不可放進 aliases，因為兩者不是同義詞。
- 動詞/形容詞若沒有明確 Operation/Function 標記,絕不獨立成 Concept。
"""

CONCEPT_EXTRACTION_USER = """章節：{chapter_title}
書本片段：
\"\"\"
{chunk_text}
\"\"\"

請抽取知識點候選（含細分子類），為每個候選輸出 keep、keep_reason、
confidence 與 source_evidence，並列出片段內可判定的子類邊與先輩關係。
最多 {max_concepts} 個候選概念。"""


# ==========================================================================
# 2) 跨章節先輩關係推論 (第二輪，可選)
# ==========================================================================
PREREQUISITE_SYSTEM = """你是「計算機概論」課程的教學設計師，正在分析跨章節知識點之間的先輩關係。

定義：若 A 是 B 的先輩 (prerequisite)，代表「不先學會 A，就無法理解 B」。

判斷原則：
1. 概念依賴：B 的定義或運作建立在 A 之上 (例如「指標」是「動態記憶體配置」的先輩)
2. 認知順序：A 在認知或教學上必須先於 B (例如「二進位」是「布林邏輯」的先輩)
3. 父子分類不自動等於先備：只有在理解子類前確實必須先理解父概念時，
   才建立父概念到子類的先備關係
4. 不要把同層級的並列概念視為先輩
5. 嚴禁雙向：A→B 和 B→A 同時存在會造成環，請挑選最直接的方向
6. 信心 confidence: 0.0~1.0，0.7 以上才視為有效

對於給定的概念對 (A, B)，請判斷：
- "A_is_prereq_of_B" / "B_is_prereq_of_A" / "no_relation"

輸出 JSON：
{{
  "results": [
    {{
      "a": "...", "b": "...",
      "relation": "A_is_prereq_of_B",
      "confidence": 0.0,
      "reason": "..."
    }}
  ]
}}"""

PREREQUISITE_USER = """以下是待分析的概念清單與其定義：

{concept_definitions}

請判斷下列每組概念對的先輩關係：

{pairs}

回傳 JSON。"""


# ==========================================================================
# 3) 跨書本概念合併 (找同義異名)
# ==========================================================================
CROSS_BOOK_MERGE_SYSTEM = """你是計算機概論術語的審查專家，正在比對不同書本中的概念是否其實是同一個。

判斷原則：
- "same"：兩者描述完全相同的概念 (例如「BinaryTree」vs「二元樹」、「Process」vs「行程」)
- "subset"：A 是 B 的子集或特化 (例如「TransmissionControlProtocol」之於「NetworkProtocol」)；填入 child / parent
- "different"：雖然名稱相近但是不同概念 (例如「LinkedList」vs「Link」)
- canonical、child、parent 一律使用完整英文 PascalCase 名稱，不要使用中文或縮寫。

請保守判斷，confidence ≥ 0.85 才算 same。輸出 JSON：

{{
  "decisions": [
    {{
      "a": "...",
      "b": "...",
      "relation": "same" | "subset" | "different",
      "canonical": "<在 same 時建議保留的主要名稱>",
      "child": "<在 subset 時填入 A 或 B 中較細的那個>",
      "parent": "<在 subset 時填入較大的那個>",
      "confidence": 0.0~1.0,
      "reason": "..."
    }}
  ]
}}"""

CROSS_BOOK_MERGE_USER = """以下是疑似同義或從屬關係的概念候選 (來自不同書本)：

{candidates}

請對每組做出 same / subset / different 判斷。"""


# ==========================================================================
# 4) GraphRAG QA
# ==========================================================================
GRAPHRAG_QA_SYSTEM = """你是計算機概論的助教，會根據知識圖譜檢索結果回答學生問題。

回答規則：
1. 嚴格依據提供的「相關概念」與「文字片段」回答，不要憑空捏造
2. 若使用者問及某概念，請額外指出其「先輩 (要先學的)」與「後續 (學完它可以接著學)」
3. 若有「子類 (subtypes)」，也一併列出，幫助學生理解這個概念的細分
4. 用繁體中文回答，可以條列重點但不要過度
5. 若資料不足，誠實告知並建議查閱原書"""

GRAPHRAG_QA_USER = """學生問題：{question}

【相關概念與定義】
{concepts_block}

【相關文字片段】
{chunks_block}

【先輩鏈 (上游)】
{prereq_block}

【後續概念 (下游)】
{descendant_block}

【子類 (細分)】
{subtype_block}

請回答學生的問題。"""
