"""Phase 0 Step 2: Gemini 把 seeds 建成 Knowledge Taxonomy

輸入: seeds.json (from taxonomy_extractor.py)
輸出: taxonomy.yaml (用戶會校對這份)

Taxonomy 結構 (以 OS 為例):
    Operating_Systems:
      Process_Management:
        Concept:
          - Process
          - Thread
          - PCB
        Algorithm:
          - CPU_Scheduling
          - Round_Robin
      Synchronization:
        Problem:
          - Critical_Section
          - Race_Condition
          - Deadlock
        Primitive:
          - Atomic_Variable
          - Test_And_Set
          - Compare_And_Swap
        Tool:
          - Mutex
          - Semaphore
          - Monitor
        Algorithm:
          - Peterson_Solution
          - Bakery_Algorithm

分類規則 (Gemini prompt 內明確定義):
    - Concept    : 抽象概念本身 (Process, Deadlock)
    - Problem    : 教材要解決的問題 (Critical Section Problem)
    - Primitive  : 底層機制 (atomic operations, memory barriers)
    - Tool       : 提供給使用者的工具/API (Semaphore, Mutex)
    - Algorithm  : 具體演算法 (Peterson Solution)
    - Metric     : 評估指標 (Throughput, Turnaround Time)
    - Model      : 理論模型 (Producer-Consumer, Reader-Writer)
    - Structure  : 資料結構 (PCB, Page Table)

用法:
    python -m kg_builder.taxonomy_builder \
        --seeds outputs/taxonomy/seeds.json \
        --output outputs/taxonomy/taxonomy.yaml \
        --domain "Operating Systems"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from config import settings
from utils.gemini_client import init_gemini


TAXONOMY_SYSTEM_PROMPT = """你是「教材知識工程」專家,任務是把候選詞彙整理成教科書知識分類 (Taxonomy)。

# 8 種分類 (Category)

- Concept    : 抽象概念本身  (例: Process, Deadlock, Virtual Memory)
- Problem    : 教材要解決的問題  (例: Critical Section Problem, Dining Philosophers)
- Primitive  : 底層機制、原語  (例: atomic operation, memory barrier, Test-and-Set)
- Tool       : 提供給使用者的工具/API  (例: Semaphore, Mutex, Monitor)
- Algorithm  : 具體演算法  (例: Peterson Solution, Banker's Algorithm, FCFS)
- Metric     : 評估指標  (例: Throughput, Turnaround Time, Response Time)
- Model      : 理論模型  (例: Producer-Consumer Model, Reader-Writer Model)
- Structure  : 資料結構  (例: PCB, Page Table, Inode)

# 判定原則
1. 只保留「教材真正教授的知識點」,以下**必須排除**:
   - 變數名 / 陣列 / 旗標  (turn, flag[], waiting[], lock_var)
   - 範例角色  (ProducerProcess, ConsumerThread, Thread1)
   - 實作細節  (KernelDataStructure, MemoryAllocationDataStructure)
   - 圖名 / 表名  (StructureOfProcessPi, Figure6_3)
   - 過度描述  (ProcessStructureForCriticalSection, AtomicVariableManipulationFunction)
   - 純資料型別  (Integer, Variable, Method)
   - 一次性名詞  (只在單一 chunk 出現)

2. Concept 命名採用**完整、通用**寫法,不用 PascalCase 拼湊:
   - Test-and-Set ✓  (不要 TestAndSetVariable)
   - Peterson's Solution ✓  (不要 StructureOfProcessPiInPetersonSolution)
   - Producer-Consumer Problem ✓  (不要 ProducerProcess + ConsumerProcess)

3. 若兩個 seed 意義相同,只留一個 canonical name,另一個記為 alias:
   - Semaphore  (canonical)
     alias: [SemaphoreVariable, SemaphoreS, WaitAndSignalPrimitive]

4. 每個 concept 記錄:
   - name           : Canonical name (英文)
   - category       : 上述 8 種之一
   - definition     : 1 句 (從 seed 的 context 摘出)
   - aliases        : 別名列表
   - source_chapters: 出現章節列表
   - importance     : Core / Supporting  (Core = 章節主題,Supporting = 章節內重要概念)

# 輸出格式 (YAML)

domain: Operating Systems
version: v1
subtrees:
  - name: Process_Management
    concepts:
      - name: Process
        category: Concept
        definition: A program in execution
        aliases: [ProcessAbstraction]
        source_chapters: ["3"]
        importance: Core
      - name: Thread
        category: Concept
        ...
  - name: Synchronization
    concepts:
      - name: Semaphore
        category: Tool
        definition: ...
        ...

# 目標
- 產出 100-200 個高品質 concept (不是越多越好)
- 每個 concept 都能被學生問「What is X?」並有教學意義
- 每個 concept 都能與其他 concept 建立 relation
"""


TAXONOMY_USER_TEMPLATE = """Domain: {domain}

以下是從 {n_seeds} 個候選詞彙 (Chapter Title / Section Title / Bold Term / Definition Pattern) 抽出的清單。
請你依照上面規則整理成 taxonomy YAML。

【候選詞彙】(格式: [priority] term - source - chapters - occurrence)

{seeds_list}

【要求】
1. 排除變數/角色/圖名/描述詞
2. 相似 concept 合併 (取 canonical + aliases)
3. 分成 subtree (以本教材章節結構為基礎)
4. 每個 concept 給 category + definition (1 句)
5. 只回傳 YAML,不要 markdown code fence,不要說明文字
"""


def _format_seeds_for_prompt(seeds: List[Dict[str, Any]], limit: int = 500) -> str:
    """把 seed 列成 prompt 用的清單 (限 top-N)"""
    lines = []
    for s in seeds[:limit]:
        term = s.get("term", "")
        pri = s.get("priority", 0)
        src = ",".join(s.get("source_summary", []))
        chs = ",".join(s.get("chapters", []))
        occ = s.get("occurrence", 1)
        lines.append(f"[{pri}] {term} - {src} - ch{chs} - x{occ}")
    return "\n".join(lines)


def _extract_yaml(text: str) -> Optional[Dict[str, Any]]:
    """從 Gemini response 抽 YAML (兼容 markdown fence)"""
    if not text:
        return None
    m = re.search(r"```(?:yaml|yml)?\s*(.+?)\s*```", text, re.S)
    if m:
        text = m.group(1)
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as e:
        print(f"[builder] YAML parse 失敗: {e}", file=sys.stderr)
        # 存 raw 供除錯
        Path("outputs/taxonomy/_raw_response.txt").write_text(
            text, encoding="utf-8"
        )
        return None


def build_taxonomy(
    seeds: List[Dict[str, Any]],
    domain: str,
    model_name: Optional[str] = None,
    max_seeds: int = 500,
) -> Optional[Dict[str, Any]]:
    model = init_gemini(
        model_name=model_name or settings.gemini_model,
        generation_config={"temperature": 0.1},
    )
    prompt = TAXONOMY_SYSTEM_PROMPT + "\n\n" + TAXONOMY_USER_TEMPLATE.format(
        domain=domain,
        n_seeds=len(seeds),
        seeds_list=_format_seeds_for_prompt(seeds, limit=max_seeds),
    )
    print(f"[builder] 呼叫 Gemini ({model_name or settings.gemini_model}) ...")
    print(f"[builder] Prompt 長度: {len(prompt)} chars, seeds: {min(len(seeds), max_seeds)}")

    resp = model.generate_content(prompt)
    text = str(getattr(resp, "text", "") or "")
    print(f"[builder] 回應長度: {len(text)} chars")

    parsed = _extract_yaml(text)
    return parsed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", "-s", required=True)
    p.add_argument("--output", "-o", default="outputs/taxonomy/taxonomy.yaml")
    p.add_argument("--domain", "-d", default="Operating Systems")
    p.add_argument("--model", default=None, help="Gemini model (預設用 settings)")
    p.add_argument("--max-seeds", type=int, default=500)
    args = p.parse_args()

    seeds = json.loads(Path(args.seeds).read_text(encoding="utf-8"))
    print(f"[builder] 載入 {len(seeds)} seeds")

    taxonomy = build_taxonomy(
        seeds, args.domain, model_name=args.model, max_seeds=args.max_seeds
    )
    if not taxonomy:
        print("❌ Taxonomy 建構失敗 (YAML 解析)")
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(taxonomy, allow_unicode=True, sort_keys=False, indent=2),
        encoding="utf-8",
    )

    # 統計
    subtrees = taxonomy.get("subtrees") or []
    total = 0
    cat_counter: Dict[str, int] = {}
    for st in subtrees:
        for c in (st.get("concepts") or []):
            total += 1
            cat = c.get("category", "Unknown")
            cat_counter[cat] = cat_counter.get(cat, 0) + 1

    print()
    print("=" * 50)
    print(f"✅ Taxonomy 寫出: {out}")
    print(f"  Subtrees: {len(subtrees)}")
    print(f"  Concepts: {total}")
    print()
    print("  Category 分佈:")
    for cat, cnt in sorted(cat_counter.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {cnt}")
    print()
    print("👉 下一步: 你**手動校對** taxonomy.yaml (增/刪/改 concept)")
    print("   校對完後跑: python build_taxonomy_kg.py --taxonomy taxonomy.yaml ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
