"""Phase 0 Step 1: 從 Silberschatz OS 教材抽 seed concepts

來源:
    1. Chapter Titles + Section Headings (每章目錄)
    2. Bold terms in body text (視為 defined terms)
    3. Glossary (若存在)
    4. Index (若存在, 抓 leaf entries)

輸出: outputs/taxonomy/seeds.json
    [
      {
        "term": "Semaphore",
        "source": "chapter_title|section|bold|glossary|index",
        "chapter": "6",
        "page": 260,
        "context": "A semaphore S is an integer variable that ..."
      },
      ...
    ]

用法:
    python -m kg_builder.taxonomy_extractor \
        --pdf-dir "outputs/真轉好了/作業系統/原黨" \
        --output outputs/taxonomy/seeds.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import fitz  # PyMuPDF


# 教材編號規則: 章節標題 (Ch6 = Chapter 6)
CHAPTER_TITLE_RE = re.compile(
    r"^(?:Chapter|CHAPTER)\s+(\d+)\s*[:：\.]?\s*(.+?)$",
    re.M,
)
# 節標題 (6.1, 6.1.2 等)
SECTION_TITLE_RE = re.compile(
    r"^\s*(\d+\.\d+(?:\.\d+)?)\s+([A-Z][A-Za-z\s\-,/&\(\)']{3,80})$",
    re.M,
)
# Definition 樣式 "X is defined as", "X refers to", "The X is"
DEFINITION_RE = re.compile(
    r"\b(?:The\s+)?([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,4})\s+(?:is|are|refers?\s+to|means|denotes?)\s+(?:defined\s+as\s+|a\s+|an\s+|the\s+)",
)
# Common OS/CS terms (initial term dictionary — 讓 Gemini 之後 refine)
# 不用來抽,只用來 boost/reject 明顯正確或明顯錯誤的 term
STOP_TERMS: Set[str] = {
    "chapter", "section", "figure", "table", "example", "exercise",
    "the", "this", "that", "these", "those", "there", "here",
    "note", "notes", "practice", "summary", "review", "questions",
    "exercises", "further", "reading", "programming", "problems",
    "bibliographical", "bibliography", "references", "index",
    "contents", "preface", "acknowledgments", "appendix",
    "figure caption", "as shown", "shown in", "see also",
    "next chapter", "last chapter",
}


def _clean_title(t: str) -> str:
    """去掉頁尾/章序號等"""
    t = re.sub(r"\s+", " ", t).strip()
    # 去掉尾巴 "... 123" 頁碼
    t = re.sub(r"\s+\d{1,4}\s*$", "", t)
    # 去掉開頭 "Chapter 6" (只留主標題)
    t = re.sub(r"^(?:Chapter|CHAPTER)\s+\d+\s*[:：\.]?\s*", "", t)
    return t.strip()


def _looks_like_concept(term: str) -> bool:
    """粗略過濾: 判斷是否可能是 concept"""
    if not term or len(term) < 3 or len(term) > 60:
        return False
    lower = term.lower().strip()
    if lower in STOP_TERMS:
        return False
    # 含頁碼、圖號、章號
    if re.search(r"\b(?:page|figure|table|section|chapter)\s*\d", lower):
        return False
    # 全數字
    if re.match(r"^[\d\.\s\-]+$", term):
        return False
    # 至少一個字母
    if not re.search(r"[A-Za-z]", term):
        return False
    # 開頭大寫 (定義項通常大寫)
    if not term[0].isupper():
        return False
    return True


def extract_seeds_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """從單一 PDF 抽 seed concepts"""
    doc = fitz.open(pdf_path)
    seeds: List[Dict[str, Any]] = []
    # 從檔名抓章號
    m = re.search(r"Chapter[_\s]+(\d+)[_\s]+(.+?)\.pdf", pdf_path.name)
    chapter_num = m.group(1) if m else "?"
    chapter_topic = _clean_title(m.group(2).replace("_", " ")) if m else pdf_path.stem

    # 章名本身就是最重要的 seed
    if chapter_topic and _looks_like_concept(chapter_topic):
        seeds.append({
            "term": chapter_topic,
            "source": "chapter_title",
            "chapter": chapter_num,
            "page": 1,
            "context": f"Chapter {chapter_num}: {chapter_topic}",
            "priority": 10,  # 最高優先
        })

    for page_idx, page in enumerate(doc):
        page_no = page_idx + 1
        text = page.get_text()
        if not text:
            continue

        # 節標題 (X.Y Title)
        for m in SECTION_TITLE_RE.finditer(text):
            sec_num = m.group(1)
            title = _clean_title(m.group(2))
            if _looks_like_concept(title):
                seeds.append({
                    "term": title,
                    "source": "section_title",
                    "chapter": chapter_num,
                    "section": sec_num,
                    "page": page_no,
                    "context": f"{sec_num} {title}",
                    "priority": 8,
                })

        # Definition 樣式 "X is defined as", "X refers to"
        for m in DEFINITION_RE.finditer(text):
            term = _clean_title(m.group(1))
            if _looks_like_concept(term):
                # 抓後面 200 字當 context
                start = max(0, m.start() - 20)
                end = min(len(text), m.end() + 200)
                ctx = text[start:end].strip().replace("\n", " ")[:300]
                seeds.append({
                    "term": term,
                    "source": "definition_pattern",
                    "chapter": chapter_num,
                    "page": page_no,
                    "context": ctx,
                    "priority": 6,
                })

        # Bold terms via font-size analysis (定義項通常粗體)
        blocks = page.get_text("dict").get("blocks", [])
        for b in blocks:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    text_span = span.get("text", "").strip()
                    flags = span.get("flags", 0)
                    # flags & 16 = bold
                    if not (flags & 16):
                        continue
                    if not _looks_like_concept(text_span):
                        continue
                    # 排除太長句子 (只留單詞或短片語)
                    if len(text_span.split()) > 5:
                        continue
                    seeds.append({
                        "term": text_span,
                        "source": "bold",
                        "chapter": chapter_num,
                        "page": page_no,
                        "context": text_span,
                        "priority": 4,
                    })

    doc.close()
    return seeds


def dedupe_and_rank(all_seeds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """去重: 同一 term 合併,取最高 priority; 累計出現次數"""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in all_seeds:
        key = s["term"].strip().lower()
        grouped[key].append(s)

    result = []
    for key, items in grouped.items():
        # 用第一個 priority 最高的當代表
        items.sort(key=lambda x: -x.get("priority", 0))
        rep = dict(items[0])
        rep["occurrence"] = len(items)
        rep["source_summary"] = list({i["source"] for i in items})
        rep["chapters"] = sorted({i.get("chapter") for i in items if i.get("chapter")})
        # 若在多章出現 → 提高 priority (cross-chunk reuse rule)
        if len(rep["chapters"]) >= 2:
            rep["priority"] = rep.get("priority", 0) + 2
        result.append(rep)
    # 排 priority
    result.sort(key=lambda x: (-x.get("priority", 0), -x.get("occurrence", 0)))
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf-dir", "-d", required=True, help="教材 PDF 資料夾")
    p.add_argument(
        "--pattern",
        default="*Chapter*.pdf",
        help="PDF 檔名 pattern (default: *Chapter*.pdf)",
    )
    p.add_argument("--output", "-o", default="outputs/taxonomy/seeds.json")
    p.add_argument(
        "--min-priority",
        type=int,
        default=4,
        help="輸出時最低 priority (預設 4, bold 以上)",
    )
    args = p.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"❌ 資料夾不存在: {pdf_dir}", file=sys.stderr)
        return 1

    pdfs = sorted(pdf_dir.glob(args.pattern))
    print(f"[extractor] 掃描 {pdf_dir}, pattern={args.pattern}, 找到 {len(pdfs)} 個 PDF")
    if not pdfs:
        return 1

    all_seeds: List[Dict[str, Any]] = []
    for i, pdf in enumerate(pdfs, 1):
        print(f"  [{i}/{len(pdfs)}] {pdf.name}", end=" ... ", flush=True)
        try:
            seeds = extract_seeds_from_pdf(pdf)
            all_seeds.extend(seeds)
            print(f"{len(seeds)} raw seeds")
        except Exception as e:
            print(f"❌ {e}")

    print(f"\n[extractor] 原始 {len(all_seeds)} → 去重中 ...")
    deduped = dedupe_and_rank(all_seeds)
    print(f"[extractor] 去重後: {len(deduped)}")

    # 過濾 min-priority
    filtered = [s for s in deduped if s.get("priority", 0) >= args.min_priority]
    print(f"[extractor] priority >= {args.min_priority}: {len(filtered)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 統計
    print("\n=== 統計 ===")
    src_count: Counter = Counter()
    for s in filtered:
        for src in s.get("source_summary", []):
            src_count[src] += 1
    for src, cnt in src_count.most_common():
        print(f"  {src}: {cnt}")

    print(f"\n=== Top 20 seeds ===")
    for s in filtered[:20]:
        print(
            f"  [{s.get('priority')}] {s['term']:40s} "
            f"ch={s.get('chapters')} occ={s.get('occurrence')}"
        )

    print(f"\n✅ 寫出: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
