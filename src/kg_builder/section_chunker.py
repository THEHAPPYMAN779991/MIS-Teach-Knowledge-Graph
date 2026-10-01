"""kg_builder/section_chunker.py - section-aware, bounded PDF chunker.

This drop-in replacement keeps page provenance for overlap text, truncates
summary/exercise tails, and guarantees that normal chunks do not exceed the
requested character size.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import fitz

from kg_builder.kp_loader_v2 import KPRegistry


@dataclass
class SectionChunk:
    chunk_id: str
    text: str
    pages: List[int]
    sections: List[str]
    primary_section: str
    page_start: int
    page_end: int


@dataclass
class _Piece:
    text: str
    page: int
    overlap_only: bool = False


_EXCLUDE_MARKERS = [
    r"^\s*(?:\d+(?:\.\d+)*\s+)?(?:Chapter\s+)?Summary\s*$",
    r"^\s*Bibliographical\s+Notes\s*$",
    r"^\s*Further\s+Reading\s*$",
    r"^\s*Bibliography\s*$",
    r"^\s*(?:Practice\s+)?Exercises?\s*$",
    r"^\s*Programming\s+(?:Problems|Projects|Exercises)\s*$",
    r"^\s*Review\s+Questions?\s*$",
]
_EXCLUDE_RE = re.compile("|".join(_EXCLUDE_MARKERS), re.M | re.I)
_EXCLUDE_COMPACT_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*)?(?:chapter)?summary|"
    r"bibliographicalnotes|furtherreading|bibliography|"
    r"(?:practice)?exercises?|programming(?:problems|projects|exercises)|"
    r"reviewquestions?$",
    re.I,
)
_NUMBERED_SECTION_HEADING_RE = re.compile(
    r"(?m)^\s*(?P<section>\d+\.\d+(?:\.\d+)*)\s*"
    r"(?:\n+\s*|[ \t]+)(?P<title>[A-Za-z][^\n]{0,160})$"
)


def _block_text(block: Dict[str, object]) -> str:
    """Return text from a PyMuPDF text block without changing reading order."""
    parts: List[str] = []
    for line in block.get("lines", []) or []:  # type: ignore[union-attr]
        for span in line.get("spans", []) or []:  # type: ignore[union-attr]
            parts.append(str(span.get("text", "")))
    return "".join(parts).strip()


def _is_exclusion_heading(value: str) -> bool:
    # Page numbers in running heads are intentionally ignored.  The resulting
    # compact text still has to be an entire heading, not a body mention.
    compact = re.sub(r"\s+", "", value or "")
    compact = re.sub(r"\d+$", "", compact)
    return bool(_EXCLUDE_COMPACT_RE.fullmatch(compact))


def _page_exclusion_marker(page: fitz.Page, text: str) -> re.Match[str] | None:
    """Find a real end-matter title without mistaking a running head for it.

    A running ``3.9 Summary`` header can appear at the top of a page that still
    contains the final body paragraphs.  Text extraction alone cannot
    distinguish it from the actual title, so page geometry is used as a second
    signal.  If no title-shaped block is found, a top-only marker is ignored.
    """
    matches = list(_EXCLUDE_RE.finditer(text))
    if not matches:
        return None

    page_height = float(page.rect.height or 1.0)
    title_blocks = []
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        value = _block_text(block)
        if not _is_exclusion_heading(value):
            continue
        sizes = [
            float(span.get("size", 0) or 0)
            for line in block.get("lines", []) or []
            for span in line.get("spans", []) or []
        ]
        y0 = float((block.get("bbox") or (0, 0, 0, 0))[1])
        # A body title is normally larger than running text or appears below
        # the top margin.  Either signal is sufficient for a stable cut.
        if max(sizes, default=0.0) >= 11.0 or y0 >= page_height * 0.18:
            title_blocks.append((y0, value))

    if title_blocks:
        # The actual title follows a possible running head in text order.
        return matches[-1]

    last = matches[-1]
    trailing_chars = len(text[last.end():].strip())
    top_only = last.start() <= max(240, int(len(text) * 0.18))
    if top_only and trailing_chars > 600:
        return None
    return last


def _strip_running_exclusion_header(text: str, actual_marker_start: int) -> str:
    """Remove a top running end-matter header while keeping preceding body text.

    The actual end-matter title is cut later.  Without this small cleanup, a
    cross-page Chunk can retain only the page's running ``3.9 Summary`` header;
    a later text-only safeguard could then mistake that header for a real
    boundary and discard the valid paragraph that follows it.
    """
    if actual_marker_start <= 0:
        return text
    prefix = text[:actual_marker_start]
    headers = list(_EXCLUDE_RE.finditer(prefix))
    if not headers:
        return prefix
    header = headers[0]
    if header.start() > max(240, int(len(prefix) * 0.18)):
        return prefix
    before = prefix[:header.start()]
    after = prefix[header.end():]
    # A printed page number normally follows a running head on its own line.
    after = re.sub(r"^\s*\d{1,4}\s*(?:\n|$)", "", after)
    return _clean_text(before + after)


def _section_sort_key(value: str) -> Tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError:
        return (9999,)


def _clean_text(text: str) -> str:
    text = (text or "").replace("\u00ad", "")
    # PDF extraction commonly represents a line-wrapped word as
    # ``inter-\nprocess``.  Join that artifact before chunk boundaries or
    # evidence quotes are calculated; a real spaced hyphen is left intact.
    text = re.sub(r"(?<=[A-Za-z])[-\u2010-\u2015]\s*\n\s*(?=[A-Za-z])", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _forward_to_token_boundary(text: str, start: int) -> int:
    """Move a proposed start right so a Chunk never starts mid-word."""
    start = max(0, min(start, len(text)))
    if start == 0 or start >= len(text):
        return start
    if not (text[start - 1].isalnum() and text[start].isalnum()):
        return start
    match = re.search(r"\s+", text[start:])
    return len(text) if match is None else start + match.end()


def _safe_cut_at_or_after(text: str, target: int) -> int:
    """Choose a token boundary near ``target`` without truncating a word."""
    target = max(1, min(target, len(text)))
    if target == len(text) or not (text[target - 1].isalnum() and text[target].isalnum()):
        return target
    # Prefer the next whitespace.  An unusually long token may exceed the
    # nominal target, which is safer than emitting two corrupt partial tokens.
    match = re.search(r"\s+", text[target:])
    return len(text) if match is None else target + match.start()


def _leading_section_id(text: str) -> str:
    match = re.match(r"^\s*(\d+\.\d+(?:\.\d+)*)\b", text or "")
    return match.group(1) if match else ""


def _split_at_known_section_headings(text: str, known_sections: Set[str]) -> List[str]:
    """Split a page fragment when a real registered heading occurs mid-block.

    PDF extraction sometimes places the tail of a code/example paragraph and
    the next heading in one paragraph.  Treating it as one source piece mixes
    unrelated sections (for example pipe examples with RPC prose).  Only
    headings listed in the section registry are honoured, so figure labels
    such as ``Figure 3.21`` cannot masquerade as textbook sections.
    """
    matches = [
        match for match in _NUMBERED_SECTION_HEADING_RE.finditer(text)
        if match.group("section") in known_sections
    ]
    if not matches:
        return [text]
    parts: List[str] = []
    cursor = 0
    for match in matches:
        if match.start() > cursor:
            prefix = text[cursor:match.start()].strip()
            if prefix:
                parts.append(prefix)
        cursor = match.start()
    tail = text[cursor:].strip()
    if tail:
        parts.append(tail)
    return parts


def _sentence_start_at_or_after(text: str, start: int) -> int:
    """Find a clean prose-sentence start for overlap, or drop the overlap.

    Word-boundary-only overlap still produces fragments such as ``is, either
    P2 or P3``.  For retrieval evidence, retaining a shorter complete
    sentence is preferable to retaining an arbitrary suffix of a sentence.
    """
    start = _forward_to_token_boundary(text, start)
    if start >= len(text):
        return len(text)
    if text[start:start + 1].isupper():
        return start
    for match in re.finditer(r"(?:[.!?][\"')\]]?\s+|\n\s*\n)(?P<char>[A-Z])", text[start:]):
        return start + match.start("char")
    return len(text)


def _split_long_text(text: str, limit: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    result: List[str] = []
    while len(text) > limit:
        lower = max(1, int(limit * 0.60))
        window = text[lower:limit + 1]
        points = [window.rfind("\n\n"), window.rfind(". "), window.rfind("? "),
                  window.rfind("! "), window.rfind("; "), window.rfind(" ")]
        point = max(points)
        cut = lower + point + (2 if point >= 0 and window[point:point+2] in {"\n\n", ". ", "? ", "! ", "; "} else 1) if point >= 0 else _safe_cut_at_or_after(text, limit)
        piece = text[:cut].strip()
        if not piece:
            cut = _safe_cut_at_or_after(text, limit)
            piece = text[:cut].strip()
        result.append(piece)
        text = text[cut:].strip()
    if text:
        result.append(text)
    return result


def _page_pieces(
    text: str,
    page: int,
    piece_limit: int,
    known_sections: Set[str],
) -> List[_Piece]:
    heading_safe_text = "\n\n".join(
        part for part in _split_at_known_section_headings(text, known_sections) if part
    )
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", heading_safe_text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []
    pieces: List[_Piece] = []
    for paragraph in paragraphs:
        for part in _split_long_text(paragraph, piece_limit):
            pieces.append(_Piece(part, page, False))
    return pieces


def _tail_overlap(pieces: List[_Piece], overlap: int) -> List[_Piece]:
    if overlap <= 0:
        return []
    remaining = overlap
    tail: List[_Piece] = []
    for piece in reversed(pieces):
        if remaining <= 0:
            break
        separator_cost = 2 if tail else 0
        available = remaining - separator_cost
        if available <= 0:
            break
        take = min(len(piece.text), available)
        start = _sentence_start_at_or_after(piece.text, len(piece.text) - take)
        value = piece.text[start:].lstrip()
        if value:
            tail.append(_Piece(value, piece.page, True))
            remaining -= len(value) + separator_cost
    tail.reverse()
    return tail


def _primary_section(text: str, pages: List[int], sections: List[str], registry: KPRegistry) -> str:
    if not sections:
        return ""
    matches: List[Tuple[int, str]] = []
    for section in sections:
        # Source PDFs often place the numeric heading and its title on separate
        # lines (for example, "3.1\nProcess Concept").  Recognise both that
        # form and the inline form so a chunk keeps its real primary section.
        match = re.search(
            rf"(?m)^\s*{re.escape(section)}\s*(?:\n+\s*|[ \t]+)\S",
            text,
        )
        if match:
            matches.append((match.start(), section))
    if matches:
        matches.sort()
        # Estimate how much text follows each detected heading before the next heading.
        spans: List[Tuple[int, str]] = []
        for index, (start, section) in enumerate(matches):
            end = matches[index + 1][0] if index + 1 < len(matches) else len(text)
            spans.append((end - start, section))
        return max(spans, key=lambda item: (item[0], _section_sort_key(item[1])))[1]

    weights: Dict[str, int] = {section: 0 for section in sections}
    for page in pages:
        page_sections = registry.section_of_page(page)
        for section in page_sections:
            if section in weights:
                weights[section] += 1
    return max(weights, key=lambda section: (weights[section], _section_sort_key(section)))


def chunk_pdf_by_sections(
    pdf_path: str,
    registry: KPRegistry,
    chunk_size: int = 2000,
    overlap: int = 300,
    min_chars: int = 500,
    verbose: bool = True,
) -> List[SectionChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    if min_chars <= 0 or min_chars > chunk_size:
        raise ValueError("min_chars must be positive and <= chunk_size")

    body_pages: Set[int] = {p for pages in registry.section_pages.values() for p in pages}
    document = fitz.open(pdf_path)
    try:
        total_pages = len(document)
        # Fallback: 若 registry 沒 source_pdf_pages (從 global registry 抽時常見),
        # 用整份 PDF 的所有頁面
        if not body_pages:
            body_pages = set(range(1, total_pages + 1))
            if verbose:
                print(f"[chunker] ⚠️ KP registry 無 source_pdf_pages, "
                      f"fallback 用全 PDF ({total_pages} 頁)")
        elif max(body_pages) > total_pages:
            raise ValueError(
                f"KP registry refers to page {max(body_pages)}, but PDF has only {total_pages} pages"
            )
        if verbose:
            print(f"[chunker] PDF: {Path(pdf_path).name}, {total_pages} pages")
            print(
                f"[chunker] KP body pages: p.{min(body_pages)}-{max(body_pages)} "
                f"({len(body_pages)} pages)"
            )

        source_pieces: List[_Piece] = []
        stopped_at: Tuple[int, str] | None = None
        for page_index in range(total_pages):
            page_number = page_index + 1
            if page_number not in body_pages:
                continue
            text = _clean_text(document[page_index].get_text("text", sort=True) or "")
            if not text:
                continue
            marker = _page_exclusion_marker(document[page_index], text)
            if marker:
                heading = " ".join(marker.group(0).split())
                text = _strip_running_exclusion_header(text, marker.start())
                stopped_at = (page_number, heading)
            if text:
                source_pieces.extend(
                    _page_pieces(
                        text,
                        page_number,
                        max(1, chunk_size - overlap - 2),
                        set(registry.section_of_page(page_number)),
                    )
                )
            if stopped_at:
                break
    finally:
        document.close()

    chunks: List[SectionChunk] = []
    buffer: List[_Piece] = []
    buffer_length = 0
    active_section = ""

    def joined_length(items: List[_Piece]) -> int:
        return sum(len(item.text) for item in items) + max(0, len(items) - 1) * 2

    def emit(force_small: bool = False) -> None:
        nonlocal buffer, buffer_length
        if not buffer:
            return
        # A final buffer containing only copied overlap is not new content.
        if all(item.overlap_only for item in buffer):
            buffer = []
            buffer_length = 0
            return
        text = "\n\n".join(item.text for item in buffer).strip()
        if len(text) < min_chars and not force_small:
            return
        pages = sorted({item.page for item in buffer})
        sections = sorted(
            {section for page in pages for section in registry.section_of_page(page)},
            key=_section_sort_key,
        )
        primary = _primary_section(text, pages, sections, registry)
        chunks.append(
            SectionChunk(
                chunk_id=f"c{len(chunks)+1:04d}",
                text=text,
                pages=pages,
                sections=sections,
                primary_section=primary,
                page_start=min(pages),
                page_end=max(pages),
            )
        )
        buffer = _tail_overlap(buffer, overlap)
        buffer_length = joined_length(buffer)

    for piece in source_pieces:
        heading_section = _leading_section_id(piece.text)
        # Do not let the overlap from the preceding numbered section become
        # evidence for the next one.  A section heading is a semantic boundary,
        # stronger than the normal cross-chunk overlap rule.
        if heading_section and active_section and heading_section != active_section and buffer:
            emit(force_small=True)
            buffer = []
            buffer_length = 0
        if heading_section:
            active_section = heading_section
        added = len(piece.text) + (2 if buffer else 0)
        if buffer and buffer_length + added > chunk_size:
            emit(force_small=True)
        # emit() intentionally retains overlap. A very large next piece may still
        # not fit with that overlap, so shrink only the copied overlap, never the
        # new source content.
        if buffer and buffer_length + len(piece.text) + 2 > chunk_size:
            allowed_overlap = max(0, chunk_size - len(piece.text) - 2)
            buffer = _tail_overlap(buffer, allowed_overlap)
            buffer_length = joined_length(buffer)
        buffer.append(piece)
        buffer_length = joined_length(buffer)
        if buffer_length >= chunk_size:
            emit(force_small=True)

    if buffer and not all(item.overlap_only for item in buffer):
        text_len = joined_length(buffer)
        if text_len < min_chars and chunks:
            # Merge a short final tail only when the hard limit remains satisfied.
            tail_text = "\n\n".join(item.text for item in buffer if not item.overlap_only).strip()
            if tail_text and len(chunks[-1].text) + 2 + len(tail_text) <= chunk_size:
                previous = chunks[-1]
                merged_text = previous.text + "\n\n" + tail_text
                pages = sorted(set(previous.pages) | {item.page for item in buffer})
                sections = sorted(
                    {section for page in pages for section in registry.section_of_page(page)},
                    key=_section_sort_key,
                )
                chunks[-1] = SectionChunk(
                    chunk_id=previous.chunk_id,
                    text=merged_text,
                    pages=pages,
                    sections=sections,
                    primary_section=_primary_section(merged_text, pages, sections, registry),
                    page_start=min(pages),
                    page_end=max(pages),
                )
            elif tail_text:
                emit(force_small=True)
        else:
            emit(force_small=True)

    if verbose:
        if stopped_at:
            print(f"[chunker] stopped at p.{stopped_at[0]} heading: {stopped_at[1]}")
        lengths = [len(chunk.text) for chunk in chunks]
        print(
            f"[chunker] created {len(chunks)} chunks; "
            f"length={min(lengths) if lengths else 0}-{max(lengths) if lengths else 0} chars"
        )
        counts: Dict[str, int] = {}
        for chunk in chunks:
            counts[chunk.primary_section] = counts.get(chunk.primary_section, 0) + 1
        print("[chunker] primary section distribution:")
        for section in sorted(counts, key=_section_sort_key):
            print(f"  {section}: {counts[section]}")
    return chunks


if __name__ == "__main__":
    import argparse
    from kg_builder.kp_loader_v2 import load_kp_v2

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--kp", required=True)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--overlap", type=int, default=300)
    parser.add_argument("--min-chars", type=int, default=500)
    args = parser.parse_args()
    registry = load_kp_v2(args.kp)
    result = chunk_pdf_by_sections(
        args.pdf, registry, args.chunk_size, args.overlap, args.min_chars
    )
    print(f"\nTotal: {len(result)} chunks")
