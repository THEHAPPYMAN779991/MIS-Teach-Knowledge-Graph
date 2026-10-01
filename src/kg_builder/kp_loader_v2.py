"""kg_builder/kp_loader_v2.py - strict canonical KP v2 loader.

Drop-in replacement with duplicate/alias validation and conservative automatic
name variants. Chapter-local KP IDs remain extraction hints, not global graph
identities.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


_AUTO_STRIP_SUFFIXES = [
    " Service", " Services", " Mechanism", " Method", " Model",
    " Approach", " Strategy", "-Service", "-Mechanism",
]
_AUTO_STRIP_PREFIXES = ["Operating-System ", "Operating System ", "OS "]
_GENERIC_VARIANTS = {
    "system", "service", "method", "model", "approach", "strategy",
    "mechanism", "process", "program", "interface", "operation",
}


def _safe_variant(value: str) -> bool:
    value = " ".join((value or "").split()).strip()
    if len(value) < 3 or value.lower() in _GENERIC_VARIANTS:
        return False
    # Single-word variants are allowed only when reasonably specific.
    return len(value.split()) >= 2 or len(value) >= 7


def _auto_expand_variants(name: str) -> List[str]:
    base = " ".join((name or "").split()).strip()
    variants: Set[str] = set()
    for suffix in _AUTO_STRIP_SUFFIXES:
        if base.endswith(suffix):
            candidate = base[:-len(suffix)].strip()
            if _safe_variant(candidate):
                variants.add(candidate)
    for prefix in _AUTO_STRIP_PREFIXES:
        if base.startswith(prefix):
            candidate = base[len(prefix):].strip()
            if _safe_variant(candidate):
                variants.add(candidate)
    if "," in base:
        candidate = base.split(",", 1)[0].strip()
        if _safe_variant(candidate):
            variants.add(candidate)
    return sorted(variants, key=str.casefold)


@dataclass
class KP:
    id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    concept_type: str = ""
    definition: str = ""
    primary_section: str = ""
    also_discussed_in: List[str] = field(default_factory=list)
    source_pdf_pages: List[int] = field(default_factory=list)
    chapter: str = ""
    _auto_aliases: List[str] = field(default_factory=list)

    @property
    def all_names(self) -> List[str]:
        return [self.name, *self.aliases, *self._auto_aliases]

    def compute_auto_aliases(self) -> None:
        auto: Set[str] = set()
        for base in [self.name, *self.aliases]:
            auto.update(_auto_expand_variants(base))
        existing = {self.name.casefold(), *(a.casefold() for a in self.aliases)}
        self._auto_aliases = [v for v in sorted(auto, key=str.casefold) if v.casefold() not in existing]


@dataclass
class SectionMeta:
    section: str
    title: str
    source_pdf_pages: List[int] = field(default_factory=list)


class KPRegistry:
    def __init__(self, chapter: str, chapter_title: str, source_pdf: str,
                 schema_version: str, scope: Dict[str, Any] | None = None,
                 policy: List[str] | None = None):
        self.chapter = chapter
        self.chapter_title = chapter_title
        self.source_pdf = source_pdf
        self.schema_version = schema_version
        self.scope = scope or {}
        self.policy = policy or []
        self.by_id: Dict[str, KP] = {}
        self.by_name: Dict[str, KP] = {}
        self.by_alias: Dict[str, KP] = {}
        self.alias_collisions: Dict[str, Set[str]] = {}
        self.sections: List[SectionMeta] = []
        self.section_pages: Dict[str, List[int]] = {}
        self.section_kps: Dict[str, List[KP]] = {}
        self.page_sections: Dict[int, List[str]] = {}
        self.by_type: Dict[str, List[KP]] = {}

    def _register_alias(self, alias: str, kp: KP) -> None:
        key = alias.casefold().strip()
        if not key or key == kp.name.casefold().strip():
            return
        canonical = self.by_name.get(key)
        if canonical and canonical.id != kp.id:
            ids = self.alias_collisions.setdefault(key, set())
            ids.update({canonical.id, kp.id})
            self.by_alias.pop(key, None)
            return
        existing = self.by_alias.get(key)
        if existing and existing.id != kp.id:
            ids = self.alias_collisions.setdefault(key, set())
            ids.update({existing.id, kp.id})
            self.by_alias.pop(key, None)
            return
        if key not in self.alias_collisions:
            self.by_alias[key] = kp

    def add_kp(self, kp: KP) -> None:
        if not kp.id:
            raise ValueError("KP id must not be empty")
        if not kp.name:
            raise ValueError(f"KP {kp.id} has an empty name")
        if not kp.primary_section:
            raise ValueError(f"KP {kp.id} has an empty primary_section")
        if kp.id in self.by_id:
            raise ValueError(f"Duplicate KP id: {kp.id}")
        name_key = kp.name.casefold().strip()
        if name_key in self.by_name:
            raise ValueError(
                f"Duplicate canonical KP name: {kp.name!r} "
                f"({self.by_name[name_key].id}, {kp.id})"
            )
        # A canonical name wins over an earlier ambiguous alias.
        prior_alias = self.by_alias.pop(name_key, None)
        if prior_alias and prior_alias.id != kp.id:
            ids = self.alias_collisions.setdefault(name_key, set())
            ids.update({prior_alias.id, kp.id})

        kp.compute_auto_aliases()
        self.by_id[kp.id] = kp
        self.by_name[name_key] = kp
        for alias in [*kp.aliases, *kp._auto_aliases]:
            self._register_alias(alias, kp)
        self.section_kps.setdefault(kp.primary_section, []).append(kp)
        self.by_type.setdefault(kp.concept_type, []).append(kp)

    def add_section(self, sec: SectionMeta) -> None:
        if not sec.section:
            raise ValueError("Section number must not be empty")
        if sec.section in self.section_pages:
            raise ValueError(f"Duplicate section: {sec.section}")
        self.sections.append(sec)
        self.section_pages[sec.section] = list(sec.source_pdf_pages)
        for page in sec.source_pdf_pages:
            self.page_sections.setdefault(page, []).append(sec.section)

    def resolve(self, term: str) -> Optional[KP]:
        key = (term or "").casefold().strip()
        return self.by_name.get(key) or self.by_alias.get(key)

    def kps_in_section(self, section: str, include_cross_ref: bool = False) -> List[KP]:
        result = list(self.section_kps.get(section, []))
        if include_cross_ref:
            seen = {kp.id for kp in result}
            for kp in self.by_id.values():
                if section in kp.also_discussed_in and kp.id not in seen:
                    result.append(kp)
                    seen.add(kp.id)
        return result

    def section_of_page(self, page: int) -> List[str]:
        return list(self.page_sections.get(page, []))

    def all_kps(self) -> List[KP]:
        return list(self.by_id.values())

    def validate(self) -> Dict[str, Any]:
        known_sections = set(self.section_pages)
        bad_primary = [kp.id for kp in self.by_id.values() if kp.primary_section not in known_sections]
        bad_cross = [
            {"kp_id": kp.id, "section": sec}
            for kp in self.by_id.values()
            for sec in kp.also_discussed_in
            if sec not in known_sections
        ]
        return {
            "status": "FAILED" if bad_primary else ("WARN" if bad_cross or self.alias_collisions else "PASSED"),
            "invalid_primary_sections": bad_primary,
            "unknown_cross_reference_sections": bad_cross,
            "ambiguous_aliases": {k: sorted(v) for k, v in self.alias_collisions.items()},
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "chapter": self.chapter,
            "total_kps": len(self.by_id),
            "sections": len(self.sections),
            "total_explicit_aliases": sum(len(kp.aliases) for kp in self.by_id.values()),
            "total_auto_aliases": sum(len(kp._auto_aliases) for kp in self.by_id.values()),
            "ambiguous_aliases": len(self.alias_collisions),
            "concept_types": {t: len(kps) for t, kps in self.by_type.items()},
            "section_kps": {s: len(kps) for s, kps in self.section_kps.items()},
            "validation": self.validate(),
        }


def _int_pages(values: Any) -> List[int]:
    pages: List[int] = []
    for value in values or []:
        try:
            page = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid PDF page number: {value!r}")
        if page <= 0:
            raise ValueError(f"PDF page numbers must be positive: {page}")
        if page not in pages:
            pages.append(page)
    return pages


def load_kp_v2(path: str) -> KPRegistry:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = str(data.get("schema_version", ""))
    if "canonical-knowledge-points" not in schema:
        raise ValueError(f"Unsupported schema: {schema}; expected canonical-knowledge-points-vN")

    registry = KPRegistry(
        chapter=str(data.get("chapter", "")).strip(),
        chapter_title=str(data.get("chapter_title", "")).strip(),
        source_pdf=str(data.get("source_pdf", "")).strip(),
        schema_version=schema,
        scope=data.get("scope", {}),
        policy=data.get("canonicalization_policy", []),
    )
    if not registry.chapter:
        raise ValueError("Missing chapter in KP registry")

    for section_data in data.get("sections", []):
        section = SectionMeta(
            section=str(section_data.get("section", "")).strip(),
            title=str(section_data.get("title", "")).strip(),
            source_pdf_pages=_int_pages(section_data.get("source_pdf_pages", [])),
        )
        registry.add_section(section)
        for item in section_data.get("knowledge_points", []):
            kp = KP(
                id=str(item.get("id", "")).strip(),
                name=str(item.get("name", "")).strip(),
                aliases=[str(a).strip() for a in item.get("aliases", []) if str(a).strip()],
                concept_type=str(item.get("concept_type", "")).strip(),
                definition=str(item.get("definition", "")).strip(),
                primary_section=str(item.get("primary_section", "")).strip(),
                also_discussed_in=[str(s).strip() for s in item.get("also_discussed_in", []) if str(s).strip()],
                source_pdf_pages=_int_pages(item.get("source_pdf_pages", [])),
                chapter=registry.chapter,
            )
            registry.add_kp(kp)

    validation = registry.validate()
    if validation["invalid_primary_sections"]:
        raise ValueError(f"Invalid primary sections: {validation['invalid_primary_sections'][:20]}")
    return registry


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    args = parser.parse_args()
    print(json.dumps(load_kp_v2(args.input).stats(), ensure_ascii=False, indent=2))
