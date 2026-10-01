"""Single ontology and verification contract for the formal GraphRAG graph.

The builder, relation verifier, and Neo4j importer must use this module rather
than maintaining separate relation allow-lists.  Pedagogical prerequisites are
not created by the per-Chunk automatic extraction pipeline; they are accepted
at import time only from the separate post-extraction layer after it has
passed evidence, confidence, cycle, and directness validation.
"""

from __future__ import annotations


FORMAL_RELATION_TYPES = frozenset({
    "HAS_SUBTYPE", "PART_OF", "IMPLEMENTS", "USES", "INVOKES",
    "PROVIDES", "MANAGES", "ENABLES", "CAUSES", "EXAMPLE_OF",
    "CONTRASTED_WITH", "MAPS_TO", "REPRESENTS", "CONTAINS", "CREATES",
    "TRANSITIONS_TO", "SAVES_TO", "LOADS_FROM", "RECLAIMS", "WAITS_ON",
    "SCHEDULES", "INCREASES", "REDUCES",
})

SYMMETRIC_RELATION_TYPES = frozenset({"CONTRASTED_WITH"})
VERIFIED_RELATION_STATUSES = frozenset({
    "VERIFIED", "VERIFIED_REVERSED", "VERIFIED_RETYPED",
})

# The importer accepts only the dedicated verified pedagogical layer, never a
# PREREQUISITE_OF relation leaked from generic per-Chunk extraction.
IMPORT_RELATION_TYPES = FORMAL_RELATION_TYPES | frozenset({"PREREQUISITE_OF"})
