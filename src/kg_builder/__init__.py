"""知識圖譜建構模組。"""
from kg_builder.concept_extractor import (
    ConceptExtractor,
    Concept,
    ExtractionResult,
    SubtypeEdge,
    LocalPrerequisiteEdge,
    deduplicate_concepts,
    deduplicate_subtype_edges,
    deduplicate_local_prerequisites,
    get_concept_book_ids,
)
from kg_builder.prerequisite_analyzer import PrerequisiteAnalyzer, PrerequisiteEdge
from kg_builder.neo4j_writer import Neo4jWriter
from kg_builder.cross_book_merger import CrossBookMerger, MergeStats

__all__ = [
    "ConceptExtractor", "Concept", "ExtractionResult",
    "SubtypeEdge", "LocalPrerequisiteEdge",
    "deduplicate_concepts", "deduplicate_subtype_edges",
    "deduplicate_local_prerequisites", "get_concept_book_ids",
    "PrerequisiteAnalyzer", "PrerequisiteEdge",
    "Neo4jWriter",
    "CrossBookMerger", "MergeStats",
]
