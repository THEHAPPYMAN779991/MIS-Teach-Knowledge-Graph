"""集中所有 Cypher 查詢樣板。"""

CONSTRAINTS = [
    "CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE",
    "CREATE CONSTRAINT chapter_id_unique IF NOT EXISTS FOR (c:Chapter) REQUIRE c.chapterId IS UNIQUE",
    "CREATE CONSTRAINT book_id_unique IF NOT EXISTS FOR (b:Book) REQUIRE b.bookId IS UNIQUE",
]

CREATE_VECTOR_INDEX = """
CREATE VECTOR INDEX $index_name IF NOT EXISTS
FOR (c:Concept) ON (c.embedding)
OPTIONS { indexConfig: {
  `vector.dimensions`: $dimensions,
  `vector.similarity_function`: 'cosine'
}}
"""

MERGE_BOOK = """
MERGE (b:Book {bookId: $bookId})
  ON CREATE SET b.title = $title, b.createdAt = datetime()
  ON MATCH  SET b.title = coalesce(b.title, $title)
RETURN b
"""

MERGE_CHAPTER = """
MERGE (ch:Chapter {chapterId: $chapterId})
  ON CREATE SET ch.title = $title, ch.order = $order, ch.level = $level
  ON MATCH  SET ch.title = coalesce(ch.title, $title), ch.order = $order, ch.level = $level
WITH ch
MATCH (b:Book {bookId: $bookId})
MERGE (ch)-[:PART_OF]->(b)
RETURN ch
"""

MERGE_CHUNK = """
MERGE (k:Chunk {chunkId: $chunkId})
  ON CREATE SET k.text = $text, k.chunkSeqId = $chunkSeqId,
                k.chapterId = $chapterId, k.bookId = $bookId, k.page = $page
WITH k
MATCH (ch:Chapter {chapterId: $chapterId})
MERGE (k)-[:IN_CHAPTER]->(ch)
RETURN k
"""

MERGE_CONCEPT = """
MERGE (c:Concept {name: $name})
  ON CREATE SET c.definition=$definition, c.aliases=$aliases,
                c.category=$category, c.isFineGrained=$isFineGrained,
                c.bookIds=$bookIds, c.createdAt=datetime()
  ON MATCH SET
    c.definition = CASE WHEN c.definition IS NULL OR size(c.definition) < size($definition)
                        THEN $definition ELSE c.definition END,
    c.aliases       = apoc.coll.toSet(coalesce(c.aliases, []) + $aliases),
    c.bookIds       = apoc.coll.toSet(coalesce(c.bookIds, []) + $bookIds),
    c.category      = coalesce(c.category, $category),
    c.isFineGrained = coalesce(c.isFineGrained, $isFineGrained)
RETURN c
"""

MERGE_CONCEPT_NO_APOC = """
MERGE (c:Concept {name: $name})
  ON CREATE SET c.definition=$definition, c.aliases=$aliases,
                c.category=$category, c.isFineGrained=$isFineGrained,
                c.bookIds=$bookIds, c.createdAt=datetime()
  ON MATCH SET
    c.definition = CASE WHEN c.definition IS NULL OR size(c.definition) < size($definition)
                        THEN $definition ELSE c.definition END,
    c.category      = coalesce(c.category, $category),
    c.isFineGrained = coalesce(c.isFineGrained, $isFineGrained),
    c.bookIds       = [x IN $bookIds WHERE NOT x IN coalesce(c.bookIds, [])]
                      + coalesce(c.bookIds, []),
    c.aliases       = [x IN $aliases WHERE NOT x IN coalesce(c.aliases, [])]
                      + coalesce(c.aliases, [])
RETURN c
"""

LINK_CONCEPT_TO_CHUNK = """
MATCH (c:Concept {name: $name})
MATCH (k:Chunk    {chunkId: $chunkId})
MERGE (c)-[r:MENTIONED_IN]->(k)
  ON CREATE SET r.count = 1
  ON MATCH  SET r.count = coalesce(r.count, 0) + 1
"""

LINK_PREREQUISITE = """
MATCH (a:Concept {name: $prereq})
MATCH (b:Concept {name: $target})
MERGE (a)-[r:PREREQUISITE_OF]->(b)
  ON CREATE SET r.confidence=$confidence, r.reason=$reason,
                r.source=$source, r.createdAt=datetime()
  ON MATCH SET
    r.confidence = CASE WHEN $confidence > coalesce(r.confidence, 0)
                        THEN $confidence ELSE r.confidence END,
    r.reason     = coalesce(r.reason, $reason)
"""

LINK_HAS_SUBTYPE = """
MATCH (p:Concept {name: $parent})
MATCH (c:Concept {name: $child})
MERGE (p)-[r:HAS_SUBTYPE]->(c)
  ON CREATE SET r.createdAt = datetime(), r.source = $source
"""

GET_SUBTYPES = """
MATCH (p:Concept {name: $name})-[:HAS_SUBTYPE]->(c:Concept)
RETURN c.name AS name, c.definition AS definition, c.category AS category
ORDER BY c.name
"""

GET_PARENT_CONCEPTS = """
MATCH (p:Concept)-[:HAS_SUBTYPE]->(c:Concept {name: $name})
RETURN p.name AS name, p.definition AS definition, p.category AS category
"""

GET_ALL_CONCEPTS = """
MATCH (c:Concept)
RETURN c.name AS name, c.definition AS definition, c.category AS category
ORDER BY c.name
"""

VECTOR_SEARCH = """
CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
YIELD node, score
RETURN node.name AS name, node.definition AS definition,
       properties(node)['category'] AS category, score
ORDER BY score DESC, node.name ASC
"""

GET_PREREQUISITES = """
MATCH path = (root:Concept)-[:PREREQUISITE_OF*1..$depth]->(c:Concept {name: $name})
RETURN DISTINCT root.name AS name, root.definition AS definition,
       length(path) AS distance
ORDER BY distance ASC, name
"""

GET_DESCENDANTS = """
MATCH path = (c:Concept {name: $name})-[:PREREQUISITE_OF*1..$depth]->(d:Concept)
RETURN DISTINCT d.name AS name, d.definition AS definition,
       length(path) AS distance
ORDER BY distance ASC, name
"""

EXPAND_CONCEPT_CONTEXT = """
MATCH (c:Concept) WHERE c.name IN $names
OPTIONAL MATCH (c)-[mentionRelation]-(sourceChunk:Chunk)
WHERE type(mentionRelation) IN ['MENTIONED_IN', 'MENTIONS']
WITH c, collect(DISTINCT sourceChunk.text) AS sourceChunks
OPTIONAL MATCH (p:Concept)-[:PREREQUISITE_OF]->(c)
OPTIONAL MATCH (c)-[:PREREQUISITE_OF]->(d:Concept)
OPTIONAL MATCH (c)-[:HAS_SUBTYPE]->(s:Concept)
OPTIONAL MATCH (parent:Concept)-[:HAS_SUBTYPE]->(c)
RETURN c.name AS name,
       c.definition AS definition,
       [text IN sourceChunks WHERE text IS NOT NULL][..3] AS sample_chunks,
       collect(DISTINCT p.name) AS prerequisites,
       collect(DISTINCT d.name) AS leads_to,
       collect(DISTINCT s.name) AS subtypes,
       collect(DISTINCT parent.name) AS parents,
       coalesce(properties(c)['bookIds'], []) AS book_ids
"""

DETECT_CYCLE_FOR_EDGE = """
MATCH (a:Concept {name: $prereq}), (b:Concept {name: $target})
RETURN EXISTS {
  MATCH path = (b)-[:PREREQUISITE_OF*1..20]->(a)
  RETURN path
} AS would_create_cycle
"""

MERGE_ALIAS_INTO_MAIN = """
MATCH (alias:Concept {name: $alias_name})
MATCH (main:Concept  {name: $main_name})
WITH alias, main
OPTIONAL MATCH (alias)-[m:MENTIONED_IN]->(k:Chunk)
FOREACH (_ IN CASE WHEN m IS NOT NULL THEN [1] ELSE [] END |
  MERGE (main)-[nm:MENTIONED_IN]->(k)
  ON CREATE SET nm.count = m.count
  ON MATCH  SET nm.count = coalesce(nm.count, 0) + coalesce(m.count, 1)
)
WITH alias, main
OPTIONAL MATCH (alias)-[op:PREREQUISITE_OF]->(t:Concept)
FOREACH (_ IN CASE WHEN op IS NOT NULL AND t.name <> $main_name THEN [1] ELSE [] END |
  MERGE (main)-[np:PREREQUISITE_OF]->(t)
    ON CREATE SET np.confidence=op.confidence, np.reason=op.reason, np.source=op.source
)
WITH alias, main
OPTIONAL MATCH (s:Concept)-[ip:PREREQUISITE_OF]->(alias)
FOREACH (_ IN CASE WHEN ip IS NOT NULL AND s.name <> $main_name THEN [1] ELSE [] END |
  MERGE (s)-[np2:PREREQUISITE_OF]->(main)
    ON CREATE SET np2.confidence=ip.confidence, np2.reason=ip.reason, np2.source=ip.source
)
WITH alias, main
OPTIONAL MATCH (alias)-[osub:HAS_SUBTYPE]->(sc:Concept)
FOREACH (_ IN CASE WHEN osub IS NOT NULL AND sc.name <> $main_name THEN [1] ELSE [] END |
  MERGE (main)-[:HAS_SUBTYPE]->(sc)
)
WITH alias, main
OPTIONAL MATCH (pc:Concept)-[isub:HAS_SUBTYPE]->(alias)
FOREACH (_ IN CASE WHEN isub IS NOT NULL AND pc.name <> $main_name THEN [1] ELSE [] END |
  MERGE (pc)-[:HAS_SUBTYPE]->(main)
)
WITH alias, main
SET main.aliases = CASE
  WHEN alias.name IN coalesce(main.aliases, []) THEN main.aliases
  ELSE coalesce(main.aliases, []) + alias.name END
WITH alias, main
DETACH DELETE alias
RETURN main.name AS canonical
"""
