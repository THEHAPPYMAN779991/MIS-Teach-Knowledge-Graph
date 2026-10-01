# ComputerScienceKG

ComputerScienceKG builds an evidence-backed knowledge graph from authorised textbook material,
writes the graph to Neo4j, creates Concept and Chunk embeddings, and exposes GraphRAG retrieval
over a local FastAPI service.

No textbook, concept registry, Neo4j data, embedding, database dump, or generated graph output is
included in this public package. Supply only content that you are authorised to process.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure either Vertex AI (`GEMINI_BACKEND=vertex` with `VERTEX_PROJECT` or
`GOOGLE_CLOUD_PROJECT`) or AI Studio (`GEMINI_BACKEND=aistudio` with a local API-key environment
variable). Configure `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` before an operation that
uses Neo4j. Never commit `.env`.

## Public entry points

```powershell
# Verify only the environment-based Neo4j connection; values are never printed.
python .\scripts\check_neo4j.py

# Start the GraphRAG API used by the MIS-Teach backend.
python .\scripts\serve_api.py

# Build authorised chapters with a local textbook directory and concept registry.
python .\scripts\build_book.py --book-dir .\BOOK --chapters 3 --output-dir .\outputs\ch03

# Import a validated build and create embeddings.
python .\scripts\ingest.py --input .\outputs\ch03\ch03.json --book-id demo_book --book-title "Demo Book" --chapter-id ch03 --chapter-num 3 --chapter-title "Demo Chapter"
python .\scripts\embed.py --chapter-id ch03 --targets both

# Query the imported graph.
python .\scripts\query.py --query "Explain a sample concept" --top-k 5 --top-m 5 --hops 1 --format json
```

## Public-package check

```powershell
python -m unittest discover -s tests
```

The `src/` modules are white-listed active modules copied without functional changes. `scripts/`
contains only stable public wrappers and a replacement for the legacy connection check; the
replacement has no URI, username, password, or API-key defaults.
