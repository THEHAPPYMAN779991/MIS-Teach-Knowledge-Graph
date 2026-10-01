"""Verify a Neo4j connection using only local environment configuration."""

from __future__ import annotations

import sys
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIR))

from config import settings  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402


def main() -> int:
    missing = [
        name
        for name, value in (
            ("NEO4J_URI", settings.neo4j_uri),
            ("NEO4J_USERNAME", settings.neo4j_username),
            ("NEO4J_PASSWORD", settings.neo4j_password),
        )
        if not value
    ]
    if missing:
        print("Missing required environment variables: " + ", ".join(missing))
        return 2

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    try:
        driver.verify_connectivity()
    finally:
        driver.close()
    print("Neo4j connectivity verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
