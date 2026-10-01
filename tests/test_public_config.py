"""Static public-package checks; no credentials, textbook, or database is required."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"


class PublicPackageTests(unittest.TestCase):
    def test_python_sources_compile(self) -> None:
        for path in SOURCE.rglob("*.py"):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_no_google_key_literal_or_unsafe_neo4j_default(self) -> None:
        for path in ROOT.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertIsNone(re.search(r"AIza[0-9A-Za-z_-]{20,}", source), path)
        config = (SOURCE / "config.py").read_text(encoding="utf-8")
        self.assertIn('os.getenv("NEO4J_URI", "")', config)
        self.assertIn('os.getenv("NEO4J_USERNAME", "")', config)
        self.assertNotIn("123456789", config)


if __name__ == "__main__":
    unittest.main()
