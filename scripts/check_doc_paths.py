#!/usr/bin/env python
"""Check that repository paths cited in the docs actually exist.

Documentation frequently points readers at a file: "the rules live in
``src/unxt/_src/quantity/register_primitives.py``". Nothing validates those --
they are prose, not links -- so they rot silently when files move, and a reader
who follows one finds nothing. This script fails when a cited path is gone.

Only paths that look like repository paths are checked: a backtick-quoted string
starting with a known top-level directory and ending in a source-file suffix.
Prose that merely mentions a module name is left alone.

Usage: ``python scripts/check_doc_paths.py [repo_root]``
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

#: Backtick-quoted paths rooted at a real top-level directory.
PATH_RE = re.compile(
    r"`((?:src|packages|tests|docs|scripts)/[A-Za-z0-9_./-]+\.(?:py|md|toml|yaml|yml|cfg))`"
)

#: Where documentation lives. Package docs are reached directly rather than
#: through the ``docs/packages/*`` symlinks, so each file is checked once.
DOC_GLOBS = ("docs/**/*.md", "packages/*/docs/*.md", "*.md")


def main(root: Path) -> int:
    """Report every cited repository path that does not exist."""
    missing: list[tuple[Path, str]] = []
    checked = 0

    for pattern in DOC_GLOBS:
        for doc in sorted(root.glob(pattern)):
            if "_build" in doc.parts:
                continue
            for cited in PATH_RE.findall(doc.read_text(encoding="utf-8")):
                checked += 1
                if not (root / cited).exists():
                    missing.append((doc.relative_to(root), cited))

    if missing:
        logger.error("%d cited path(s) do not exist:", len(missing))
        for doc, cited in missing:
            logger.error("  %s: %s", doc, cited)
        logger.error("Update the path, or drop it if the file is gone.")
        return 1

    logger.info("All %d cited repository paths exist.", checked)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()))
