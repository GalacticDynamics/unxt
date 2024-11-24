"""Doctest configuration."""

import os
from collections.abc import Callable, Iterable, Sequence
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE
from typing import Any

from sybil import Document, Region, Sybil
from sybil.parsers.myst import (
    DocTestDirectiveParser as MarkdownDocTestDirectiveParser,
    PythonCodeBlockParser as MarkdownPythonCodeBlockParser,
    SkipParser as MarkdownSkipParser,
)
from sybil.parsers.rest import DocTestParser as ReSTDocTestParser

from optional_dependencies import OptionalDependencyEnum, auto

optionflags = ELLIPSIS | NORMALIZE_WHITESPACE

parsers: Sequence[Callable[[Document], Iterable[Region]]] = [
    MarkdownDocTestDirectiveParser(optionflags=optionflags),
    MarkdownPythonCodeBlockParser(doctest_optionflags=optionflags),
    MarkdownSkipParser(),
]

docs = Sybil(parsers=parsers, patterns=["*.md"])
python = Sybil(
    parsers=[ReSTDocTestParser(optionflags=optionflags), *parsers], patterns=["*.py"]
)


pytest_collect_file = (docs + python).pytest()


class OptDeps(OptionalDependencyEnum):
    """Optional dependencies for ``unxt``."""

    ASTROPY = auto()
    GALA = auto()
    MATPLOTLIB = auto()


collect_ignore_glob = []
if not OptDeps.ASTROPY.installed:
    collect_ignore_glob.append("src/unxt/_interop/unxt_interop_astropy/*")
if not OptDeps.GALA.installed:
    collect_ignore_glob.append("src/unxt/_interop/unxt_interop_gala/*")
    collect_ignore_glob.append("docs/interop/gala.md")
if not OptDeps.MATPLOTLIB.installed:
    collect_ignore_glob.append("src/unxt/_interop/unxt_interop_mpl/*")


def pytest_generate_tests(metafunc: Any) -> None:
    os.environ["UNXT_ENABLE_RUNTIME_TYPECHECKING"] = "beartype.beartype"
