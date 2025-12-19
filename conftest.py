"""Doctest configuration."""

import os
from collections.abc import Callable, Iterable, Sequence
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE
from typing import Any

from sybil import Document, Region, Sybil
from sybil.parsers import myst, rest

from optional_dependencies import OptionalDependencyEnum, auto

optionflags = ELLIPSIS | NORMALIZE_WHITESPACE

parsers: Sequence[Callable[[Document], Iterable[Region]]] = [
    myst.DocTestDirectiveParser(optionflags=optionflags),
    myst.PythonCodeBlockParser(doctest_optionflags=optionflags),
    myst.SkipParser(),
]

docs = Sybil(parsers=parsers, patterns=["*.md"])
python = Sybil(
    parsers=[
        *parsers,
        rest.PythonCodeBlockParser(),
        rest.DocTestParser(optionflags=optionflags),
        rest.SkipParser(),
    ],
    patterns=["*.py"],
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
    collect_ignore_glob.append("docs/interop/matplotlib.md")


def pytest_generate_tests(metafunc: Any) -> None:
    os.environ["UNXT_ENABLE_RUNTIME_TYPECHECKING"] = "beartype.beartype"
