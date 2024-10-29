"""Doctest configuration."""

import os
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE
from typing import Any

from sybil import Sybil
from sybil.parsers.rest import DocTestParser, PythonCodeBlockParser, SkipParser

from optional_dependencies import OptionalDependencyEnum, auto

pytest_collect_file = Sybil(
    parsers=[
        DocTestParser(optionflags=ELLIPSIS | NORMALIZE_WHITESPACE),
        PythonCodeBlockParser(),
        SkipParser(),
    ],
    patterns=["*.rst", "*.py"],
).pytest()


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
if not OptDeps.MATPLOTLIB.installed:
    collect_ignore_glob.append("src/unxt/_interop/unxt_interop_mpl/*")


def pytest_generate_tests(metafunc: Any) -> None:
    os.environ["UNXT_ENABLE_RUNTIME_TYPECHECKING"] = "beartype.beartype"
