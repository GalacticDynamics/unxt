"""Doctest configuration."""

import os
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE
from typing import Any

from sybil import Sybil
from sybil.parsers.myst import (
    DocTestDirectiveParser as MarkdownDocTestParser,
    PythonCodeBlockParser as MarkdownPythonCodeBlockParser,
    SkipParser as MarkdownSkipParser,
)
from sybil.parsers.rest import (
    DocTestParser as ReSTDocTestParser,
    PythonCodeBlockParser as ReSTPythonCodeBlockParser,
    SkipParser as ReSTSkipParser,
)

from optional_dependencies import OptionalDependencyEnum, auto

markdown_examples = Sybil(
    parsers=[
        MarkdownDocTestParser(),
        MarkdownPythonCodeBlockParser(),
        MarkdownSkipParser(),
    ],
    patterns=["*.md"],
)

rest_examples = Sybil(
    parsers=[
        ReSTDocTestParser(optionflags=ELLIPSIS | NORMALIZE_WHITESPACE),
        ReSTPythonCodeBlockParser(),
        ReSTSkipParser(),
    ],
    patterns=["*.py"],
)


pytest_collect_file = (markdown_examples + rest_examples).pytest()


class OptDeps(OptionalDependencyEnum):
    """Optional dependencies for ``unxt``."""

    ASTROPY = auto()
    GALA = auto()
    MATPLOTLIB = auto()
    ZEROTH = auto()


collect_ignore_glob = []
if not OptDeps.ASTROPY.installed:
    collect_ignore_glob.append("src/unxt/_interop/unxt_interop_astropy/*")
if not OptDeps.GALA.installed:
    collect_ignore_glob.append("src/unxt/_interop/unxt_interop_gala/*")
if not OptDeps.MATPLOTLIB.installed or not OptDeps.ZEROTH.installed:
    collect_ignore_glob.append("src/unxt/_interop/unxt_interop_mpl/*")


def pytest_generate_tests(metafunc: Any) -> None:
    os.environ["UNXT_ENABLE_RUNTIME_TYPECHECKING"] = "beartype.beartype"
