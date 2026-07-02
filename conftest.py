"""Doctest configuration."""

import os
from collections.abc import Callable, Iterable, Sequence
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE
from typing import Any

from sybil import Document, Region, Sybil
from sybil.evaluators.doctest import DocTestEvaluator
from sybil.evaluators.python import PythonEvaluator
from sybil.parsers import myst
from sybil.parsers.abstract.codeblock import PythonDocTestOrCodeBlockParser
from sybil.parsers.abstract.doctest import DocTestStringParser

from optional_dependencies import OptionalDependencyEnum, auto

optionflags = ELLIPSIS | NORMALIZE_WHITESPACE


class PlainDocTestParser:
    """Parser for plain >>> doctests in Python docstrings."""

    def __init__(self, doctest_optionflags: int = 0) -> None:
        self.doctest_parser = DocTestStringParser(DocTestEvaluator(doctest_optionflags))

    def __call__(self, document: Document) -> Iterable[Region]:
        """Parse plain doctest prompts from Python docstring text."""
        yield from self.doctest_parser(document.text, document.path)


class PyconCodeBlockParser(PythonDocTestOrCodeBlockParser):
    """Parser for MyST pycon code blocks with doctest evaluation."""

    def __init__(
        self,
        future_imports: Sequence[str] = (),
        doctest_optionflags: int = 0,
    ) -> None:
        """Initialize parser state."""
        self.doctest_parser = DocTestStringParser(DocTestEvaluator(doctest_optionflags))
        self.codeblock_parser = myst.CodeBlockParser(
            language="pycon",
            evaluator=PythonEvaluator(future_imports),
        )


markdown_parsers: Sequence[Callable[[Document], Iterable[Region]]] = [
    PyconCodeBlockParser(doctest_optionflags=optionflags),
    myst.DocTestDirectiveParser(optionflags=optionflags),
    myst.PythonCodeBlockParser(doctest_optionflags=optionflags),
    myst.SkipParser(),
]

docs = Sybil(parsers=markdown_parsers, patterns=["*.md"])
python = Sybil(
    parsers=[
        myst.SkipParser(),
        myst.PythonCodeBlockParser(doctest_optionflags=optionflags),
        PlainDocTestParser(doctest_optionflags=optionflags),
    ],
    patterns=["*.py"],
)


pytest_collect_file = (docs + python).pytest()


class OptDeps(OptionalDependencyEnum):
    """Optional dependencies for ``unxt``."""

    ASTROPY = auto()
    GALA = auto()
    UNXTS_INTEROP_GALA = auto()
    UNXTS_INTEROP_MATPLOTLIB = auto()


collect_ignore_glob = []
if not OptDeps.ASTROPY.installed:
    collect_ignore_glob.append("src/unxt/_interop/unxt_interop_astropy/*")
# The package docs are collected through the docs/packages/<name> symlinks, so
# ignore that (symlink) path, not the real package path.
# gala is skipped where it cannot build (Windows), even though the
# unxts.interop.gala package itself is installed; gate on gala being importable.
if not (OptDeps.UNXTS_INTEROP_GALA.installed and OptDeps.GALA.installed):
    collect_ignore_glob.append("docs/packages/unxts.interop.gala/*")
    collect_ignore_glob.append("packages/unxts.interop.gala/docs/*")
if not OptDeps.UNXTS_INTEROP_MATPLOTLIB.installed:
    collect_ignore_glob.append("docs/packages/unxts.interop.matplotlib/*")
    collect_ignore_glob.append("packages/unxts.interop.matplotlib/docs/*")


def pytest_generate_tests(metafunc: Any) -> None:
    os.environ["UNXT_ENABLE_RUNTIME_TYPECHECKING"] = "beartype.beartype"
