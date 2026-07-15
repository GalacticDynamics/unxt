"""Doctest configuration."""

import importlib.util
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
    """External backends for ``unxt``.

    Only genuine third-party backends belong here. ``OptionalDependencyEnum``
    keys each member on its installed version, so any two members that share a
    version silently collapse into a single enum alias. unxt's own ``unxts.*``
    sub-packages are released together and so usually share a version (bug-fix
    releases can make them diverge), so their presence is checked with
    ``_is_installed`` (an import-spec lookup via ``find_spec``) instead (see
    ``unxt._interop.optional_deps`` for the same reasoning).
    """

    ASTROPY = auto()
    GALA = auto()


def _is_installed(module: str) -> bool:
    """Return whether ``module`` has a discoverable import spec.

    Uses :func:`importlib.util.find_spec` (reports installed / findable, not that
    importing would succeed). Defined locally (rather than importing the
    equivalent helper from ``unxt``) so that ``conftest`` never imports ``unxt``
    before ``pytest_generate_tests`` sets ``UNXT_ENABLE_RUNTIME_TYPECHECKING``.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


collect_ignore_glob = []
if not OptDeps.ASTROPY.installed:
    collect_ignore_glob.append("src/unxt/_interop/unxt_interop_astropy/*")
# The package docs are collected through the docs/packages/<name> symlinks, so
# ignore that (symlink) path, not the real package path.
# `unxts.interop.gala` is an optional extra, so it may be absent; and even when
# it is present its `gala` dependency may be unimportable (gala is skipped where
# it cannot build, e.g. Windows). Ignore its docs unless both are available.
if not (_is_installed("unxts.interop.gala") and OptDeps.GALA.installed):
    collect_ignore_glob.append("docs/packages/unxts.interop.gala/*")
    collect_ignore_glob.append("packages/unxts.interop.gala/docs/*")
if not _is_installed("unxts.interop.matplotlib"):
    collect_ignore_glob.append("docs/packages/unxts.interop.matplotlib/*")
    collect_ignore_glob.append("packages/unxts.interop.matplotlib/docs/*")
if not _is_installed("unxts.linalg"):
    collect_ignore_glob.append("docs/packages/unxts.linalg/*")
    collect_ignore_glob.append("packages/unxts.linalg/docs/*")
if not _is_installed("unxts.parametric"):
    collect_ignore_glob.append("docs/packages/unxts.parametric/*")
    collect_ignore_glob.append("packages/unxts.parametric/docs/*")


def pytest_generate_tests(metafunc: Any) -> None:
    os.environ["UNXT_ENABLE_RUNTIME_TYPECHECKING"] = "beartype.beartype"
