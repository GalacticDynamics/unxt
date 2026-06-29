"""Doctest configuration."""

import os
import re
from collections.abc import Callable, Iterable, Sequence
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE, DocTestRunner, Example, OutputChecker
from typing import Any

from sybil import Document, Region, Sybil
from sybil.evaluators.doctest import DocTestEvaluator
from sybil.evaluators.python import PythonEvaluator
from sybil.parsers import myst
from sybil.parsers.abstract.codeblock import PythonDocTestOrCodeBlockParser
from sybil.parsers.abstract.doctest import DocTestStringParser

from optional_dependencies import OptionalDependencyEnum, auto

optionflags = ELLIPSIS | NORMALIZE_WHITESPACE
_JAX_SCALAR_ARRAY_ELLIPSIS_RE = re.compile(
    r"(Array\(.*?dtype=[^,\n)]+), \.\.\.(\))"
)


def _normalize_jax_repr(text: str) -> str:
    """Normalize unstable JAX ``Array(...)`` repr metadata for doctests.

    JAX may optionally include ``weak_type=True`` for scalar arrays, and some
    scalar examples use ``...,`` after the dtype to allow for extra repr
    metadata. Both details vary across JAX and Python versions but do not
    affect behavior.
    """
    text = text.replace(", weak_type=True", "")
    # Normalize ``dtype=float32, ...)`` to ``dtype=float32...)`` so doctest
    # ellipsis matching works whether extra scalar repr metadata is present or
    # absent in a given JAX/Python combination.
    return _JAX_SCALAR_ARRAY_ELLIPSIS_RE.sub(r"\1...\2", text)


class JaxAwareOutputChecker(OutputChecker):
    """Output checker that ignores unstable JAX repr metadata."""

    def check_output(self, want: str, got: str, optionflags: int) -> bool:
        return super().check_output(
            _normalize_jax_repr(want), _normalize_jax_repr(got), optionflags
        )

    def output_difference(self, example: Example, got: str, optionflags: int) -> str:
        return super().output_difference(
            Example(
                source=example.source,
                want=_normalize_jax_repr(example.want),
                exc_msg=example.exc_msg,
                lineno=example.lineno,
                indent=example.indent,
                options=example.options,
            ),
            _normalize_jax_repr(got),
            optionflags,
        )


class JaxAwareDocTestEvaluator(DocTestEvaluator):
    """Sybil doctest evaluator with JAX-aware output normalization."""

    def __init__(self, optionflags: int = 0) -> None:
        self.runner = DocTestRunner(
            optionflags=optionflags, checker=JaxAwareOutputChecker()
        )


class PlainDocTestParser:
    """Parser for plain >>> doctests in Python docstrings."""

    def __init__(self, doctest_optionflags: int = 0) -> None:
        self.doctest_parser = DocTestStringParser(
            JaxAwareDocTestEvaluator(doctest_optionflags)
        )

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
        self.doctest_parser = DocTestStringParser(
            JaxAwareDocTestEvaluator(doctest_optionflags)
        )
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
