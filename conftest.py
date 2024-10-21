"""Doctest configuration."""

import os
import platform
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE
from typing import Any

from sybil import Sybil
from sybil.parsers.rest import DocTestParser, PythonCodeBlockParser, SkipParser

from optional_dependencies import OptionalDependencyEnum, auto

# TODO: stop skipping doctests on Windows when there is uniform support for
#       numpy 2.0+ scalar repr. On windows it is printed as 1.0 instead of
#       `np.float64(1.0)`.
parsers = (
    [DocTestParser(optionflags=ELLIPSIS | NORMALIZE_WHITESPACE)]
    if platform.system() != "Windows"
    else []
) + [
    PythonCodeBlockParser(),
    SkipParser(),
]

pytest_collect_file = Sybil(
    parsers=parsers,
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
