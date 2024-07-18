"""Doctest configuration."""

import platform
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE
from importlib.util import find_spec

from sybil import Sybil
from sybil.parsers.rest import DocTestParser, PythonCodeBlockParser, SkipParser

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


# TODO: via separate optional_deps package
HAS_ASTROPY = find_spec("astropy") is not None
HAS_GALA = find_spec("gala") is not None

collect_ignore_glob = []
if not HAS_ASTROPY:
    collect_ignore_glob.append("src/unxt/_interop/unxt_interop_astropy/*")
if not HAS_GALA:
    collect_ignore_glob.append("src/unxt/_interop/unxt_interop_gala/*")
