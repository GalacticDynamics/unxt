"""Doctest configuration."""

import platform
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

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
