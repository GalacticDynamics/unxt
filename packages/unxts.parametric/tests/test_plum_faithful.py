"""Tests for `plum` related things."""

from plum import is_faithful
from unxts.parametric import ParametricQuantity


def test_is_faithful():
    """Test `ParametricQuantity` is a faithful type.

    See
    https://beartype.github.io/plum/types.html#performance-and-faithful-types

        Plum achieves performance by caching the dispatch process. Unfortunately,
        efficient caching is not always possible. Efficient caching is possible
        for so-called faithful types.

        A type ``t`` is *faithful* if, for all ``x``, the following is true:

        ::

            isinstance(x, t) == issubclass(type(x), t)
    """
    x = ParametricQuantity(1, "m")
    assert isinstance(x, ParametricQuantity)
    assert issubclass(type(x), ParametricQuantity)

    # The previous tests are equivalent to the following:
    assert is_faithful(ParametricQuantity)
