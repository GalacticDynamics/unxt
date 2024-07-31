"""Tests for :mod:`plum` related things."""

from plum import is_faithful

from unxt import Quantity


def test_is_faithful():
    """Test :class:`unxt.Quantity` is a faithful type.

    See
    https://beartype.github.io/plum/types.html#performance-and-faithful-types

        Plum achieves performance by caching the dispatch process. Unfortunately,
        efficient caching is not always possible. Efficient caching is possible
        for so-called faithful types.

        A type ``t`` is *faithful* if, for all ``x``, the following is true:

        ::

            isinstance(x, t) == issubclass(type(x), t)
    """
    x = Quantity(1, "m")
    assert isinstance(x, Quantity)
    assert issubclass(type(x), Quantity)

    # The previous tests are equivalent to the following:
    assert is_faithful(Quantity)
