"""Comparison functions."""

__all__ = ("equivalent",)

from plum import dispatch

from .base import AbstractUnitSystem


@dispatch
def equivalent(a: AbstractUnitSystem, b: AbstractUnitSystem, /) -> bool:
    """Whether two unit systems describe the same physical structure.

    Two unit systems are equivalent when they share the same set of base
    dimensions **and**, for each dimension, their units are interconvertible.
    Unlike ``==``, the specific units need not be identical -- only
    convertible -- so systems that measure the same dimensions in different
    (but compatible) units are equivalent.

    Examples
    --------
    >>> from unxt.unitsystems import galactic, solarsystem, si, equivalent

    ``galactic`` (kpc, Myr, solMass, rad) and ``solarsystem`` (AU, yr, solMass,
    rad) have the same base dimensions and interconvertible units:

    >>> equivalent(galactic, solarsystem)
    True

    >>> equivalent(galactic, galactic)
    True

    A system with different base dimensions is not equivalent:

    >>> equivalent(galactic, si)
    False

    """
    dims = set(a.base_dimensions)
    if dims != set(b.base_dimensions):
        return False
    # Same dimensions: require each dimension's units to be interconvertible.
    return all(a[d].is_equivalent(b[d]) for d in dims)
