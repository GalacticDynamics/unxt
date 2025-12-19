"""Hypothesis strategies for UnitSystem objects."""

__all__ = ["unitsystems"]

from hypothesis import strategies as st

import unxt as u


@st.composite
def unitsystems(
    draw: st.DrawFn,
    *units: str | u.AbstractUnit | st.SearchStrategy[u.AbstractUnit],
) -> u.AbstractUnitSystem:
    """Generate hypothesis strategy for unxt UnitSystem objects.

    This strategy supports all the same unit input options as `u.unitsystem()`.
    The additional capability provided here is that any unit input can also be a
    hypothesis strategy, enabling property-based testing with varying units.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function (automatically provided by @st.composite).
    *units : str | unxt.AbstractUnit | st.SearchStrategy[u.AbstractUnit]
        Variable number of unit specifications. Each can be:
        - str: Fixed unit string (e.g., "kpc", "Myr", "Msun")
        - unxt.AbstractUnit: Fixed unit object
        - SearchStrategy: Strategy that generates units

    Returns
    -------
    u.AbstractUnitSystem
        A UnitSystem object created from the specified units.

    Examples
    --------
    >>> from hypothesis import given
    >>> import unxt as u
    >>> import unxt_hypothesis as ust

    Create a unit system with fixed units:

    >>> @given(usys=ust.unitsystems("m", "s", "kg", "rad"))
    ... def test_mks_system(usys):
    ...     assert isinstance(usys, u.AbstractUnitSystem)
    ...     assert len(usys) == 4

    Create a unit system with varying length units:

    >>> @given(usys=ust.unitsystems(ust.units("length"), "s", "kg", "rad"))
    ... def test_varying_length_system(usys):
    ...     assert isinstance(usys, u.AbstractUnitSystem)
    ...     assert len(usys) == 4
    ...     # Length unit will vary, but time is always seconds
    ...     assert usys["time"] == u.unit("s")

    Use predefined unit systems by name:

    >>> @given(usys="galactic")  # Predefined
    ... def test_galactic_system(usys):
    ...     assert isinstance(usys, u.AbstractUnitSystem)
    ...     assert len(usys) == 4

    """
    # Process each unit argument
    unit_objs = [
        u.unit(draw(x) if isinstance(x, st.SearchStrategy) else x) for x in units
    ]
    # Create and return the unit system
    return u.unitsystem(*unit_objs)
