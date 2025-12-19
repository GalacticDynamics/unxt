"""Unit strategies."""

__all__ = ("derived_units", "units")

from typing import Any

import astropy.units as apyu
from hypothesis import strategies as st

import unxt as u

SI_DIMS_STRAT = st.sampled_from(tuple(u.dimension_of(x) for x in u.unitsystems.si))


@st.composite
def derived_units(  # pylint: disable=unreachable
    draw: st.DrawFn,
    base: str | u.AbstractUnit | st.SearchStrategy[u.AbstractUnit],
    /,
    *,
    integer_powers: bool = True,
    max_complexity: int = 3,
) -> u.AbstractUnit:
    """Generate hypothesis strategy for units from a given base unit.

    This strategy generates units that are dimensionally equivalent to the input
    base unit by combining the base unit's base units and discovered composed
    forms.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function (automatically provided by @st.composite).
    base : str | u.AbstractUnit | st.SearchStrategy[u.AbstractUnit]
        Base unit (e.g., "m", "s", "kg") or a `hypothesis` strategy that
        generates such units.
    integer_powers : bool, optional
        If True, only generate units with integer powers of base units.  Default
        is True.
    max_complexity : int, optional
        Maximum number of additional base unit factors to combine.  Higher
        values create more complex compound units. Default is 3.

    Returns
    -------
    `unxt.AbstractUnit`
        A unit with the same physical dimension as the input.

    Examples
    --------
    >>> import hypothesis.strategies as st
    >>> from hypothesis import given
    >>> import unxt_hypothesis as ust
    >>> import unxt as u

    Generate units derived from a base unit:

    >>> @given(unit=ust.derived_units("m"))
    ... def test_length_units(unit):
    ...     assert u.dimension_of(unit) == u.dimension("length")

    Generate units from a strategy:

    >>> @given(unit=ust.derived_units(st.sampled_from(["km", "m/s^2"])))
    ... def test_velocity_derived(unit):
    ...     assert u.dimension_of(unit) in (
    ...         u.dimension("velocity"),
    ...         u.dimension("acceleration"),
    ...     )

    Control complexity of generated units:

    >>> @given(unit=ust.derived_units("kg", max_complexity=1))
    ... def test_simple_mass_units(unit):
    ...     assert u.dimension_of(unit) == u.dimension("mass")

    Allow non-integer powers:

    >>> @given(unit=ust.derived_units("m", integer_powers=False))
    ... def test_fractional_powers(unit):
    ...     dim = u.dimension_of(unit)
    ...     assert dim == u.dimension("length")

    """
    # Convert base to a unit if it's a string or draw from strategy
    base_unit = u.unit(draw(base) if isinstance(base, st.SearchStrategy) else base)

    # Collect all possible unit forms
    # 1. Start with the base unit
    candidates = [base_unit]

    # 2. Add composed forms (limited to first 20 to avoid huge search space)
    composed = list(base_unit.compose())[:20]
    candidates.extend(composed)

    # 3. Optionally create compound units by combining cancelling factors
    if max_complexity > 0:
        # Get SI base decomposition
        decomposed = base_unit.decompose()
        si_bases = list(decomposed.bases)
        list(decomposed.powers)

        # Get a pool of other SI base units to use as cancelling factors
        all_si_bases = [
            apyu.Unit(x) for x in ("m", "s", "kg", "A", "K", "mol", "cd", "rad")
        ]

        # Find SI bases not used in this dimension (for pure cancellation)
        # and those that are used (for modification)
        available_bases = [b for b in all_si_bases if b not in si_bases]

        if available_bases:
            # Generate a few compound units with cancelling factors
            num_extra = draw(st.integers(min_value=0, max_value=max_complexity))

            if num_extra > 0:
                # Choose random bases to combine
                extra_bases = draw(
                    st.lists(
                        st.sampled_from(available_bases),
                        min_size=num_extra,
                        max_size=num_extra,
                        unique=True,
                    )
                )

                # For each base, generate a power and its inverse
                compound = base_unit
                for extra_base in extra_bases:
                    if integer_powers:
                        power = draw(st.integers(min_value=1, max_value=3))
                    else:
                        power = draw(st.floats(min_value=0.5, max_value=3.0))

                    # Multiply and divide to cancel dimensions
                    compound = compound * (extra_base**power) / (extra_base**power)

                candidates.append(compound)

    # Choose one of the candidate units
    return draw(st.sampled_from(candidates))


@st.composite
def units(
    draw: st.DrawFn,
    dimension: str
    | u.AbstractDimension
    | st.SearchStrategy[u.AbstractDimension] = SI_DIMS_STRAT,
    /,
    **kwargs: Any,
) -> u.AbstractUnit:
    """Generate hypothesis strategy for units with a given physical dimension.

    This strategy generates units that are dimensionally equivalent to the input
    dimension by combining the dimension's base units and discovered composed
    forms.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function (automatically provided by @st.composite).
    dimension : str | apyu.PhysicalType
        Physical dimension (e.g., "velocity", "length") or a `hypothesis`
        strategy that generates such dimensions. Default is a strategy sampling
        from SI base dimensions.
    **kwargs : Any
        Additional keyword arguments passed to `derived_units()`.
        For example:

        - integer_powers : bool
            If True, only generate units with integer powers of base units.  Default
            is True.
        - max_complexity : int
            Maximum number of additional base unit factors to combine.  Higher
            values create more complex compound units. Default is 3.

    Returns
    -------
    unxt.AbstractUnit
        A unit with the same physical dimension as the input.

    Examples
    --------
    >>> import hypothesis.strategies as st
    >>> from hypothesis import given
    >>> import unxt_hypothesis as ust
    >>> import unxt as u

    >>> @given(unit=ust.units("velocity"))
    ... def test_velocity_units(unit):
    ...     assert u.dimension_of(unit) == u.dimension("velocity")

    >>> @given(unit=ust.units("length", integer_powers=False))
    ... def test_length_units(unit):
    ...     dim = u.dimension_of(unit)
    ...     assert dim == u.dimension("length")

    The dimension argument can also be a strategy:

    >>> from hypothesis import strategies as st
    >>> @given(unit=ust.units(st.sampled_from(["length", "time", "mass"])))
    ... def test_multiple_dimensions(unit):
    ...     dim = u.dimension_of(unit)
    ...     assert dim in [
    ...         u.dimension("length"),
    ...         u.dimension("time"),
    ...         u.dimension("mass"),
    ...     ]

    Notes
    -----
    The strategy works by:

    1. Getting the canonical unit for the dimension
    2. Decomposing it into base SI units
    3. Finding composed forms using `.compose()`
    4. Generating random combinations by multiplying/dividing
       other units with cancelling dimensions

    """
    # Convert to Dimension object, drawing if necessary
    dimension = (
        u.dimension(draw(dimension))
        if isinstance(dimension, st.SearchStrategy)
        else u.dimension(dimension)
    )

    # Get the canonical unit for this dimension
    base_unit = dimension._unit  # noqa: SLF001

    return draw(derived_units(base_unit, **kwargs))
