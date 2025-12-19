"""Hypothesis strategies for Quantity objects."""

__all__ = ["quantities", "wrap_to"]

from typing import Any

import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.array_api import make_strategies_namespace

import unxt as u
from .units import units

# Create array API strategies namespace for JAX
xps = make_strategies_namespace(jnp)


SI_UNITS_STRAT = st.sampled_from(tuple(u.unit(x) for x in u.unitsystems.si))


@st.composite
def quantities(
    draw: st.DrawFn,
    /,
    unit: str
    | u.AbstractUnit
    | st.SearchStrategy[u.AbstractUnit | u.AbstractDimension]
    | u.AbstractDimension = SI_UNITS_STRAT,
    *,
    quantity_cls: type[u.AbstractQuantity] = u.Quantity,
    dtype: Any | st.SearchStrategy[np.dtype] = jnp.float32,
    shape: int | tuple[int, ...] | st.SearchStrategy[tuple[int, ...]] | None = None,
    elements: st.SearchStrategy[float] | None = None,
    unique: bool = False,
    **kwargs: Any,
) -> u.AbstractQuantity:
    """Generate hypothesis strategy for unxt Quantity objects.

    This strategy combines JAX array generation with unit specifications to
    create valid Quantity objects for property-based testing.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function (automatically provided by @st.composite).
    unit : str | st.SearchStrategy[unxt.AbstractUnit]
        Unit specification for the Quantity. Can be: - str: Fixed unit (e.g.,
        "kpc", "km/s") - unxt.AbstractUnit: Fixed unit object -
        unxt.AbstractDimension: Dimension to derive unit from (uses
        `unxt_hypothesis.units` strategy) - SearchStrategy: Strategy that
        generates units (e.g., from `units()`) or dimensions.  The default
        strategy samples from SI base units.
    quantity_cls : type[unxt.AbstractQuantity], optional
        The target quantity class to convert to. Default is unxt.Quantity.  Can
        be any AbstractQuantity subclass like Quantity or Angle.
    dtype : Any, optional
        NumPy/JAX dtype for the underlying array. Default is jnp.float32.  Can
        also be a SearchStrategy that generates dtypes.
    shape : int | tuple[int, ...] | st.SearchStrategy[tuple[int, ...]] | None
        Shape of the array. Can be: - int: 1D array of that length - tuple:
        fixed shape - SearchStrategy: strategy that generates shapes - None:
        scalar (shape ())
    elements : st.SearchStrategy[float] | None, optional
        Strategy for generating array elements. If None, uses finite floats.
    unique : bool, optional
        Whether array elements should be unique. Default is False.
    **kwargs : Any
        Additional keyword arguments (currently unused, reserved for future
        use).

    Returns
    -------
    u.Quantity
        A Quantity object with the specified unit and array properties.

    Examples
    --------
    Basic usage with fixed unit and shape:

    >>> from hypothesis import given, strategies as st
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import unxt_hypothesis as ust

    >>> @given(q=ust.quantities("kpc", shape=3))
    ... def test_position(q):
    ...     assert q.unit == u.unit("kpc")
    ...     assert q.shape == (3,)

    >>> @given(q=ust.quantities("km/s", shape=()))
    ... def test_scalar_velocity(q):
    ...     assert q.unit == u.unit("km/s")
    ...     assert q.shape == ()

    Using a unit strategy to vary units across test runs:

    >>> @given(q=ust.quantities(ust.units("length"), shape=3))
    ... def test_length_with_varying_units(q):
    ...     # Unit will vary across test runs
    ...     assert u.dimension_of(q) == u.dimension("length")
    ...     assert q.shape == (3,)

    As a convenience, instead of nesting strategies with ``ust.units``, you can
    directly pass the dimension object (not a string) to ``ust.quantities``:

    >>> @given(q=ust.quantities(u.dimension("velocity"), shape=3))
    ... def test_velocity_from_dimension(q):
    ...     # Will generate different velocity units (m/s, km/s, etc.)
    ...     assert u.dimension_of(q) == u.dimension("velocity")
    ...     assert q.shape == (3,)

    This (and ``ust.units``) support a strategy for dimensions:

    >>> dim_strat = st.sampled_from([u.dimension("length"), u.dimension("mass")])
    >>> @given(q=ust.quantities(dim_strat, shape=()))
    ... def test_mixed_dimensions(q):
    ...     # Will generate either length or mass quantities
    ...     dim = u.dimension_of(q)
    ...     assert dim in (u.dimension("length"), u.dimension("mass"))

    Using dtype as a strategy:

    >>> dtype_strat = st.sampled_from([jnp.float32, jnp.float64])
    >>> @given(q=ust.quantities("m", dtype=dtype_strat))
    ... def test_varying_dtype(q):
    ...     # dtype will vary between float32 and float64
    ...     assert q.dtype in (jnp.float32, jnp.float64)

    Using elements to constrain value ranges for distances (positive values):

    >>> @given(
    ...     q=ust.quantities(
    ...         "kpc", shape=3, elements=st.floats(min_value=0, max_value=100, width=32)
    ...     )
    ... )
    ... def test_positive_distance(q):
    ...     # All elements will be positive (suitable for distances)
    ...     assert jnp.all(q.value >= 0)
    ...     assert jnp.all(q.value <= 100)

    Using elements for longitude angles (0 to 360 degrees):

    >>> @given(
    ...     q=ust.quantities(
    ...         "deg",
    ...         shape=(),
    ...         elements=st.floats(min_value=0, max_value=360, width=32),
    ...     )
    ... )
    ... def test_longitude_range(q):
    ...     # Longitude in [0, 360] degrees
    ...     assert 0 <= q.value <= 360

    Using quantity_cls to generate Angle objects:

    >>> @given(angle=ust.quantities("rad", quantity_cls=u.Angle, shape=3))
    ... def test_angle_generation(angle):
    ...     # Creates Angle instances instead of Quantity
    ...     assert isinstance(angle, u.Angle)
    ...     assert angle.unit == u.unit("rad")
    ...     assert angle.shape == (3,)

    """
    # Handle unit specification - draw from strategy if needed
    if isinstance(unit, st.SearchStrategy):
        unit = draw(unit)
    unit_obj = (
        draw(units(unit)) if isinstance(unit, u.AbstractDimension) else u.unit(unit)
    )

    # Handle shape specification
    if shape is None:
        array_shape = ()
    elif isinstance(shape, int):
        array_shape = (shape,)
    elif isinstance(shape, tuple):
        array_shape = shape
    else:
        # It's a strategy, draw from it
        array_shape = draw(shape)

    # DType handling
    dtype = draw(dtype) if isinstance(dtype, st.SearchStrategy) else dtype

    # Default elements strategy if not provided
    if elements is None:
        # Use xps.from_dtype to get the appropriate strategy for the dtype
        elements = xps.from_dtype(dtype, allow_nan=False, allow_infinity=False)

    # Generate array using JAX array-api strategy
    # For arrays, use the JAX array API strategies
    values = draw(
        xps.arrays(dtype=dtype, shape=array_shape, elements=elements, unique=unique)
    )

    # Create Quantity with the specified unit
    return quantity_cls(values, unit_obj, **kwargs)


@st.composite
def wrap_to(
    draw: st.DrawFn,
    quantity: st.SearchStrategy[u.AbstractQuantity],
    /,
    min: u.AbstractQuantity | st.SearchStrategy[u.AbstractQuantity],
    max: u.AbstractQuantity | st.SearchStrategy[u.AbstractQuantity],
) -> u.AbstractQuantity:
    """Generate hypothesis strategy for wrapped quantities.

    This strategy takes a quantity strategy and wraps the generated values
    to the specified [min, max) range using the wrap_to function.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function (automatically provided by @st.composite).
    quantity : st.SearchStrategy[unxt.AbstractQuantity]
        Strategy that generates the base quantity to wrap.
    min : unxt.AbstractQuantity | st.SearchStrategy[unxt.AbstractQuantity]
        Minimum value of the wrapping range (inclusive).
    max : unxt.AbstractQuantity | st.SearchStrategy[unxt.AbstractQuantity]
        Maximum value of the wrapping range (exclusive).

    Returns
    -------
    unxt.AbstractQuantity
        The wrapped quantity in the range [min, max).

    Examples
    --------
    Wrap angles to 0-360 degree range:

    >>> from hypothesis import given
    >>> import hypothesis.strategies as st
    >>> import unxt as u
    >>> import unxt_hypothesis as ust

    >>> @given(
    ...     angle=ust.wrap_to(
    ...         ust.quantities("deg", quantity_cls=u.Angle),
    ...         min=u.Quantity(0, "deg"),
    ...         max=u.Quantity(360, "deg"),
    ...     )
    ... )
    ... def test_wrapped_angle(angle):
    ...     assert 0 <= angle.value <= 360

    Wrap angles with dynamic min/max:

    >>> @given(
    ...     angle=ust.wrap_to(
    ...         ust.quantities("rad", quantity_cls=u.Angle),
    ...         min=st.just(u.Quantity(0, "rad")),
    ...         max=st.just(u.Quantity(6.28318530718, "rad")),
    ...     )
    ... )
    ... def test_wrapped_angle_rad(angle):
    ...     assert 0 <= angle.value <= 6.28318530718

    """
    # Draw the base quantity
    q = draw(quantity)

    # Draw min/max if they're strategies
    min_val = draw(min) if isinstance(min, st.SearchStrategy) else min
    max_val = draw(max) if isinstance(max, st.SearchStrategy) else max

    # Wrap the quantity to the specified range
    return u.quantity.wrap_to(q, min_val, max_val)
