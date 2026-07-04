"""Parametric ``dimension_of`` overload (registered on import)."""

__all__: tuple[str, ...] = ()

import equinox as eqx

from unxt.dims import AbstractDimension, dimension_of

from .base_parametric import AbstractParametricQuantity


@dimension_of.dispatch
def dimension_of(
    obj: type[AbstractParametricQuantity], /
) -> AbstractDimension:
    """Return the dimension of a parametrized quantity type.

    Examples
    --------
    >>> import unxt as u
    >>> import unxts.parametric as up

    >>> try:
    ...     u.dimension_of(up.PQ)
    ... except Exception as e:
    ...     print(e)
    can only get dimensions from parametrized ParametricQuantity --
    ParametricQuantity[dim].

    >>> u.dimension_of(up.PQ["length"])
    PhysicalType('length')

    """
    obj = eqx.error_if(
        obj,
        not hasattr(obj, "_type_parameter"),
        f"can only get dimensions from parametrized {obj.__name__} "
        f"-- {obj.__name__}[dim].",
    )
    return obj._type_parameter  # noqa: SLF001
