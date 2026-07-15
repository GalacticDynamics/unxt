"""JAX primitive rules specific to ``ParametricQuantity`` (registered on import).

These rules only fire when a `ParametricQuantity` (or another
`AbstractParametricQuantity`) is involved, so they live with the parametric
package rather than in core ``unxt``. Core ``unxt`` provides the corresponding
plain-``Quantity`` registrations, so importing this module *adds* the parametric
specialisations without displacing any core behaviour.
"""

__all__: tuple[str, ...] = ()

from dataclasses import replace

import quax
from astropy.units import (
    dimensionless_unscaled as one,  # pylint: disable=no-name-in-module
)
from jax import lax
from jaxtyping import ArrayLike
from plum import type_unparametrized as type_np

from unxts.api import ustrip

from .base_parametric import AbstractParametricQuantity as ABCPQ  # noqa: N814
from .parametric import ParametricQuantity
from unxt.quantity import AbstractQuantity as ABCQ  # noqa: N814

# ==============================================================================
# clamp


@quax.register(lax.clamp_p)
def clamp_p_qvq(
    min: ABCPQ["dimensionless"], x: ArrayLike, max: ABCPQ["dimensionless"]
) -> ArrayLike:
    """Clamp an array between two dimensionless parametric quantities.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import quaxed.lax as lax
    >>> from unxts.parametric import PQ

    >>> min = PQ(0, "")
    >>> max = PQ(2, "")
    >>> x = jnp.asarray([-1, 1, 3])
    >>> lax.clamp(min, x, max)
    Array([0, 1, 2], dtype=int32)

    """
    return lax.clamp(ustrip(one, min), x, ustrip(one, max))


@quax.register(lax.clamp_p)
def clamp_p_qqv(
    min: ABCPQ["dimensionless"], x: ABCPQ["dimensionless"], max: ArrayLike
) -> ABCPQ["dimensionless"]:
    """Clamp a dimensionless parametric quantity between a quantity and a value.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import quaxed.lax as lax
    >>> from unxts.parametric import PQ

    >>> min = PQ(0, "")
    >>> max = jnp.asarray(2)
    >>> q = PQ([-1, 1, 3], "")
    >>> lax.clamp(min, q, max)
    ParametricQuantity(Array([0, 1, 2], dtype=int32), unit='')

    """
    return replace(x, value=lax.clamp(ustrip(one, min), ustrip(one, x), max))


# ==============================================================================
# pow


@quax.register(lax.pow_p)
def pow_p_qq(x: ABCQ, y: ABCPQ["dimensionless"], /) -> ABCQ:
    """Power of a quantity by a (dimensionless) parametric-quantity exponent.

    The exponent must be a (dimensionless) parametric quantity to select this
    registration; a plain ``Quantity`` exponent is handled by core ``unxt``.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> from unxts.parametric import PQ

    >>> q1 = u.Q(2.0, "m")
    >>> p = PQ(3, "")
    >>> jnp.power(q1, p)
    Quantity(Array(8., dtype=float32...), unit='m3')
    >>> q1**p
    Quantity(Array(8., dtype=float32...), unit='m3')

    Non-scalar exponents raise a ValueError:

    >>> p_arr = PQ([3, 2], "")
    >>> try:
    ...     q1**p_arr
    ... except ValueError as e:
    ...     print(e)
    Exponent must be a scalar.

    """
    yv = ustrip(one, y)
    y0 = yv[()]
    if y0.ndim != 0:
        msg = "Exponent must be a scalar."
        raise ValueError(msg)
    return type_np(x)(value=lax.pow(ustrip(x), y0), unit=x.unit**y0)


@quax.register(lax.pow_p)
def pow_p_vq(x: ArrayLike, y: ABCPQ["dimensionless"], /) -> ABCQ:
    """Array raised to a (dimensionless) parametric-quantity exponent.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxts.parametric import PQ

    >>> x = jnp.array([2.0])
    >>> p = PQ(3, "")
    >>> jnp.power(x, p)
    ParametricQuantity(Array([8.], dtype=float32), unit='')

    """
    return replace(y, value=lax.pow(x, ustrip(y)))


# ==============================================================================
# rem


@quax.register(lax.rem_p)
def rem_p_uqv(
    x: ParametricQuantity["dimensionless"], y: ArrayLike, /
) -> ParametricQuantity["dimensionless"]:
    """Remainder of a dimensionless parametric quantity and an array.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from unxts.parametric import PQ

    >>> q1 = PQ(10, "")
    >>> q2 = jnp.array(3)
    >>> q1 % q2
    ParametricQuantity(Array(1, dtype=int32...), unit='')

    """
    return replace(x, value=ustrip(x) % y)
