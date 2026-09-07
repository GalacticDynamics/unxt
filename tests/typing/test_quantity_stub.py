"""Static-typing coverage for `src/unxt/quantity.pyi`'s full declared surface.

Runs as a normal pytest (every call must actually work) AND under the
`pyright`, `ty`, and `mypy` nox sessions. A `.pyi` fully shadows its `.py`
module for every type checker, so an omission in the stub does not fall back
to the real source -- it makes that name invisible (or wrongly typed) to
mypy/pyright/ty alike. This file is the regression guard for that: each
function below exercises one section of the stub, so a signature that drifts
from the real implementation surfaces here first.

Two top-level names are deliberately *not* exercised via `unxt.<name>`:
`unxt.is_any_quantity` isn't re-exported at the top level at all (only
`unxt.quantity.is_any_quantity` is), and `unxt.equivalent` is the *same*
shared `plum`-dispatched function as `unxt.quantity.equivalent` but is
imported into `unxt/__init__.py` from `unxt.unitsystems`, so mypy only sees
that module's `AbstractUnitSystem`-typed registration at the top level even
though the quantity-typed call also works at runtime. Both are called via
`unxt.quantity` explicitly below to test the stub actually being changed here.
"""

from typing import Any, assert_type

import jax
import numpy as np

import unxt as u
import unxt.quantity as uq


def test_arithmetic_dunders_typecheck_and_run() -> None:
    """Forward ops are `Any`; reflected ops keep the mixin's `AbstractQuantity`.

    `__add__`/`__mul__`/etc. are declared `Any` because they coerce foreign
    astropy operands and can promote across quantity subtypes via `plum` --
    not something a hand-written stub can narrow honestly. `__radd__`/`__rmul__`
    etc. come straight from `quax_blocks.NumpyBinaryOpsMixin[Any, AbstractQuantity]`
    and do have a concrete return type, checked here.
    """
    q1 = u.Quantity(1.0, "m")
    q2 = u.Quantity(2.0, "m")

    # Forward ops: no useful static type to assert (declared `Any`), but the
    # calls must still type-check under `--strict` with no stub-side error.
    assert (q1 + q2).unit == q1.unit
    assert (q1 * q2).unit.to_string() == "m2"

    # Reflected ops: triggered when the left operand doesn't know how to
    # combine with a `Quantity` (a plain `int`/`float`).
    assert_type(2.0 * q1, u.AbstractQuantity)
    assert_type(2.0 + u.Quantity(1.0, ""), u.AbstractQuantity)
    assert (2.0 * q1).ustrip("m") == 2.0

    # Unary ops.
    assert_type(-q1, u.AbstractQuantity)
    assert_type(+q1, u.Quantity)  # `__pos__` is `Self`-typed, unlike `__neg__`.
    assert (-q1).ustrip("m") == -1.0
    assert (+q1).ustrip("m") == 1.0


def test_comparison_dunders_typecheck_and_run() -> None:
    """`==`/`<`/etc. on `Quantity`/`Angle` return a plain (jaxtyping) `Array`.

    `StaticQuantity` overrides `__eq__`/`__ne__` with a looser `Any` (it can
    return a scalar Python `bool` instead), asserted separately below.
    """
    q1 = u.Quantity([1.0, 2.0], "m")
    q2 = u.Quantity([1.0, 3.0], "m")

    assert_type(q1 == q2, jax.Array)
    assert_type(q1 < q2, jax.Array)
    assert bool((q1 == q2)[0])
    assert bool((q1 < q2)[1])

    sq1 = u.StaticQuantity(np.array([1.0, 2.0]), "m")
    sq2 = u.StaticQuantity(np.array([1.0, 2.0]), "m")
    # `StaticQuantity.__eq__` is declared `Any` -- it returns a scalar `bool`
    # for another `StaticQuantity`, unlike the element-wise array above.
    result = sq1 == sq2
    assert result is True


def test_ustrip_uconvert_overloads_typecheck_and_run() -> None:
    """The `ustrip`/`uconvert`/`uconvert_value`/`AllowValue` overload sets."""
    q = u.Quantity(1000.0, "m")

    assert_type(uq.ustrip(q), jax.Array | np.ndarray[Any, Any])
    assert_type(uq.ustrip("km", q), jax.Array | np.ndarray[Any, Any])
    # `u.unit(...)` itself returns `Any` (untyped, a different module), so this
    # overload resolves to `Any` too -- runtime check only, nothing to assert_type.
    assert uq.ustrip(u.unit("km"), q) == 1.0
    assert uq.ustrip("km", q) == 1.0

    assert_type(uq.uconvert("km", q), u.AbstractQuantity)
    assert_type(uq.uconvert(u.unit("km"), q), u.AbstractQuantity)
    assert uq.uconvert("km", q).unit == u.unit("km")

    # `uconvert_value`'s return is `ArrayLike` (a broad jaxtyping union); a
    # runtime check is more useful here than pinning that union's exact spelling.
    assert uq.uconvert_value("m", "km", 1.0) == 1000.0

    # `AllowValue`: a plain (non-quantity) value passes through unchanged.
    # `AllowValue.__new__` is `NoReturn` (it's a pure flag, never instantiated),
    # which makes mypy infer the *bare class reference* as `Callable[[], Never]`
    # instead of `type[AllowValue]` -- a real mypy quirk for uninstantiable
    # classes, not a stub gap (the real runtime class has the same `__new__`).
    x = jax.numpy.array(1.0)
    assert_type(uq.ustrip(uq.AllowValue, x), Any)  # type: ignore[arg-type]
    assert uq.ustrip(uq.AllowValue, x) is x  # type: ignore[arg-type]
    assert_type(uq.ustrip(uq.AllowValue, "km", x), Any)  # type: ignore[arg-type]
    assert uq.ustrip(uq.AllowValue, "km", x) is x  # type: ignore[arg-type]
    # And a quantity is still stripped when it appears.
    assert uq.ustrip(uq.AllowValue, "km", q) == 1.0  # type: ignore[arg-type]


def test_angle_wrap_to_typecheck_and_run() -> None:
    """`Angle.wrap_to` and the functional `unxt.quantity.wrap_to`."""
    angle = u.Angle(370, "deg")
    lo, hi = u.Q(0, "deg"), u.Q(360, "deg")

    assert_type(angle.wrap_to(lo, hi), uq.AbstractAngle)
    assert_type(uq.wrap_to(angle, lo, hi), u.AbstractQuantity)
    assert angle.wrap_to(lo, hi).ustrip("deg") == 10


def test_predicates_typecheck_and_run() -> None:
    """`is_any_quantity` narrows via `TypeGuard`; `is_unit_convertible`/`equivalent`."""
    q = u.Quantity(1.0, "m")
    obj: object = q

    if uq.is_any_quantity(obj):
        assert_type(obj, u.AbstractQuantity)

    assert_type(uq.is_unit_convertible("cm", "m"), bool)
    assert uq.is_unit_convertible("cm", "m")
    assert not uq.is_unit_convertible("s", "m")

    assert uq.equivalent(u.Quantity(1000.0, "m"), u.Quantity(1.0, "km"))


def test_array_like_properties_typecheck_and_run() -> None:
    """Shape/indexing surface shared with plain arrays."""
    q = u.Quantity([1.0, 2.0, 3.0], "m")

    assert_type(q.shape, tuple[int, ...])
    assert_type(q.ndim, int)
    assert_type(q.size, int)
    assert_type(len(q), int)
    assert_type(q[0], u.Quantity)

    assert q.shape == (3,)
    assert q.ndim == 1
    assert q.size == 3
    assert len(q) == 3
    assert q[0].ustrip("m") == 1.0
