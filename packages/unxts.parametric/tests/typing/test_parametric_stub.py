"""Static-typing coverage for `unxts/parametric/__init__.pyi`'s declared surface.

Runs as a normal pytest (every call must actually work); also type-checkable
directly with `mypy`/`pyright`/`ty` (this package has no dedicated nox/CI
wiring for that yet, unlike core `unxt`'s `tests/typing` -- see the stub's
own module docstring). A `.pyi` fully shadows its `.py` module for every type
checker, so an omission in the stub does not fall back to the real source --
it makes that name invisible (or wrongly typed) to mypy/pyright/ty alike.
"""

from typing import assert_type

import jax
import pytest
import unxts.parametric as up

# `unxts.parametric._src.config`, not the public `unxts.parametric.config`
# submodule: importing the latter here would register it in `sys.modules` and
# overwrite `up.config` (the *instance* `__init__.py` assigns under the same
# name) with the *module* object -- a real, pre-existing name collision in
# the package (submodule imports always shadow a same-named parent-package
# attribute), unrelated to the stub and out of scope to fix here.
from unxts.parametric._src.config import ParametricConfig

import unxt as u


def test_construction_and_inherited_surface_typecheck_and_run() -> None:
    """`PQ`/`ParametricQuantity` construct and inherit `AbstractQuantity`'s surface."""
    q1 = up.PQ(1.0, "m")
    q2 = up.ParametricQuantity([1.0, 2.0], "m")

    assert_type(q1, up.ParametricQuantity)
    assert_type(q2, up.ParametricQuantity)
    assert q1.unit == u.unit("m")

    # Inherited from `AbstractQuantity` (`unxt/quantity.pyi`): `Self`-typed
    # methods narrow to `ParametricQuantity`, not the declared `AbstractQuantity`.
    assert_type(q1.uconvert("cm"), up.ParametricQuantity)
    assert q1.uconvert("cm").ustrip("cm") == 100.0
    assert_type(q1.shape, tuple[int, ...])


def test_dimension_checking_typecheck_and_run() -> None:
    """`PQ["length"]` runtime-checks the constructor's unit dimension.

    The subscript itself (`PQ["length"]`) is *not* something this stub makes
    mypy accept in a type-checked expression -- see the stub's module
    docstring for why no `Generic` declaration fixes that. This test only
    checks the parts that do type-check: construction, and the checked
    subscript-constructor call wrapped in `# type: ignore` at the one line
    that needs it.
    """
    q = up.PQ["length"](1, "m")  # type: ignore[misc, name-defined]
    assert q.ustrip("m") == 1

    with pytest.raises(ValueError, match="Physical type mismatch"):
        up.PQ["length"](1, "s")  # type: ignore[misc, name-defined]


def test_comparison_dunders_typecheck_and_run() -> None:
    """`==` on `ParametricQuantity` returns a plain (jaxtyping) `Array`.

    Regression guard for the pyright quirk documented on `ParametricQuantity`
    in the stub: it must declare `__eq__`/`__ne__` itself, not just inherit
    them, or pyright infers `q1 == q2` as `bool`.
    """
    q1 = up.PQ([1.0, 2.0], "m")
    q2 = up.PQ([1.0, 3.0], "m")

    assert_type(q1 == q2, jax.Array)
    assert bool((q1 == q2)[0])
    assert not bool((q1 == q2)[1])


def test_config_typecheck_and_run() -> None:
    """`unxts.parametric.config` is a real, typed singleton, not `Any`."""
    assert_type(up.config, ParametricConfig)
    assert hasattr(up.config, "quantity_repr")
