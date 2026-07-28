"""Static-typing smoke test: quantity constructors accept a unit string.

Runs as a normal pytest (the constructors must not raise) AND under the
`pyright`, `ty`, and `mypy` nox sessions (the string-unit argument must
type-check).
Regression guard for the `unit`-field converter: the quantity classes use the
``unit`` API function (liberal ``Any`` input, Postel's law) as their
``eqx.field`` converter, so ``Quantity(1, "m")`` type-checks while ``.unit``
still reads as a unit.

pyright and ty both read equinox's `converter` field-specifier, so both
validate the full constructor: the str `unit` and the raw-int `value` are
accepted and `.unit` reads back as `AbstractUnit`. mypy does less: it types the
`value` field as `Array | StaticValue` (rejecting a raw int/float -- suppressed
below) and the `unit` param / `.unit` result as `Any`, so it neither
discriminates the unit argument nor checks `.unit`. What mypy *does* guard is
that the constructors type-check under the strict config and return the right
type (`assert_type(Quantity(1, "m"), Quantity)` is a real check -- a return-type
regression fails it). A regression here is the trigger to revisit a `.pyi` stub.

The `# type: ignore[arg-type]` on each constructor marks mypy's `value`-converter
gap per call site (rather than a blanket module-level disable), so any *other*
`arg-type` error the fixture grows still surfaces; `warn_unused_ignores` flags
them if the gap ever closes. pyright/ty apply the converter and need no ignore.
"""

from typing import assert_type

import astropy.units as apyu

import unxt as u


def test_string_unit_constructors_typecheck_and_run() -> None:
    """A unit string is accepted by the quantity constructors.

    The ``assert_type`` calls are checked statically by pyright (the actual
    guard); the runtime assertions make it a real behavioural test too -- the
    string unit must be parsed to the expected unit.
    """
    # The README's first line -- and the bug this guards.
    assert_type(u.Quantity(1, "m"), u.Quantity)  # type: ignore[arg-type]

    # ``Angle`` shares the same ``unit``-field converter. StaticQuantity does
    # too, but its *value* field is separately opaque to pyright (it types
    # ``value: StaticValue``, rejecting ``Literal[1]``), so a smoke line for it
    # would fail pyright for a reason unrelated to this unit-field fix -- a
    # separate follow-up. ParametricQuantity lives in the ``unxts.parametric``
    # package and is guarded by that package's own suite.
    assert_type(u.Angle(1, "rad"), u.Angle)  # type: ignore[arg-type]

    # Runtime: the string unit is parsed to the right unit on each constructor.
    assert u.Quantity(1, "m").unit == apyu.Unit("m")  # type: ignore[arg-type]
    assert u.Q(1.0, "m").unit == apyu.Unit("m")  # type: ignore[arg-type]
    assert u.Angle(1, "rad").unit == apyu.Unit("rad")  # type: ignore[arg-type]

    # A real unit object is still accepted and round-trips.
    assert u.Quantity(1, apyu.Unit("m")).unit == apyu.Unit("m")  # type: ignore[arg-type]

    # The ``unit`` field reads back as a unit, not ``str``.
    assert_type(u.Quantity(1, "m").unit, u.AbstractUnit)  # type: ignore[arg-type]
