"""Static-typing smoke test: quantity constructors accept a unit string.

Runs as a normal pytest (the constructors must not raise) AND under the
`pyright` nox session (the string-unit argument must type-check). Regression
guard for the plum-dispatch converter that pyright reads as opaque; see
`unxt.units.parse_unit`.

Pyright-scoped: mypy / ty do not implement the `converter` field-specifier
extension, so they will not validate this. A mypy/ty regression is the trigger
to revisit a `.pyi` stub.
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
    assert_type(u.Quantity(1, "m"), u.Quantity)

    # ``Angle`` shares the same ``unit``-field converter. StaticQuantity does
    # too, but its *value* field is separately opaque to pyright (it types
    # ``value: StaticValue``, rejecting ``Literal[1]``), so a smoke line for it
    # would fail pyright for a reason unrelated to this unit-field fix -- a
    # separate follow-up. ParametricQuantity lives in the ``unxts.parametric``
    # package and is guarded by that package's own suite.
    assert_type(u.Angle(1, "rad"), u.Angle)

    # Runtime: the string unit is parsed to the right unit on each constructor.
    assert u.Quantity(1, "m").unit == apyu.Unit("m")
    assert u.Q(1.0, "m").unit == apyu.Unit("m")
    assert u.Angle(1, "rad").unit == apyu.Unit("rad")

    # A real unit object is still accepted and round-trips.
    assert u.Quantity(1, apyu.Unit("m")).unit == apyu.Unit("m")

    # The ``unit`` field reads back as a unit, not ``str``.
    assert_type(u.Quantity(1, "m").unit, u.AbstractUnit)
