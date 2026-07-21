"""Static-typing smoke test: quantity constructors accept a unit string.

Runs as a normal pytest (the constructors must not raise) AND under the
`pyright` nox session (the string-unit argument must type-check). Regression
guard for the plum-dispatch converter that pyright reads as opaque; see
`unxt._src.units.parse_unit`.

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
    guard); the constructor calls also exercise runtime so this doubles as a
    smoke test.
    """
    # The README's first line -- and the bug this guards.
    assert_type(u.Quantity(1, "m"), u.Quantity)

    u.Quantity(1, "m")
    u.Q(1.0, "m")
    u.Angle(1, "rad")

    # A real unit object is still accepted. (``u.unit("m")`` itself is typed
    # ``object`` by pyright -- the public ``unit`` dispatch is opaque -- so its
    # return value is deliberately not exercised here; improving ``unit()``'s
    # return type is a separate follow-up.)
    u.Quantity(1, apyu.Unit("m"))

    # The ``unit`` field reads back as a unit, not ``str``.
    assert_type(u.Quantity(1, "m").unit, u.AbstractUnit)
