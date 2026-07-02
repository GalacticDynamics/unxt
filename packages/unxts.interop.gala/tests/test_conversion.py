"""Tests for unxts.interop.gala unit-system conversions."""

import astropy.units as apyu
import gala.units as gu
import unxts.interop.gala  # noqa: F401 -- registers the conversion methods
from plum import convert

import unxt as u


def test_gala_to_unxt():
    usys = gu.UnitSystem(apyu.km, apyu.s, apyu.Msun, apyu.radian)
    assert convert(usys, u.AbstractUnitSystem) == u.unitsystem(
        apyu.km, apyu.s, apyu.Msun, apyu.radian
    )


def test_unxt_to_gala():
    usys = u.unitsystem(apyu.km, apyu.s, apyu.Msun, apyu.radian)
    assert convert(usys, gu.UnitSystem) == gu.UnitSystem(
        apyu.km, apyu.s, apyu.Msun, apyu.radian
    )


def test_round_trip():
    usys = gu.galactic
    round_tripped = convert(convert(usys, u.AbstractUnitSystem), gu.UnitSystem)
    # gala's ``UnitSystem.__eq__`` also compares derived/registered units, so
    # compare the base units that the conversion round-trips.
    assert round_tripped._core_units == usys._core_units
