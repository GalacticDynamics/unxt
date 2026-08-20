# pylint: disable=import-error

"""Test the `_ANGULAR_UNITS` memoisation cache in `AbstractAngle.__check_init__`."""

import contextlib
import gc

import astropy.units as apyu

import unxt as u
from unxt._src.quantity.base_angle import _ANGULAR_UNITS


def test_non_angular_unit_is_never_cached():
    for _ in range(2):
        with contextlib.suppress(ValueError):
            u.Angle(1.0, "m")
    assert apyu.m not in _ANGULAR_UNITS


def _scaled_radian() -> apyu.UnitBase:
    return apyu.CompositeUnit(2, [apyu.rad], [1])  # not interned by astropy


def test_angular_unit_cache_is_weak():
    unit = _scaled_radian()
    u.Angle(1.0, unit)
    assert unit in _ANGULAR_UNITS

    del unit
    gc.collect()

    assert _scaled_radian() not in _ANGULAR_UNITS
