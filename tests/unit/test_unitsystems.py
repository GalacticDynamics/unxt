"""Test the `unxt.unitsystems` module."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from unxt.unitsystems import AbstractUnitSystem, dimensionless, unitsystem

# ===================================================================


def test_unitsystem_constructor() -> None:
    """Test the :class:`~unxt.UnitSystem` constructor."""
    usys = unitsystem(5 * u.kpc, 50 * u.Myr, 1e5 * u.Msun, u.rad)
    assert np.isclose((8 * u.Myr).decompose(usys).value, 8 / 50)


def test_compare() -> None:
    """Test the :meth:`~unxt.UnitSystem.compare` method."""
    usys1 = unitsystem(u.kpc, u.Myr, u.radian, u.Msun, u.mas / u.yr)
    usys1_clone = unitsystem(u.kpc, u.Myr, u.radian, u.Msun, u.mas / u.yr)

    usys2 = unitsystem(u.kpc, u.Myr, u.radian, u.Msun, u.kiloarcsecond / u.yr)
    usys3 = unitsystem(u.kpc, u.Myr, u.radian, u.kg, u.mas / u.yr)

    assert usys1 == usys1_clone
    assert usys1_clone == usys1

    assert usys1 != usys2
    assert usys2 != usys1

    assert usys1 != usys3
    assert usys3 != usys1


def test_pickle(tmpdir: Path) -> None:
    """Test pickling and unpickling a :class:`~unxt.UnitSystem`."""
    usys = unitsystem(u.kpc, u.Myr, u.radian, u.Msun)

    path = tmpdir / "test.pkl"
    with path.open(mode="wb") as f:
        pickle.dump(usys, f)

    with path.open(mode="rb") as f:
        usys2 = pickle.load(f)  # noqa: S301

    assert usys == usys2


def test_non_slot():
    # Passes
    class NoSlots1(AbstractUnitSystem):
        pass

    # Fails
    with pytest.raises(TypeError, match="cannot inherit"):

        @dataclass
        class NoSlots2(AbstractUnitSystem):
            pass

    # Passes
    @dataclass(frozen=True, slots=True)
    class NoSlots3(AbstractUnitSystem):
        pass


class TestDimensionlessUnitSystem:
    """Test `unxt.unitsystems.DimensionlessUnitSystem`."""

    def test_getitem(self) -> None:
        """Test :meth:`unxt.unitsystems.DimensionlessUnitSystem.__getitem__`."""
        assert dimensionless["dimensionless"] == u.one

        with pytest.raises(u.UnitConversionError):
            dimensionless["length"]

    def test_decompose(self) -> None:
        """Test that dimensionless unitsystem can be decomposed."""
        with pytest.raises(ValueError, match="can not be decomposed into"):
            (15 * u.kpc).decompose(dimensionless)
