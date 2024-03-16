"""Test the :mod:`~unxt.unitsystems` module."""

import pickle
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from unxt.unitsystems import UnitSystem, dimensionless


class TestUnitSystem:
    """Test :class:`~unxt.UnitSystem`."""

    def test_constructor(self) -> None:
        """Test the :class:`~unxt.UnitSystem` constructor."""
        usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun)

        match = "must specify a unit for the physical type .*mass"
        with pytest.raises(ValueError, match=match):
            UnitSystem(u.kpc, u.Myr, u.radian)  # no mass

        match = "must specify a unit for the physical type .*angle"
        with pytest.raises(ValueError, match=match):
            UnitSystem(u.kpc, u.Myr, u.Msun)

        match = "must specify a unit for the physical type .*time"
        with pytest.raises(ValueError, match=match):
            UnitSystem(u.kpc, u.radian, u.Msun)

        match = "must specify a unit for the physical type .*length"
        with pytest.raises(ValueError, match=match):
            UnitSystem(u.Myr, u.radian, u.Msun)

        usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun)
        usys = UnitSystem(usys)

    def test_constructor_quantity(self) -> None:
        """Test the :class:`~unxt.UnitSystem` constructor with quantities."""
        usys = UnitSystem(5 * u.kpc, 50 * u.Myr, 1e5 * u.Msun, u.rad)
        assert np.isclose((8 * u.Myr).decompose(usys).value, 8 / 50)

    def test_preferred(self) -> None:
        """Test the :meth:`~unxt.UnitSystem.preferred` method."""
        usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.km / u.s)
        q = 15.0 * u.km / u.s
        assert usys.preferred("velocity") == u.km / u.s
        assert q.decompose(usys).unit == u.kpc / u.Myr
        assert usys.as_preferred(q).unit == u.km / u.s

    # ===============================================================

    def test_compare(self) -> None:
        """Test the :meth:`~unxt.UnitSystem.compare` method."""
        usys1 = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.mas / u.yr)
        usys1_clone = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.mas / u.yr)

        usys2 = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.kiloarcsecond / u.yr)
        usys3 = UnitSystem(u.kpc, u.Myr, u.radian, u.kg, u.mas / u.yr)

        assert usys1 == usys1_clone
        assert usys1_clone == usys1

        assert usys1 != usys2
        assert usys2 != usys1

        assert usys1 != usys3
        assert usys3 != usys1

    def test_pickle(self, tmpdir: Path) -> None:
        """Test pickling and unpickling a :class:`~unxt.UnitSystem`."""
        usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun)

        path = tmpdir / "test.pkl"
        with path.open(mode="wb") as f:
            pickle.dump(usys, f)

        with path.open(mode="rb") as f:
            usys2 = pickle.load(f)  # noqa: S301

        assert usys == usys2


class TestDimensionlessUnitSystem:
    """Test :class:`~unxt.unitsystems.DimensionlessUnitSystem`."""

    def test_getitem(self) -> None:
        """Test :meth:`~unxt.unitsystems.DimensionlessUnitSystem.__getitem__`."""
        assert dimensionless["dimensionless"] == u.one
        assert dimensionless["length"] == u.one

    def test_decompose(self) -> None:
        """Test that dimensionless unitsystem can be decomposed."""
        with pytest.raises(ValueError, match="can not be decomposed into"):
            (15 * u.kpc).decompose(dimensionless)

    def test_preferred(self) -> None:
        """Test the ``DimensionlessUnitSystem.preferred`` method."""
        with pytest.raises(ValueError, match="are not convertible"):
            dimensionless.as_preferred(15 * u.kpc)
