"""`UnitsMatrix` / `QuantityMatrix` <-> astropy structured units and quantities."""

import astropy.units as apyu
import jax.numpy as jnp
import numpy as np
import plum
import pytest
from unxts.linalg import QuantityMatrix, UnitsMatrix

#: Unit layouts, exercised in both directions.
LAYOUTS = [
    pytest.param(("m", "s", "kg"), (3,), id="1d"),
    pytest.param((("m", "s"), ("kg", "rad")), (2, 2), id="2d"),
]


@pytest.mark.parametrize(("units", "shape"), LAYOUTS)
def test_units_matrix_to_structured_unit(units: tuple, shape: tuple) -> None:
    del shape
    result = plum.convert(UnitsMatrix(units), apyu.StructuredUnit)

    assert isinstance(result, apyu.StructuredUnit)
    assert result == apyu.StructuredUnit(units)


@pytest.mark.parametrize(("units", "shape"), LAYOUTS)
def test_structured_unit_to_units_matrix(units: tuple, shape: tuple) -> None:
    result = plum.convert(apyu.StructuredUnit(units), UnitsMatrix)

    assert isinstance(result, UnitsMatrix)
    assert result.shape == shape
    for index in np.ndindex(shape):
        expected = units
        for i in index:
            expected = expected[i]
        assert result[index] == apyu.Unit(expected)


@pytest.mark.parametrize(("units", "shape"), LAYOUTS)
def test_units_round_trip(units: tuple, shape: tuple) -> None:
    del shape
    umat = UnitsMatrix(units)
    assert plum.convert(plum.convert(umat, apyu.StructuredUnit), UnitsMatrix) == umat


def test_a_layout_deeper_than_a_matrix_is_refused_by_depth() -> None:
    """Astropy nests to any depth; `UnitsMatrix` is a vector or a matrix.

    Without the check this reaches `UnitsMatrix`'s own "ragged structure"
    error, which names the wrong fault: the layout is not ragged, it is deep.
    """
    deep = apyu.StructuredUnit(((("m", "s"), ("kg", "rad")),))
    with pytest.raises(ValueError, match="3-deep"):
        plum.convert(deep, UnitsMatrix)


def test_an_empty_layout_keeps_its_own_message() -> None:
    with pytest.raises(ValueError, match="at least one element"):
        plum.convert(apyu.StructuredUnit(()), UnitsMatrix)


class TestQuantityMatrixToAstropyQuantity:
    """The unit layout claims the trailing axes; one axis in front is batch."""

    def test_values_and_unit_reach_astropy(self) -> None:
        qmat = QuantityMatrix(jnp.array([3.0, 4.0]), unit=("m", "kg"))
        result = plum.convert(qmat, apyu.Quantity)

        assert result.unit == apyu.StructuredUnit(("m", "kg"))
        assert float(np.array(result)["f0"]) == pytest.approx(3.0)
        assert float(np.array(result)["f1"]) == pytest.approx(4.0)

    def test_a_nested_layout_nests_the_record(self) -> None:
        qmat = QuantityMatrix(
            jnp.array([[1.0, 2.0], [3.0, 4.0]]), unit=(("m", "s"), ("kg", "rad"))
        )
        result = plum.convert(qmat, apyu.Quantity)

        assert result.unit == apyu.StructuredUnit((("m", "s"), ("kg", "rad")))
        assert result.shape == ()
        assert float(np.array(result)["f0"]["f1"]) == pytest.approx(2.0)

    def test_a_flat_layout_reads_the_leading_axis_as_batch(self) -> None:
        qmat = QuantityMatrix(jnp.arange(6.0).reshape(2, 3), unit=("m", "s", "kg"))
        result = plum.convert(qmat, apyu.Quantity)

        assert result.shape == (2,)
        assert float(np.array(result)[1]["f2"]) == pytest.approx(5.0)

    def test_the_leaf_dtype_is_kept(self) -> None:
        qmat = QuantityMatrix(jnp.array([1.0, 2.0], dtype=jnp.float32), ("km", "s"))
        assert plum.convert(qmat, apyu.Quantity).dtype["f0"] == np.float32

    def test_an_empty_layout_is_refused_by_name(self) -> None:
        """Legal here (see `UnitsMatrix.__pow__`), impossible in astropy."""
        qmat = QuantityMatrix(
            jnp.zeros((0,)), unit=UnitsMatrix(np.empty(0, dtype=object))
        )
        with pytest.raises(ValueError, match="at least one field"):
            plum.convert(qmat, apyu.Quantity)

    def test_more_batch_axes_than_astropy_allows_are_refused(self) -> None:
        qmat = QuantityMatrix(jnp.ones((2, 2, 3)), unit=("m", "s", "kg"))
        with pytest.raises(ValueError, match="one batch axis"):
            plum.convert(qmat, apyu.Quantity)
