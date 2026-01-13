"""Tests for named dimension strategies."""

import hypothesis.strategies as st
from hypothesis import given, settings

import unxt as u
import unxt_hypothesis as ust


def test_DIMENSION_NAMES_public() -> None:  # noqa: N802
    """Ensure the named dimension list is public and unique."""
    names = ust.DIMENSION_NAMES
    assert isinstance(names, tuple)
    assert names
    assert len(names) == len(set(names))


@given(name=st.sampled_from(ust.DIMENSION_NAMES))
@settings(max_examples=30)
def test_all_names_resolve(name: str) -> None:
    """All named dimensions should resolve via ``unxt.dimension``."""
    dim = u.dimension(name)
    assert isinstance(dim, u.AbstractDimension)


@given(dim=ust.named_dimensions())
@settings(max_examples=30)
def test_named_dimensions_strategy(dim: u.AbstractDimension) -> None:
    """Strategy should return valid dimensions."""
    assert isinstance(dim, u.AbstractDimension)
