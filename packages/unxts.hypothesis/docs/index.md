# `unxts.hypothesis`

```{toctree}
:maxdepth: 1
:hidden:

strategies
testing-guide
recipes
api
```

[Hypothesis](https://hypothesis.readthedocs.io/) strategies for property-based testing with [unxt](https://github.com/GalacticDynamics/unxt). It generates random `Quantity`, `Angle`, `Unit`, `Dimension` and `UnitSystem` objects, so you can assert that a property holds for _all_ inputs rather than for the handful you thought to write down.

:::{note}

`unxts.hypothesis` is the canonical package. The legacy `unxt-hypothesis` distribution remains available as a thin backward-compatible shim that re-exports this package, so existing `import unxt_hypothesis` code keeps working unchanged.

:::

## Install

::::{tab-set}

:::{tab-item} uv

```bash
uv add unxts.hypothesis
```

:::

:::{tab-item} pip

```bash
pip install unxts.hypothesis
```

:::

::::

## At a glance

```python
import jax
from hypothesis import given

import unxt as u
import unxts.hypothesis as ust


@given(q=ust.quantities(unit="km/s"))
def test_quantity_property(q):
    """Every generated quantity has a JAX value and the requested unit."""
    assert isinstance(q.value, jax.Array)
    assert q.unit == u.unit("km/s")
```

Hypothesis runs that test on many generated quantities and, when one fails, shrinks it to the smallest input that still fails.

## Pages

**How-to**

- [How to write property-based tests](testing-guide) — the workflow, from a first property to debugging a shrunk failure.
- [How to combine strategies](recipes) — composing strategies, testing unitful functions, and narrowing generation to a domain.

**Reference**

- [Strategies](strategies) — every strategy, its parameters and what it generates, plus the `st.from_type()` registrations.
- [API](api) — the generated API documentation.

## Public API

`unxts.hypothesis` exposes the strategies `named_dimensions`, `derived_units`, `units`, `quantities`, `unitsystems`, `angles` and `wrap_to`, along with `DIMENSION_NAMES`. Importing the package also registers `st.from_type()` strategies for `unxt`'s quantity, angle and unit-system types.

## See also

- [Hypothesis documentation](https://hypothesis.readthedocs.io/)
- [What is property-based testing?](https://hypothesis.works/articles/what-is-property-based-testing/)
