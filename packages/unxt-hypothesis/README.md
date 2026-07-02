# unxt-hypothesis

> **Note:** As of v2.0.0, `unxt-hypothesis` is a backward-compatible shim for [`unxts.hypothesis`](https://pypi.org/p/unxts.hypothesis) — the new canonical location for this package. No code changes are required: `import unxt_hypothesis` continues to work exactly as before. New projects should prefer `pip install unxts.hypothesis` and `import unxts.hypothesis`.

[Hypothesis](https://hypothesis.readthedocs.io/) strategies for property-based testing with [unxt](https://github.com/GalacticDynamics/unxt).

## Installation

```bash
pip install unxt-hypothesis       # legacy name — continues to work long-term
pip install unxts.hypothesis      # canonical name going forward
```

## Migration

If you currently depend on `unxt-hypothesis`, you have two options:

1. **Do nothing** — `unxt-hypothesis` re-exports the full public API from `unxts.hypothesis` and will be maintained long-term. Existing code requires no changes.

2. **Migrate to `unxts.hypothesis`** — change your dependency and imports:

   ```diff
   - pip install unxt-hypothesis
   + pip install unxts.hypothesis
   ```

   ```diff
   - import unxt_hypothesis as ust
   + import unxts.hypothesis as ust
   ```

   All public strategies (`quantities`, `units`, `angles`, `unitsystems`, etc.) are identical — the canonical package is the source of truth and the shim simply re-exports from it.

## Strategies

- `named_dimensions()` — physical dimensions from Astropy's catalogue
- `units(dimension)` — random `Unit` objects
- `derived_units(base)` — dimensionally-equivalent units
- `quantities(*, shape, dtype, unit)` — random `ParametricQuantity` objects
- `unitsystems(*units)` — random `UnitSystem` objects
- `angles(*, wrap_to)` — random `Angle` objects
- `wrap_to(quantity, min, max)` — range-constrained quantities

## Quick Start

```python
from hypothesis import given
import unxt as u
import unxts.hypothesis as ust  # or: import unxt_hypothesis as ust


@given(q=ust.quantities(unit="km/s"))
def test_velocity(q):
    assert q.unit == u.unit("km/s")
```

## Documentation

- [unxts.hypothesis documentation](https://unxt.readthedocs.io/en/latest/packages/unxts.hypothesis/)
- [unxt documentation](https://unxt.readthedocs.io/)
- [Hypothesis documentation](https://hypothesis.readthedocs.io/)
