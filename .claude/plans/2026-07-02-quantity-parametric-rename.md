# Quantity → ParametricQuantity / BareQuantity → Quantity Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the lightweight non-parametric quantity (formerly `BareQuantity`) the default `unxt.Quantity` (alias `Q`), and rename the plum-parametric implementation (formerly `Quantity`) to `ParametricQuantity` (new alias `PQ`), with a deprecation shim, docs admonition explaining the JAX-compilation rationale, and new cross-class tests proving the new `Quantity` works everywhere `ParametricQuantity` does.

**Architecture:** This is a name _swap_, not just a rename: the identifier `Quantity` changes meaning. The swap is done as one atomic mechanical pass (word-boundary-aware, so `BareQuantity`/`AbstractQuantity`/`StaticQuantity` are not corrupted), followed by deliberate re-pointing of "default quantity" construction sites, a deprecation shim for `BareQuantity`, test/doctest triage, and new interchangeability coverage. Class hierarchy after the change:

```
AbstractQuantity                      (base, non-parametric)      [base.py]
├── Quantity  (alias Q)               ex-BareQuantity             [quantity.py, ex-unchecked.py]
├── AbstractParametricQuantity        (@parametric abstract)      [base_parametric.py]
│   ├── ParametricQuantity (alias PQ) ex-Quantity                 [parametric.py, ex-quantity.py]
│   └── StaticQuantity                (unchanged name)            [static_quantity.py]
└── AbstractAngle
    └── Angle                         (unchanged)
BareQuantity → deprecated module-__getattr__ alias for Quantity
```

**Tech Stack:** Python ≥3.11, JAX/equinox/quax, plum (parametric + dispatch + promotion rules), wadler-lindig reprs, pytest + Sybil doctests (`>>>` in docstrings, README.md, and `docs/**/*.md`), uv workspace, nox, gitmoji commit convention.

**Repo:** `/Users/nmrs/local/unxt` (work on a branch off `main`, e.g. `quantity-parametric-rename`). This is the headline breaking change for the planned **v2.0.0** release.

## Global Constraints

- NEVER use `from __future__ import annotations`. Write jaxtyping dims as string-literal annotations (e.g. `Shaped[Array, "*shape"]`) so annotations stay runtime-evaluable (plum dispatch depends on this).
- Commit messages follow the repo's gitmoji convention, e.g. `♻️ refactor!: ...`, `✨ feat: ...`, `✅ test: ...`, `📝 docs: ...`, `🐛 fix: ...`. Use `!` on the breaking commits.
- pytest treats `DeprecationWarning` as an error (`filterwarnings = ["error", ...]` in pyproject.toml). Any test exercising the `BareQuantity` shim must use `pytest.warns(DeprecationWarning)`.
- Doctests run via Sybil (see root `conftest.py`), NOT `--doctest-modules`. `uv run pytest src/unxt README.md docs tests` exercises doctests + tests. Never put `>>>` examples in files under `docs/` unless they are meant to be executed.
- Word-boundary discipline: `\bQuantity\b` must never touch `BareQuantity`, `AbstractQuantity`, `AbstractParametricQuantity`, `StaticQuantity`, `AstropyQuantity`, `ParametricQuantity`, or lowercase `quantity` (module paths). All bulk renames use the 3-step perl swap defined in Task 1.
- Do not edit anything under `docs/_build/` or `__pycache__/`.
- The public behavioral contract of this change: after it, the only things that stop working for users who used `Quantity` are (a) runtime dimension checking (`Quantity["time"](1, "m")` no longer raises — the subscript is a no-op) and (b) plum dispatch on dimension-specific types (`Quantity["length"]` as an annotation now means plain `Quantity`). Everything else must keep working. Preserve the no-op `__class_getitem__` on the new `Quantity` — it is what keeps `Quantity["length"](...)` construction working.

## Design Decisions (locked in — do not relitigate during implementation)

| Item | Decision |
| --- | --- |
| File names | `_src/quantity/quantity.py` → `_src/quantity/parametric.py`; `_src/quantity/unchecked.py` → `_src/quantity/quantity.py` (via `git mv`, history-preserving) |
| `short_name` | New `Quantity.short_name = "Q"`; `ParametricQuantity.short_name = "PQ"` (drives `use_short_name` reprs; machinery is on `AbstractQuantity.__pdoc__`, base.py:755, so it works for non-parametric classes) |
| Aliases | `Q = Quantity` (in new quantity.py); `PQ = ParametricQuantity` (in parametric.py). Both exported from `unxt` top level and `unxt.quantity` |
| Promotion semantics | Unchanged (parametric still wins when the user opted into it): `Quantity + ParametricQuantity → ParametricQuantity`; `StaticQuantity + ParametricQuantity → ParametricQuantity`; `StaticQuantity + Quantity → Quantity`; `AbstractAngle + ParametricQuantity → ParametricQuantity`; `AbstractAngle + Quantity → Quantity` |
| `BareQuantity` | Deprecated alias of new `Quantity`, provided via module-level `__getattr__` in `unxt/quantity.py` (public module) AND `unxt/_src/quantity/__init__.py`. Removed from all `__all__` tuples (so `import *` doesn't trigger the warning). Same class object, so isinstance/dispatch through the alias keep working |
| Hard-coded constructors in src | Dispatch registrations (e.g. `full`, `linspace` in register_dispatches.py) and interop converters that construct "a quantity" now construct the new default `Quantity`, not `ParametricQuantity` — that is the point of the change. Exceptions: code inside `parametric.py`/`base_parametric.py`/`static_quantity.py` internals, and any conversion whose _target type_ is explicitly parametric (e.g. `convert(x, ParametricQuantity)`) |
| Release | v2.0.0 (tag validation for the 2.0 threshold already landed on main). No version-number edits needed (hatch-vcs) |
| Pickles | Old pickles referencing `unxt._src.quantity.quantity.Quantity` will resolve to the new class. Private module path; accepted breakage; noted in migration docs |

---

### Task 1: Atomic mechanical name-and-file swap across the whole repo

The identifier swap cannot be split without leaving the tree in a state where `Quantity` means the wrong thing, so this task is one atomic commit. Steps inside it are granular.

**Files:**

- Rename: `src/unxt/_src/quantity/quantity.py` → `src/unxt/_src/quantity/parametric.py`
- Rename: `src/unxt/_src/quantity/unchecked.py` → `src/unxt/_src/quantity/quantity.py`
- Modify (bulk): every tracked `.py`/`.md` file matching `BareQuantity|Quantity` under `src/`, `tests/`, `packages/`, `docs/` (excluding `docs/_build`), and `README.md`
- Modify (by hand): `src/unxt/_src/quantity/__init__.py`, `src/unxt/_src/quantity/parametric.py`, `src/unxt/_src/quantity/quantity.py`, `src/unxt/quantity.py`, `src/unxt/__init__.py`

**Interfaces:**

- Produces (relied on by every later task):
  - `unxt.Quantity` / `unxt.Q` — non-parametric class (ex-`BareQuantity`), `short_name="Q"`, no-op `__class_getitem__`
  - `unxt.ParametricQuantity` / `unxt.PQ` — `@final @parametric` class (ex-`Quantity`), `short_name="PQ"`
  - `unxt.quantity` module exports: `Quantity`, `Q`, `ParametricQuantity`, `PQ`, plus all previously-exported names except `BareQuantity`

- [ ] **Step 1: Branch**

```bash
cd /Users/nmrs/local/unxt && git switch -c quantity-parametric-rename main
```

- [ ] **Step 2: Rename the two module files (before the text swap, so the swap edits the final paths)**

```bash
cd /Users/nmrs/local/unxt
git mv src/unxt/_src/quantity/quantity.py src/unxt/_src/quantity/parametric.py
git mv src/unxt/_src/quantity/unchecked.py src/unxt/_src/quantity/quantity.py
```

- [ ] **Step 3: Run the 3-step word-boundary swap over all tracked files**

The order inside the perl program matters: `BareQuantity` is stashed to a sentinel BEFORE the generic `\bQuantity\b` rule runs, then restored as `Quantity`.

```bash
cd /Users/nmrs/local/unxt
git grep -lE 'BareQuantity|Quantity' -- 'src' 'tests' 'packages' 'docs' 'README.md' \
  | grep -v 'docs/_build' \
  | xargs perl -pi -e '
      s/\bBareQuantity\b/__TMP_BAREQ__/g;
      s/\bQuantity\b/ParametricQuantity/g;
      s/__TMP_BAREQ__/Quantity/g;
    '
```

After this, the codebase is semantically identical to before (all classes renamed consistently, dispatches/conversions/promotions/doctests self-consistent) EXCEPT the items fixed in Steps 4–8. Sanity check that no corrupted identifiers exist:

```bash
git grep -nE 'ParametricParametric|BareParametric|__TMP_BAREQ__|AbstractParametricParametric' && echo "CORRUPTION FOUND" || echo "clean"
```

Expected: `clean`.

- [ ] **Step 4: Fix cross-module imports and move the promotion rule**

The swap leaves stale _module_ paths (lowercase, untouched by design). In `src/unxt/_src/quantity/quantity.py` (ex-unchecked.py), the file currently reads (post-swap):

```python
__all__ = ("Quantity",)
...
from .base import AbstractQuantity
from .quantity import ParametricQuantity  # ← stale: imports itself
from .value import StaticValue, convert_to_quantity_value

...
add_promotion_rule(Quantity, ParametricQuantity, ParametricQuantity)
```

Change it to (removing the parametric import and the promotion rule entirely — the rule moves to parametric.py so this module has no dependency on the parametric machinery):

```python
__all__ = ("Q", "Quantity")

from typing import Any, ClassVar

import equinox as eqx
from jaxtyping import Array, Shaped

from .base import AbstractQuantity
from .value import StaticValue, convert_to_quantity_value
from unxt.units import AbstractUnit, unit as parse_unit


class Quantity(AbstractQuantity):
    """The default quantity: units without dimension parametrization.

    This class is not parametrized by its dimensionality, making it a single
    class (and a single JAX pytree type) regardless of dimension. For runtime
    dimension checking and dimension-specific dispatch, see
    `unxt.ParametricQuantity`.

    Examples
    --------
    >>> import unxt as u
    >>> u.Quantity(1, "m")
    Quantity(Array(1, dtype=int32, weak_type=True), unit='m')

    """

    value: Shaped[Array | StaticValue, "*shape"] = eqx.field(
        converter=convert_to_quantity_value
    )
    """The value of the `Quantity`."""

    unit: AbstractUnit = eqx.field(static=True, converter=parse_unit)
    """The unit associated with this value."""

    short_name: ClassVar[str] = "Q"

    def __class_getitem__(cls: "type[Quantity]", item: Any) -> "type[Quantity]":
        """No-op support for ``Quantity[...]`` syntax.

        The dimension parameter is accepted for interchangeability with
        `unxt.ParametricQuantity` but is NOT checked.

        >>> from unxt.quantity import Quantity
        >>> Quantity["length"]
        <class 'unxt...quantity...Quantity'>

        """
        return cls


Q = Quantity
"""Convenience alias for `Quantity`."""
```

(Keep the existing pylint-disable header comment and any docstring content from the old file that still applies; the docstring/example bodies above replace the "fast implementation" wording. Adjust the doctest repr output to whatever the class actually prints — verify by running it.)

In `src/unxt/_src/quantity/parametric.py` (ex-quantity.py), post-swap it defines `ParametricQuantity` with `Q = ParametricQuantity` at the bottom and `short_name: ClassVar[str] = "Q"` (base_parametric-derived class body, line ~116/222 pre-swap). Change:

```python
__all__ = ("PQ", "ParametricQuantity")
```

and at the class body:

```python
short_name: ClassVar[str] = "PQ"
```

and at the bottom, replace `Q = ParametricQuantity` with:

```python
PQ = ParametricQuantity
"""Convenience alias for `ParametricQuantity`."""

add_promotion_rule(Quantity, ParametricQuantity, ParametricQuantity)
```

adding the needed imports at the top of parametric.py:

```python
from plum import add_promotion_rule, parametric

from .quantity import Quantity
```

(There is no import cycle: quantity.py no longer imports parametric.py.) Also update the module docstring of each file to describe its new contents, and sweep both files' docstrings for stale sentences like "A fast implementation of the Quantity class" / references to the old split.

- [ ] **Step 5: Update `src/unxt/_src/quantity/__init__.py`**

Replace the star-import lines for the renamed modules (order: base → parametric deps → quantity is fine either way since parametric imports quantity directly):

```python
from .angle import *
from .base import *
from .base_angle import *
from .base_parametric import *
from .flag import *
from .parametric import *
from .quantity import *
from .static_quantity import *
from .value import *

from .register_api import *
from .register_conversions import *
from .register_dispatches import *
from .register_primitives import *
from .register_ufuncs import *
```

(There is no line for `unchecked` anymore.) Then add the deprecation `__getattr__` at the bottom of this file:

```python
def __getattr__(name: str) -> object:
    if name == "BareQuantity":
        import warnings

        warnings.warn(
            "`BareQuantity` has been renamed to `Quantity` and is now the "
            "default quantity class (unxt v2). The parametric class formerly "
            "named `Quantity` is now `ParametricQuantity`. `BareQuantity` "
            "will be removed in a future release.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return Quantity
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
```

- [ ] **Step 6: Update the public module `src/unxt/quantity.py`**

Post-swap its `__all__` already reads `"ParametricQuantity"`/`"Quantity"` correctly (the old `"Quantity"`/`"BareQuantity"` entries were swapped in place). Verify, then: add `"PQ"` to `__all__` (keep `"Q"`); update the import block from `._src.quantity` to import `ParametricQuantity, PQ, Q, Quantity` (drop the direct `BareQuantity` import — post-swap it appears as a duplicate `Quantity` import; remove the duplicate); and add the same deprecated-alias hook at module bottom:

```python
def __getattr__(name: str) -> object:
    if name == "BareQuantity":
        import warnings

        warnings.warn(
            "`BareQuantity` has been renamed to `Quantity` and is now the "
            "default quantity class (unxt v2). The parametric class formerly "
            "named `Quantity` is now `ParametricQuantity`. `BareQuantity` "
            "will be removed in a future release.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return Quantity
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
```

Also rewrite the module docstring's "Core Classes" list: `Quantity` is "the default quantity class, no dimension parametrization, aliased as `Q`"; `ParametricQuantity` is "dimension-parametrized with runtime checking, aliased as `PQ`"; delete the `BareQuantity` bullet. Fix the docstring's doctest examples per the new reprs (run Sybil to verify, Task 6 catches stragglers).

- [ ] **Step 7: Update the top-level `src/unxt/__init__.py`**

Post-swap `__all__` contains `"ParametricQuantity"` (was `"Quantity"`) and `"Q"`. Make the quantity block of `__all__` read:

```python
"AbstractQuantity",
"Angle",
"PQ",
"ParametricQuantity",
"Q",
"Quantity",
"StaticQuantity",
```

and the corresponding import:

```python
from .quantity import (
    AbstractQuantity,
    Angle,
    ParametricQuantity,
    PQ,
    Q,
    Quantity,
    StaticQuantity,
    is_unit_convertible,
    uconvert,
    uconvert_value,
    ustrip,
)
```

(Do NOT add `BareQuantity` at top level — it was never exported there.)

- [ ] **Step 8: Smoke-test imports and core semantics**

```bash
cd /Users/nmrs/local/unxt && uv run python -c "
import unxt as u
from unxt.quantity import ParametricQuantity, Quantity

assert u.Q is u.Quantity, 'Q must alias the new (non-parametric) Quantity'
assert u.PQ is u.ParametricQuantity
assert not hasattr(type(u.Quantity), '__infer_type_parameter__') or True

q = u.Q(1.0, 'm')
assert type(q).__name__ == 'Quantity'
assert u.Quantity['length'] is u.Quantity, 'subscript must be a no-op'

pq = u.PQ(1.0, 'm')
assert type(pq).mro()[1].__name__ == 'ParametricQuantity' or 'ParametricQuantity' in repr(type(pq))
print(repr(q))
print(repr(pq))
print(repr(q + pq), '<- promotion: must be ParametricQuantity')
"
```

Expected: `Q` prints `Quantity(...)`, `PQ` prints `ParametricQuantity(...)`, `q + pq` is a `ParametricQuantity`.

- [ ] **Step 9: Commit**

```bash
git add -A && git commit -m "♻️ refactor!: rename BareQuantity->Quantity, Quantity->ParametricQuantity

The non-parametric implementation is now the default Quantity (alias Q).
The plum-parametric implementation is ParametricQuantity (alias PQ).
Parametric quantities are a new class per dimension, which makes every
new dimension a new pytree type and triggers jax.jit recompilation, so
the single-class implementation is the better default."
```

(The test suite is NOT expected to pass at this commit; Tasks 2–6 restore it. That is acceptable for this one atomic rename commit.)

---

### Task 2: `BareQuantity` deprecation shim tests

**Files:**

- Test: `tests/unit/test_deprecations.py` (create)

**Interfaces:**

- Consumes: `unxt.quantity.__getattr__` shim from Task 1.
- Produces: nothing new; locks the shim's contract.

- [ ] **Step 1: Write the failing-or-passing test file**

```python
"""Tests for deprecated aliases kept for the v1 -> v2 transition."""

import pytest

import unxt as u


def test_barequantity_is_deprecated_alias():
    """`BareQuantity` warns and resolves to the new default `Quantity`."""
    with pytest.warns(DeprecationWarning, match="renamed to `Quantity`"):
        from unxt.quantity import BareQuantity  # noqa: PLC0415

    assert BareQuantity is u.Quantity


def test_barequantity_attribute_access_warns():
    """Attribute access (not just import) also warns."""
    import unxt.quantity as uq  # noqa: PLC0415

    with pytest.warns(DeprecationWarning):
        cls = uq.BareQuantity
    assert cls is u.Quantity


def test_barequantity_not_in_public_all():
    """The deprecated name is not advertised via __all__ / star-import."""
    import unxt.quantity as uq  # noqa: PLC0415

    assert "BareQuantity" not in uq.__all__


def test_unknown_attribute_raises():
    import unxt.quantity as uq  # noqa: PLC0415

    with pytest.raises(AttributeError, match="NotAThing"):
        _ = uq.NotAThing
```

- [ ] **Step 2: Run it**

```bash
uv run pytest tests/unit/test_deprecations.py -v
```

Expected: 4 passed. If the `match=` string fails, align it with the actual warning text from Task 1 Step 5/6 (single source of truth: the warning message).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_deprecations.py
git commit -m "✅ test: BareQuantity deprecation shim contract"
```

---

### Task 3: Re-point "default quantity" construction sites from ParametricQuantity to Quantity

The mechanical swap preserved old semantics: everywhere the library _constructed_ the old parametric `Quantity` now constructs `ParametricQuantity`. Per the design decision, sites that just need "a quantity" should construct the new default `Quantity`. This is a behavior change (outputs lose parametricity) — deliberate and documented.

**Files (audit each; change construction sites, leave dispatch _annotations_ on abstract types alone):**

- Modify: `src/unxt/_src/quantity/register_dispatches.py` (~36 post-swap `ParametricQuantity` refs: `arange`, `empty_like`, `full`, `full_like`, `linspace`, `ones_like`, `zeros_like`)
- Modify: `src/unxt/_src/quantity/register_conversions.py` (keep BOTH conversion targets: `convert(x, Quantity)` and `convert(x, ParametricQuantity)` must each work; post-swap the file already has both — verify the `type_to=` pairs are `Quantity` ex-BareQuantity and `ParametricQuantity` ex-Quantity, and that doctest outputs match)
- Modify: `src/unxt/_interop/unxt_interop_astropy/quantity.py` (the `AbstractQuantity.from_` dispatches at ex-lines 53/75: if they construct a hard-coded class rather than `cls`, point them at `Quantity`; keep both `conversion_method(type_from=AstropyQuantity, type_to=ParametricQuantity)` and `type_to=Quantity` registrations)
- Modify: `packages/unxts.interop.xarray/src/unxts/interop/xarray/_src/accessors.py` and `conversion.py` (DataArray→quantity conversion should yield the default `Quantity`)
- Modify: `packages/unxts.interop.matplotlib/src/unxts/interop/matplotlib/_src/converter.py` (unit-conversion plumbing: default `Quantity`)
- Modify: `packages/unxts.hypothesis/src/unxts/hypothesis/_src/quantities.py` (default strategies generate the default `Quantity`; keep/add a strategy or `cls=` parameter for `ParametricQuantity` so parametric behavior remains testable)
- Audit only (likely no change): `src/unxt/_src/quantity/register_api.py`, `register_ufuncs.py`, `register_primitives.py` (annotations are on `AbstractQuantity`/concrete pairs and must keep matching the same classes as before the swap), `src/unxt/_src/experimental.py` (the swap already turned its internal `BareQuantity` uses into `Quantity` — correct as-is), `packages/unxts.api/`, `packages/unxts.interop.gala/`
- Audit `src/unxt/_src/dimensions.py` doctest references.

- [ ] **Step 1: For each file above, decide per occurrence using this rule**

Annotation in a `@dispatch`/`@quax.register`/`@conversion_method` signature → leave as the post-swap name (semantics preserved). Constructor call producing a return value where the input was not specifically parametric → change `ParametricQuantity(...)` to `Quantity(...)`. When the code can instead use `type(x)(...)` or `replace(x, ...)` to preserve the input's class, prefer that — it is strictly more compatible.

- [ ] **Step 2: Run the targeted tests**

```bash
uv run pytest tests/unit/test_quantity.py tests/integration -x -q 2>&1 | tail -20
```

Expected at this stage: failures limited to repr/doctest strings and parametric-specific assertions (Task 5 fixes those). No `ImportError`/`NameError`/dispatch-lookup errors — if plum reports ambiguous or missing dispatch, a registration pair was broken; fix before proceeding.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "♻️ refactor: construct the default Quantity in factory dispatches and interop"
```

---

### Task 4: `lax.rem_p` support for the new default `Quantity`

The only dimension-parametrized dispatch in the library is `rem_p_uqv` (register_primitives.py ex-line 4289, post-swap annotated `ParametricQuantity["dimensionless"]`). The bare class had no `rem` support, so `jnp.remainder(Q(...), array)` would fail — an interchangeability gap.

**Files:**

- Modify: `src/unxt/_src/quantity/register_primitives.py` (adjacent to `rem_p_uqv`)
- Test: `tests/unit/test_quantity.py` (append) or the interchangeability module from Task 7 — put it here now, Task 7 also covers it generically.

**Interfaces:**

- Produces: `rem_p` registration on the new `Quantity`, dimensionless-checked at trace time via `ustrip("", x)`.

- [ ] **Step 1: Write the failing test**

```python
def test_remainder_bare_quantity_dimensionless():
    """jnp.remainder works on the (non-parametric) default Quantity."""
    q = u.Quantity(jnp.asarray([5.0, 7.0]), "")
    got = jnp.remainder(q, jnp.asarray(3.0))
    assert isinstance(got, u.Quantity)
    assert jnp.array_equal(got.value, jnp.asarray([2.0, 1.0]))


def test_remainder_bare_quantity_dimensionful_raises():
    q = u.Quantity(jnp.asarray([5.0, 7.0]), "m")
    with pytest.raises(Exception):  # UnitConversionError from ustrip
        _ = jnp.remainder(q, jnp.asarray(3.0))
```

(Match the import style of the surrounding test file — it uses `unxt as u` and `quaxed.numpy` / `jax.numpy`; mirror whichever the neighboring remainder/lax tests use.)

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_quantity.py -k remainder -v
```

Expected: FAIL (no quax registration matches `Quantity` for `rem_p`).

- [ ] **Step 3: Implement, next to `rem_p_uqv` in register_primitives.py**

```python
@register(lax.rem_p)
def rem_p_qv(x: Quantity, y: ArrayLike, /) -> Quantity:
    """Remainder for a non-parametric dimensionless quantity and an array.

    >>> import unxt as u
    >>> import quaxed.numpy as jnp
    >>> q = u.Quantity(10, "")
    >>> jnp.remainder(q, 3)
    Quantity(Array(1, dtype=int32, weak_type=True), unit='')

    """
    return Quantity(lax.rem(ustrip("", x), y), unit=one)
```

Match the exact decorator name (`@register` vs `@quax.register`), the `one`/dimensionless-unit constant, and the construction idiom used by `rem_p_uqv` in that file — copy its body shape, swapping the class and using `ustrip("", x)` (which raises for non-dimensionless input, giving the runtime check the parametric annotation used to provide).

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest tests/unit/test_quantity.py -k remainder -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "✨ feat: lax.rem_p support for the default Quantity"
```

---

### Task 5: Test-suite triage

Post-swap, tests are self-consistently renamed but three categories fail. Fix file-by-file.

**Files (highest-impact first):**

- Modify: `tests/unit/test_quantity.py` (246 `Q(` calls — now exercise the new bare `Quantity`; parametric-only assertions must switch to `PQ`)
- Modify: `tests/unit/test_quantity_printing.py` (repr/short_name tests: `Q(...)` short-name reprs now come from `Quantity`; add/adjust `PQ[...]` expectations)
- Modify: `tests/unit/test_config.py` (34 refs — repr-config doctests over parametric reprs)
- Modify: `tests/integration/quaxed/test_numpy.py` (340 `Q(`), `test_lax.py` (123 `Q(`), `tests/integration/test_plum.py`, `tests/unit/test_numpy_ufunc.py`, `tests/unit/test_static_quantity.py`, `tests/integration/astropy/test_astropy_interop_quantity.py`, `tests/integration/equinox/test_module.py`, benchmarks
- Modify: `packages/*/tests/*.py` (public-API tests asserting exported names)

**Failure categories and their fixes:**

1. **Parametric features tested through `Q`** — subscript construction expecting dimension checking, `ValueError` on unit/dimension mismatch, `dimension_of(Quantity["length"])`, plum parametric dispatch tests (`tests/integration/test_plum.py`), `AbstractParametricQuantity` isinstance checks. Fix: switch those tests to `u.PQ` / `ParametricQuantity`. Where a test checked `Quantity["time"](1, "m")` raises, ADD a companion assertion that the new `Quantity["time"](1, "m")` does NOT raise (documents the intended difference).
2. **Repr strings** — expected `ParametricQuantity(...)` where the object is now constructed as bare `Quantity` (or vice versa), and short-name reprs (`Q(...)`/`PQ['length'](...)`). Fix: align expected strings with actual output; in test_quantity_printing.py ensure BOTH classes' reprs are covered, including `use_short_name=True` for each.
3. **Class-identity assertions** — `type(result) is ParametricQuantity` where Task 3 changed a factory to return `Quantity`. Fix: update the expected class; if the test's purpose was "output preserves input class", prefer testing that property with both classes.

- [ ] **Step 1: Run the suite and enumerate failures**

```bash
uv run pytest tests -x -q 2>&1 | tail -5   # first failure
uv run pytest tests -q 2>&1 | tail -30     # full tally
```

- [ ] **Step 2: Fix per the categories above, one file per iteration, re-running that file after each**

```bash
uv run pytest tests/unit/test_quantity.py -q
uv run pytest tests/unit/test_quantity_printing.py -q
# ... etc for each file listed above
```

- [ ] **Step 3: Full test pass (excluding doctests)**

```bash
uv run pytest tests -q
```

Expected: all pass.

- [ ] **Step 4: Run workspace package tests**

```bash
uv run --extra workspace pytest packages -q
```

(If the extras/paths differ, use `uv run nox -s test` which iterates packages.) Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "✅ test: triage suite for the Quantity/ParametricQuantity rename"
```

---### Task 6: Doctest and docs sweep (Sybil)

The swap edited docstrings/markdown consistently, but examples that construct via the **alias `Q`** kept their text (`u.Q(...)`) while the sed rewrote their _expected output_ to `ParametricQuantity(...)` — those now mismatch, since `u.Q` produces the new `Quantity`. Also prose that _describes_ the old design must be rewritten by hand.

**Files:**

- Modify (mechanical output fixes, driven by Sybil failures): `src/unxt/**/*.py` docstrings (register_primitives.py is the big one: ~625 pre-swap name occurrences, mostly doctests), `src/unxt/_src/config.py`, `src/unxt/_src/experimental.py`, `src/unxt/_src/dimensions.py`, `README.md`
- Modify (prose rewrite by hand): `docs/guides/sharp-bits.md` (the "use BareQuantity for speed" advice inverts: bare is now the default; the sharp bit is now "ParametricQuantity triggers recompilation per dimension"), `docs/guides/type-checking.md`, `docs/glossary.md` (define `Quantity`, `ParametricQuantity`, note `BareQuantity` as deprecated), `docs/interop/dataclassish.md`, `docs/guides/quantity.md`, `docs/guides/dimensions.md`, `docs/guides/configuration.md` (repr examples: `Q(...)`/`PQ['length'](...)` short names)

- [ ] **Step 1: Run Sybil doctests and fix failures iteratively**

```bash
uv run pytest src/unxt README.md docs -q 2>&1 | tail -30
```

Fix pattern for the alias-output mismatches: an example constructing with `u.Q(` / `Q(` must show `Quantity(...)` output; constructing with `u.PQ(`/`ParametricQuantity(` must show `ParametricQuantity(...)`; short-name repr examples show `Q(...)`/`PQ(...)`. Re-run until clean. Where an example's _pedagogical point_ was parametric (e.g. showing `['length']` in the repr or demonstrating dimension checking), convert the construction to `u.PQ` rather than deleting the parameter display.

- [ ] **Step 2: Rewrite the prose docs listed above**

Content requirements: `Quantity`/`Q` presented as the default everywhere; `ParametricQuantity`/`PQ` presented as the opt-in for runtime dimension checking and dimension-specific plum dispatch; `BareQuantity` mentioned only as a deprecated alias in the glossary and migration text. README's "BareQuantity" section becomes a "ParametricQuantity" section with the inverted framing.

- [ ] **Step 3: Verify full doctest + test pass**

```bash
uv run pytest src/unxt README.md docs tests -q
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "📝 docs: doctest and prose sweep for the v2 Quantity rename"
```

---

### Task 7: Interchangeability test module (the coverage the old suite lacked)

Prove the new `Quantity` works everywhere `ParametricQuantity` does, and pin the two documented differences. This is the regression net for downstream users.

**Files:**

- Test: `tests/unit/test_quantity_interchangeability.py` (create)

**Interfaces:**

- Consumes: `unxt.Quantity`, `unxt.ParametricQuantity`, `unxt.StaticQuantity`, `unxt.Angle`, public API functions.

- [ ] **Step 1: Write the test module**

```python
"""`Quantity` must be usable everywhere `ParametricQuantity` is.

The v2 rename makes the non-parametric class the default. These tests run
the core API over both classes to guarantee interchangeability, and pin
the two intentional differences (no runtime dimension checking, no
dimension-specific dispatch).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from plum import convert

import unxt as u
from unxt.quantity import AbstractQuantity, ParametricQuantity, Quantity


@pytest.fixture(params=[Quantity, ParametricQuantity], ids=lambda c: c.__name__)
def Qcls(request):
    """Both public quantity classes."""
    return request.param


# ---------------------------------------------------------------- construction


def test_construct_scalar(Qcls):
    q = Qcls(1.5, "m")
    assert isinstance(q, AbstractQuantity)
    assert q.unit == u.unit("m")


def test_construct_array(Qcls):
    q = Qcls(jnp.asarray([1.0, 2.0]), "km")
    assert q.shape == (2,)


def test_construct_via_subscript(Qcls):
    # No-op for Quantity, checked for ParametricQuantity — both must accept.
    q = Qcls["length"](2.0, "m")
    assert q.unit == u.unit("m")


def test_from_(Qcls):
    q = Qcls.from_(jnp.asarray([1.0]), "m")
    assert isinstance(q, Qcls)


# ---------------------------------------------------------------- arithmetic


def test_arithmetic_same_class(Qcls):
    a, b = Qcls(2.0, "m"), Qcls(3.0, "m")
    assert (a + b).value == 5.0
    assert (a - b).value == -1.0
    assert (a * b).unit == u.unit("m2")
    assert (a / b).unit == u.unit("")
    assert (a**2).unit == u.unit("m2")
    assert bool(a < b)


def test_arithmetic_with_bare_array_dimensionless(Qcls):
    a = Qcls(2.0, "")
    assert (a + jnp.asarray(1.0)).value == 3.0


# ------------------------------------------------------------------ unit API


def test_uconvert(Qcls):
    q = u.uconvert("km", Qcls(1000.0, "m"))
    assert q.value == 1.0


def test_ustrip(Qcls):
    assert u.ustrip("m", Qcls(1.0, "m")) == 1.0


def test_is_unit_convertible(Qcls):
    assert u.is_unit_convertible("km", Qcls(1.0, "m"))


def test_dimension_of_instance(Qcls):
    assert u.dimension_of(Qcls(1.0, "m")) == u.dimension("length")


# ------------------------------------------------------------ JAX transforms


def test_jit(Qcls):
    @jax.jit
    def f(x):
        return x * 2

    got = f(Qcls(3.0, "m"))
    assert got.value == 6.0


def test_grad(Qcls):
    from unxt import experimental  # noqa: PLC0415

    def f(x):
        return x**2

    got = experimental.grad(f, units=("m",))(Qcls(3.0, "m"))
    assert got.value == 6.0


def test_vmap(Qcls):
    got = jax.vmap(lambda x: x + x)(Qcls(jnp.asarray([1.0, 2.0]), "s"))
    assert jnp.array_equal(got.value, jnp.asarray([2.0, 4.0]))


def test_tree_roundtrip(Qcls):
    q = Qcls(jnp.asarray([1.0, 2.0]), "m")
    leaves, treedef = jax.tree.flatten(q)
    q2 = jax.tree.unflatten(treedef, leaves)
    assert isinstance(q2, type(q))
    assert q2.unit == q.unit


def test_eqx_module_field(Qcls):
    class M(eqx.Module):
        x: AbstractQuantity

    m = M(x=Qcls(1.0, "m"))
    assert m.x.value == 1.0


# ------------------------------------------------------------------ promotion


def test_promotion_with_static_quantity(Qcls):
    s = u.StaticQuantity(1.0, "m")
    got = s + Qcls(1.0, "m")
    assert isinstance(got, Qcls)


def test_mixing_bare_and_parametric_promotes_to_parametric():
    got = Quantity(1.0, "m") + ParametricQuantity(1.0, "m")
    assert isinstance(got, ParametricQuantity)


# ----------------------------------------------------------------- conversion


def test_convert_between_classes(Qcls):
    q = convert(Qcls(1.0, "m"), Quantity)
    assert isinstance(q, Quantity)
    pq = convert(Qcls(1.0, "m"), ParametricQuantity)
    assert isinstance(pq, ParametricQuantity)


def test_convert_to_astropy(Qcls):
    apy = pytest.importorskip("astropy.units")
    got = convert(Qcls(1.0, "m"), apy.Quantity)
    assert got.unit == apy.m


def test_astropy_compat_methods(Qcls):
    pytest.importorskip("astropy.units")
    q = Qcls(1000.0, "m")
    assert q.to("km").value == 1.0
    assert q.to_value("km") == 1.0


# ------------------------------------------- the two intentional differences


def test_only_parametric_checks_dimensions():
    """Documented v2 difference #1: no runtime dimension checking on Quantity."""
    with pytest.raises(ValueError, match="[Pp]hysical type"):
        ParametricQuantity["time"](1.0, "m")
    # The same expression on the default Quantity is a silent no-op.
    q = Quantity["time"](1.0, "m")
    assert q.unit == u.unit("m")


def test_only_parametric_supports_dimension_dispatch():
    """Documented v2 difference #2: dimension-specific types for dispatch."""
    assert ParametricQuantity["length"] is not ParametricQuantity
    assert Quantity["length"] is Quantity
```

- [ ] **Step 2: Run it; triage genuine gaps**

```bash
uv run pytest tests/unit/test_quantity_interchangeability.py -v
```

Expected: all pass. Every failure here is either (a) a wrong assumption in the test about unxt's actual API — fix the test to use the real API (e.g. the correct `experimental.grad` signature, `u.dimension_of` vs `unxt.dims.dimension_of`, `from_` overloads), or (b) a REAL interchangeability gap in the new `Quantity` — fix it in src (as Task 4 did for `rem_p`) and note it in the commit message. Do not delete a failing test without classifying it.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "✅ test: Quantity/ParametricQuantity interchangeability suite"
```

---

### Task 8: Docs admonition and migration guidance

**Files:**

- Modify: `docs/guides/quantity.md` (admonition near the top, right after the intro of `Quantity`)
- Modify: `README.md` (short version of the same message in the quantity section)
- Modify: `docs/guides/sharp-bits.md` (cross-link the admonition from the recompilation sharp-bit rewritten in Task 6)

- [ ] **Step 1: Add the admonition to `docs/guides/quantity.md`**

```markdown
:::{admonition} Changed in v2: `Quantity` is no longer parametric :class: important

Prior to v2, `unxt.Quantity` was parametrized by its dimension: `Quantity["length"]` and `Quantity["time"]` are _different classes_, created on the fly. Because JAX keys its compilation cache on the pytree structure — which includes the class of every node — every quantity with a different dimension triggered a fresh `jax.jit` compilation. That is pretty inefficient given JAX's compilation model.

As of v2 the default `unxt.Quantity` (alias `u.Q`) is the lightweight, non-parametric implementation formerly named `BareQuantity`: a single class for all dimensions. The parametric implementation is still available as `unxt.ParametricQuantity` (alias `u.PQ`) — reach for it only when you need its two extra features:

1. **Runtime dimension checking** — `u.PQ["length"](1, "m")` validates the unit against the dimension at construction; the default `u.Q["length"](1, "m")` accepts the subscript for compatibility but does not check it.
2. **Dispatch on specific dimensions** — `u.PQ["length"]` is a real type usable in `plum` dispatch annotations; `u.Q["length"]` is just `Quantity`.

Everything else — arithmetic, unit conversion, JAX transforms, interop — works identically with either class. `BareQuantity` remains as a deprecated alias of `Quantity` and will be removed in a future release. :::
```

(Match the repo's existing MyST admonition syntax — if other pages use ` ```{admonition} ` fenced style rather than `:::`, use that style.)

- [ ] **Step 2: Add the short README version**

In the README's quantity intro, after introducing `u.Q`, add a paragraph: "**New in v2:** `Quantity` is the fast non-parametric class (formerly `BareQuantity`). The dimension-parametric class is now `ParametricQuantity` (`u.PQ`); use it when you want runtime dimension checking or dimension-specific dispatch — each parametrized dimension is its own class, so it can trigger extra `jax.jit` compilations." Keep the existing code examples consistent with it.

- [ ] **Step 3: Verify docs still build / doctests pass**

```bash
uv run pytest docs README.md -q
```

Expected: pass (the admonition contains no executable code).

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "📝 docs: v2 admonition — why the default Quantity is non-parametric"
```

---

### Task 9: Final verification

- [ ] **Step 1: Full suite, all packages, doctests, lint**

```bash
cd /Users/nmrs/local/unxt
uv run pytest src/unxt README.md docs tests -q
uv run nox -s test           # per-package pytest sweep
uv run nox -s lint           # pre-commit/ruff/mypy per repo config
```

Expected: all green. Fix anything that surfaces (common stragglers: `__all__` sorting lint, unused-import from removed `BareQuantity` imports, pylint on the renamed modules).

- [ ] **Step 2: Grep for leftovers**

```bash
git grep -n "BareQuantity" -- 'src' 'tests' 'packages' 'docs' 'README.md' | grep -v 'docs/_build' | grep -viE 'deprecat|glossary|migration|renamed'
```

Expected: no output — every remaining `BareQuantity` mention must be deprecation/migration-related.

```bash
git grep -rn "unchecked" src/unxt | grep -v Binary
```

Expected: no references to the deleted module path.

- [ ] **Step 3: Downstream smoke check (informational, do not fix here)**

The wider ecosystem (`coordinax`, `galax`, user code) imports `unxt.Quantity` and gets the new class. Run a quick import-and-use smoke against the sibling checkout if available:

```bash
uv run python -c "
import unxt as u
q = u.Quantity([1., 2.], 'm')
print(u.uconvert('km', q))
print(u.PQ['length'](1., 'm'))
"
```

Report (don't act on) anything that suggests downstream repos need matching PRs.

- [ ] **Step 4: Commit any final fixes**

```bash
git add -A && git commit -m "💚 ci: final lint/test fixes for the v2 quantity rename"
```

Release itself (tag `v2.0.0` per RELEASING.md) is a separate human step — do not tag.
