(migration-v2)=

# Migrating to v2

This guide covers the breaking changes introduced in `unxt` v2: the rename of the quantity classes and the extraction of the parametric quantity into the separate `unxts.parametric` package. If you are starting fresh with `unxt`, you do not need this guide — consult the [Quantity guide](guides/quantity.md) and the [parametric quantity guide](packages/unxts.parametric/index.md) for the current API.

---

## Class Rename Mapping

| v1 name | v2 name | Short alias | Notes |
| --- | --- | --- | --- |
| `BareQuantity` | `Quantity` | `u.Q` | Now the **default**, non-parametric class |
| `Quantity` (parametric) | `ParametricQuantity` | `up.PQ` | Opt-in; **moved** to the `unxts.parametric` package |
| — | `Q` | `u.Q` | New alias for `Quantity` |
| — | `PQ` | `up.PQ` | New alias for `ParametricQuantity` (in `unxts.parametric`) |

```{note}
As of v2, `ParametricQuantity`/`PQ` live in the separate **`unxts.parametric`**
package rather than in `unxt`. Install it with `pip install unxts.parametric`
and import as `import unxts.parametric as up` (so `up.PQ`). Accessing
`unxt.ParametricQuantity` / `u.PQ` now raises an `AttributeError` pointing here.
```

---

## Package Split: `ParametricQuantity` Moved to `unxts.parametric`

In v2 the parametric quantity classes live in a **separate package**, `unxts.parametric`, rather than in `unxt`. Core `unxt` no longer imports or depends on `ParametricQuantity` at all. Install the package to opt in:

```bash
pip install unxts.parametric   # or: uv add unxts.parametric
```

Accessing the moved names on `unxt` now raises `AttributeError` with a message pointing to the new package — this covers `unxt.ParametricQuantity`, `unxt.PQ`, `unxt.AbstractParametricQuantity`, and their `unxt.quantity.*` equivalents.

### Update your imports

| v1 (`unxt`) | v2 (`unxts.parametric`) |
| --- | --- |
| `from unxt import ParametricQuantity` | `from unxts.parametric import ParametricQuantity` |
| `from unxt import PQ` | `from unxts.parametric import PQ` |
| `from unxt.quantity import AbstractParametricQuantity` | `from unxts.parametric import AbstractParametricQuantity` |
| `u.PQ(...)` / `u.ParametricQuantity(...)` | `up.PQ(...)` (with `import unxts.parametric as up`) |

The v1 form (shown for reference — this raises `AttributeError` on v2):

<!-- skip: start -->

```python
import unxt as u

q = u.PQ(1, "m")
```

<!-- skip: end -->

The v2 form:

```{code-block} python
>>> import unxts.parametric as up

>>> up.PQ(1, "m")
ParametricQuantity(Array(1, dtype=int32...), unit='m')
```

### Angle operations now return the default `Quantity`

Trigonometric and product operations on an `Angle` (`cos`, `sin`, `tan`, `cbrt`, `Angle @ Angle`, `Angle * Angle`, integer/array powers, etc.) previously produced a `ParametricQuantity`. Because core `unxt` can no longer reference the parametric class, in v2 they produce the lightweight default `Quantity`:

```{code-block} python
>>> import unxt as u
>>> import quaxed.numpy as jnp

>>> jnp.cos(u.Angle(0, "deg"))  # v1: ParametricQuantity
Quantity(Array(1., dtype=float32...), unit='')

>>> u.Angle([1, 2, 3], "deg") @ u.Angle([4, 5, 6], "deg")  # v1: ParametricQuantity
Quantity(Array(32, dtype=int32), unit='deg2')
```

The value and unit are unchanged — only the wrapping class differs. If you specifically need a parametric result, convert explicitly with `convert(result, up.PQ)`.

### Parametric operands need `unxts.parametric` imported

A few JAX primitive rules fire only when a _parametric_ quantity is involved — raising a quantity to a dimensionless `ParametricQuantity` exponent, `%` (remainder), and `clamp` with parametric bounds. These rules are registered as an import side effect of `unxts.parametric`. Importing the package (which you do to use `up.PQ` at all) registers them; if a `ParametricQuantity` reaches your code some other way, `import unxts.parametric` once at startup.

### Astropy conversion

Converting an `astropy.units.Quantity` **to a `ParametricQuantity`** is now registered by `unxts.parametric` (import it to enable). Conversion to the default `Quantity` remains in core `unxt`:

```{code-block} python
>>> from astropy.units import Quantity as AstropyQuantity
>>> from plum import convert
>>> import unxt as u
>>> import unxts.parametric as up

>>> convert(AstropyQuantity(1.0, "cm"), u.Quantity)  # core unxt
Quantity(Array(1., dtype=float32), unit='cm')

>>> convert(AstropyQuantity(1.0, "cm"), up.PQ)  # needs unxts.parametric
ParametricQuantity(Array(1., dtype=float32), unit='cm')
```

### Config: `include_params` moved to `unxts.parametric.config`

The `include_params` display option — whether `repr()`/`str()` show the `['length']`-style dimension parameter — only affects parametric quantities, so it moved out of `unxt.config` into `unxts.parametric.config`. `unxt.config` now rejects it as an unknown option.

| v1 (`unxt.config`) | v2 (`unxts.parametric.config`) |
| --- | --- |
| `u.config.quantity_repr.include_params` | `up.config.quantity_repr.include_params` |
| `u.config.override(quantity_repr__include_params=True)` | `up.config.override(quantity_repr__include_params=True)` |
| `[tool.unxt.quantity.repr]` → `include_params` | `[tool.unxts.parametric.quantity.repr]` → `include_params` |

Defaults are unchanged (`repr` hides the parameter, `str` shows it). The other display settings (`short_arrays`, `use_short_name`, `named_unit`, `indent`) remain in `unxt.config`. See the [parametric quantity guide](packages/unxts.parametric/index.md#configuration).

### Config file section renamed to `[tool.unxts.unxt]`

`unxt`'s `pyproject.toml` display configuration moved from `[tool.unxt...]` to `[tool.unxts.unxt...]`, matching the shared `unxts.*` namespace (alongside `[tool.unxts.parametric...]`). Rename any existing sections:

| v1                          | v2                                |
| --------------------------- | --------------------------------- |
| `[tool.unxt.quantity.repr]` | `[tool.unxts.unxt.quantity.repr]` |
| `[tool.unxt.quantity.str]`  | `[tool.unxts.unxt.quantity.str]`  |

In-code configuration via `u.config` is unchanged.

---

## Deprecation: `BareQuantity`

`BareQuantity` is now a **deprecated alias** of `Quantity`. Accessing it emits a `DeprecationWarning` and returns the (new) `Quantity` class:

```{code-block} python
>>> import warnings
>>> import unxt as u

>>> with warnings.catch_warnings(record=True) as w:
...     warnings.simplefilter("always")
...     BQ = u.quantity.BareQuantity  # DeprecationWarning
...     assert BQ is u.Quantity  # same class

>>> w[0].category.__name__
'DeprecationWarning'
```

`BareQuantity` **will be removed in a future release**. Update your code now:

The v1 form (deprecated; still works, but warns):

<!-- skip: start -->

```python
from unxt import BareQuantity

q = BareQuantity(1.0, "m")
```

<!-- skip: end -->

The v2 form:

```{code-block} python
>>> from unxt import Quantity  # or: import unxt as u; u.Q(...)

>>> Quantity(1.0, "m")
Quantity(Array(1., dtype=float32...), unit='m')
```

---

## Behavioral Changes for Former `Quantity` Users

If you used the old parametric `Quantity`, two behaviors have changed:

### (a) Subscripting no longer dimension-checks by default

In v1, `Quantity["length"](1, "s")` would raise a `ValueError` because `"s"` (seconds) is not a length unit. In v2, the same call on the new default `Quantity` accepts any unit without checking:

On the v2 default `Quantity` the subscript is a no-op, so no check happens:

```{code-block} python
>>> import unxt as u
>>> import unxts.parametric as up

>>> u.Q["length"](1, "s")
Quantity(Array(1, dtype=int32...), unit='s')
```

`ParametricQuantity` still raises on a mismatch:

```{code-block} python
>>> try:
...     up.PQ["length"](1, "s")
... except ValueError as e:
...     print(e)
Physical type mismatch.
```

**Migration:** Replace `Quantity["<dim>"](...)` calls that relied on dimension checking with `ParametricQuantity["<dim>"](...)` (or `up.PQ["<dim>"](...)`).

### (a2) `isinstance` checks against `Quantity` no longer match parametric quantities

```{warning}
This is the one v1 break that gives you **neither a warning nor an error** — the
check simply returns `False` and your code takes the other branch.
```

In v1, `u.Quantity` _was_ the parametric class, so `isinstance(x, u.Quantity)` matched parametric instances. In v2 they are unrelated concrete classes, and neither is a subclass of the other:

```{code-block} python
>>> import unxt as u
>>> import unxts.parametric as up

>>> pq = up.PQ(1.0, "m")
>>> q = u.Q(1.0, "m")

>>> isinstance(pq, u.Quantity)   # v1: True
False

>>> isinstance(q, up.PQ)
False
```

**Migration:** check against the abstract base, which both concrete classes share:

```{code-block} python
>>> isinstance(pq, u.AbstractQuantity)
True

>>> isinstance(q, u.AbstractQuantity)
True
```

The same applies to `issubclass`, to `plum`/`beartype` annotations written as `Quantity`, and to any runtime validation (`pydantic`, `attrs` validators, manual type guards) that narrows on `u.Quantity`. If you want "any quantity", use `u.AbstractQuantity` (or `u.quantity.is_any_quantity`); reserve `u.Quantity` for "specifically the non-parametric class".

### (b) Plum dispatch on dimension-specific types requires `ParametricQuantity`

In v1, `Quantity["length"]` was a distinct class usable in `plum` dispatch annotations. In v2, `u.Q["length"] is u.Quantity` — subscripting the default `Quantity` returns the same class, making it useless for dimension-based dispatch.

So an annotation written this way no longer selects on dimension:

<!-- skip: start -->

```python
@dispatch
def f(x: Quantity["length"]):
    ...  # was a distinct type in v1
```

<!-- skip: end -->

Use `ParametricQuantity` for dimension-specific dispatch instead:

```{code-block} python
>>> from plum import dispatch
>>> import unxts.parametric as up

>>> @dispatch
... def f(x: up.PQ["length"]):
...     return "length!"

>>> @dispatch
... def f(x: up.PQ["time"]):
...     return "time!"

>>> f(up.PQ(1.0, "m"))
'length!'

>>> f(up.PQ(1.0, "s"))
'time!'
```

---

## Why the Default Changed: Pytree Types, Not JIT Cache Misses

The motivation for making the non-parametric class the default is how it interacts with JAX's pytree machinery — but it is worth being precise about what does and does not change.

`ParametricQuantity` encodes the physical dimension in its _type_: `ParametricQuantity["length"]` and `ParametricQuantity["time"]` are distinct Python classes, created on demand, and each is registered as its own JAX pytree node type. The new default `Quantity` is a _single_ class — and a single pytree node type — for every dimension.

**What this does _not_ change: `jax.jit` recompilation.** A quantity's `unit` is a _static_ field, so it lives in the pytree aux data (the treedef), which is part of the `jit` cache key. A jitted function therefore specializes per distinct unit with _either_ class — a function compiled for a length quantity in metres is **not** reused for a time quantity in seconds, because their units (and treedefs) differ. Since a unit already implies its dimension, `ParametricQuantity`'s per-dimension _class_ is redundant with the per-unit key and adds no extra compilations. The two classes produce the same number of `jit` compilations for the same inputs.

**What it _does_ change: the cost of the type proliferation itself.** With `ParametricQuantity`, every physical dimension you touch:

- creates a new Python class the first time it is used (via `plum`'s parametric machinery),
- registers a new JAX pytree node type and grows `plum`'s dispatch tables, all of which must be tracked and searched, and
- pays a per-construction cost for dimension inference and the `__check_init__` validation.

The single-class `Quantity` avoids all of that: one class, one registered pytree type, no on-the-fly class creation, and a lighter construction/dispatch path. That is the efficiency win — a smaller, simpler type surface and cheaper per-operation overhead — rather than fewer `jit` compilations.

`ParametricQuantity` remains available — and is the right choice — when you genuinely need:

- **Runtime dimension checking**: `up.PQ["length"](1, "s")` raises immediately.
- **Dimension-specific `plum` dispatch**: `up.PQ["length"]` is a distinct type.

For everything else — arithmetic, unit conversion, JAX transforms, interop — `Quantity` and `ParametricQuantity` behave identically.

---

## Note on Pickles

Old pickles that reference the private module path `unxt._src.quantity.quantity.Quantity` resolve to the **new** `Quantity` class (i.e. the former `BareQuantity`). If you have pickles that stored instances of the old parametric `Quantity` (now `ParametricQuantity`), they will deserialize as the wrong type. Re-generate those pickles after upgrading.
