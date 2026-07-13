(migration-v2)=

# Migrating to v2

This guide covers the breaking changes introduced in `unxt` v2, specifically the rename of the quantity classes. If you are starting fresh with `unxt`, you do not need this guide — consult the [Quantity guide](guides/quantity.md) for the current API.

---

## Class Rename Mapping

| v1 name | v2 name | Short alias | Notes |
| --- | --- | --- | --- |
| `BareQuantity` | `Quantity` | `u.Q` | Now the **default**, non-parametric class |
| `Quantity` (parametric) | `ParametricQuantity` | `u.PQ` | Opt-in; encodes dimension in type |
| — | `Q` | `u.Q` | New alias for `Quantity` |
| — | `PQ` | `u.PQ` | New alias for `ParametricQuantity` |

---

## Deprecation: `BareQuantity`

`BareQuantity` is now a **deprecated alias** of `Quantity`. Accessing it emits a `DeprecationWarning` and returns the (new) `Quantity` class:

<!-- skip: start -->

```python
import warnings
import unxt as u

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    BQ = u.quantity.BareQuantity  # DeprecationWarning
    assert BQ is u.Quantity  # same class

print(w[0].category.__name__)  # DeprecationWarning
```

<!-- skip: end -->

`BareQuantity` **will be removed in a future release**. Update your code now:

<!-- skip: start -->

```python
# Before (v1)
from unxt import BareQuantity

q = BareQuantity(1.0, "m")

# After (v2)
from unxt import Quantity  # or: import unxt as u; u.Q(...)

q = Quantity(1.0, "m")
```

<!-- skip: end -->

---

## Behavioral Changes for Former `Quantity` Users

If you used the old parametric `Quantity`, two behaviors have changed:

### (a) Subscripting no longer dimension-checks by default

In v1, `Quantity["length"](1, "s")` would raise a `ValueError` because `"s"` (seconds) is not a length unit. In v2, the same call on the new default `Quantity` accepts any unit without checking:

<!-- skip: start -->

```python
import unxt as u

# v2 default Quantity — subscript is a no-op, no check
u.Q["length"](1, "s")  # Quantity(1, unit='s') — no error

# ParametricQuantity — still raises on mismatch
try:
    u.PQ["length"](1, "s")
except ValueError as e:
    print(e)  # Physical type mismatch.
```

<!-- skip: end -->

**Migration:** Replace `Quantity["<dim>"](...)` calls that relied on dimension checking with `ParametricQuantity["<dim>"](...)` (or `u.PQ["<dim>"](...)`).

### (b) Plum dispatch on dimension-specific types requires `ParametricQuantity`

In v1, `Quantity["length"]` was a distinct class usable in `plum` dispatch annotations. In v2, `u.Q["length"] is u.Quantity` — subscripting the default `Quantity` returns the same class, making it useless for dimension-based dispatch.

<!-- skip: start -->

```python
# v1 dispatch on old Quantity (broken in v2)
# @dispatch
# def f(x: Quantity["length"]): ...  # was a distinct type in v1

# v2 — use ParametricQuantity for dimension-specific dispatch
from plum import dispatch
import unxt as u


@dispatch
def f(x: u.PQ["length"]):
    return "length!"


@dispatch
def f(x: u.PQ["time"]):
    return "time!"
```

<!-- skip: end -->

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

- **Runtime dimension checking**: `u.PQ["length"](1, "s")` raises immediately.
- **Dimension-specific `plum` dispatch**: `u.PQ["length"]` is a distinct type.

For everything else — arithmetic, unit conversion, JAX transforms, interop — `Quantity` and `ParametricQuantity` behave identically.

---

## Note on Pickles

Old pickles that reference the private module path `unxt._src.quantity.quantity.Quantity` resolve to the **new** `Quantity` class (i.e. the former `BareQuantity`). If you have pickles that stored instances of the old parametric `Quantity` (now `ParametricQuantity`), they will deserialize as the wrong type. Re-generate those pickles after upgrading.
