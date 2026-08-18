---
name: unxt
description: >
  Use when writing, reviewing, or debugging code that imports unxt (Quantity, ParametricQuantity, units, dims, unit systems), or that passes a unxt Quantity through JAX/quax/quaxed functions. Also use when a dimension mismatch error like "'yr' (time) and 'km' (length) are not convertible" appears, when plum raises an ambiguous-dispatch error on unit()/dimension()/ convert(), when Quantity["length"] silently accepts a mismatched unit instead of raising, when code reaches for a Quantity's private `_mk` constructor, when a StaticQuantity comparison behaves unit-blind, or when upgrading code that still references the deprecated BareQuantity.
---

# unxt

`unxt` gives JAX unitful quantities: `Quantity` is a `quax.ArrayValue` (an Equinox PyTree), so it flows through `jax.jit`/`vmap`/`grad` like any other JAX array-ish type, while carrying a unit and enforcing unit-safe arithmetic.

Checked against unxt 2.0.x (quax>=0.4.2, quax-blocks>=0.5.0, quaxed>=0.10.5, plum-dispatch>=2.7.0, astropy>=7.1), Python >=3.12. Docs: <https://unxt.readthedocs.io/en/>.

**Read the [quax](https://github.com/nstarman/quax/blob/main/skills/quax/SKILL.md), [quaxed](https://github.com/GalacticDynamics/quaxed/blob/main/skills/quaxed/SKILL.md), and [quax-blocks](https://github.com/GalacticDynamics/quax-blocks/blob/main/skills/quax-blocks/SKILL.md) skills first** for anything about `quaxify`, dispatch resolution, or the mixin operator overloads — this skill covers only what's specific to `unxt`, and doesn't restate any of it.

## Quick start

```pycon
>>> import jax.numpy as jnp
>>> import unxt as u

>>> velocity = u.Q(30.0, "m/s")
>>> time = u.Q(2.0, "s")
>>> velocity * time
Quantity(Array(60., dtype=float32, ...), unit='m')

>>> u.uconvert("km", velocity * time)
Quantity(Array(0.06, dtype=float32, ...), unit='km')
```

## `Quantity` vs `ParametricQuantity` vs `StaticQuantity` vs (deprecated) `BareQuantity`

| Class | Package | Dimension in type? | Checked at construction? | Use when |
| --- | --- | --- | --- | --- |
| `Quantity`/`u.Q` | `unxt` (default) | no — one pytree type for every dimension | no | almost always; the fast, general default |
| `ParametricQuantity`/`up.PQ` | `unxts.parametric` (opt-in) | yes, e.g. `PQ["length"]` | yes, raises on mismatch | you want a runtime guard or dimension-specific dispatch, and can accept a distinct pytree type per dimension |
| `StaticQuantity` | `unxt` | no | — | value must be `jax.jit(static_argnames=...)`-hashable; equality is **unit-label-based**, not physical-equivalence |
| `BareQuantity` | `unxt` | — | — | **deprecated**, alias of `Quantity` — don't use in new code, see the [migration guide](https://unxt.readthedocs.io/en/latest/migration.html) |

The subscript-without-checking trap: `u.Q["length"]` accepts the subscript syntax but performs **no** dimension check — it silently builds a `Quantity` with whatever unit you give it:

```pycon
>>> u.Q["length"](2, "s")  # wrong dimension, no error
Quantity(Array(2, dtype=int32, ...), unit='s')
```

```pycon
>>> import unxts.parametric as up
>>> up.PQ["length"](2, "s")  # doctest: +SKIP
# ValueError: Physical type mismatch.
```

If you need the guard, you need `unxts.parametric`'s `PQ`, not `u.Q[...]`.

## Functional API, operator-first argument order

The functional API is primary (the OO methods just call it). Argument order is inspired by Unitful.jl: **operator first, operand last** — read `uconvert("cm", q)` as "convert[to `cm`](q)":

```pycon
>>> q = u.Q(1, "m")
>>> u.uconvert("cm", q)  # function form — operator first
Quantity(Array(100., dtype=float32, ...), unit='cm')
>>> q.uconvert("cm")  # OO form — same result
Quantity(Array(100., dtype=float32, ...), unit='cm')
```

Don't write `uconvert(q, "cm")` — that's the wrong order and, depending on dispatch coverage, may raise a plum ambiguity/no-method error instead of silently doing the wrong thing.

## `_mk` is private API — do not use unless you mean it

`Quantity._mk` (and type-specific overrides like `QuantityMatrix._mk`) is **not exported and not covered by semver**. It writes the `value`/`unit` fields directly and skips _both_ the plum-dispatched value/unit converters _and_ `__check_init__` — the checks that make normal construction safe. It exists purely as a ~50x-faster hot-path constructor for code that has already proven its inputs are normalised.

**Warn the user explicitly before introducing `_mk` in code written for them.** It is only safe when:

- the `value` is already the right array type/dtype for this quantity's storage, and
- the `unit` is already a real `AbstractUnit` instance (not a string), and
- no dimension check was needed anyway (or was already performed by the caller).

Get any of that wrong and you get a `Quantity` that looks valid but violates its own invariants — e.g. a unit that's still a string, silently breaking every downstream dispatch that expects `AbstractUnit`. `StaticQuantity` overrides `_mk` back to the _checked_ constructor for exactly this reason: its converter is load-bearing, not redundant. If a value/unit pair isn't provably pre-normalised, use the normal constructor or `revalue`, not `_mk`.

## Dimensions reject `+`/`-`, not just anything unexpected

`u.dimension(...)` parses a small expression grammar: `* / ** ()` work, but unary `+`/`-` raise on purpose ("dimensions are invariant under negation") — this is a deliberate rejection, not a missing feature to "fix":

```pycon
>>> u.dimension("length / time")
PhysicalType({'speed', 'velocity'})
```

## This looks like a bug, it's intentional — don't "fix" it

- **`StaticQuantity` equality is unit-label-based, not physical.** `==` compares unit _labels_ (`same_unit_label`), not physical equivalence — two quantities with the same value but different unit spelling for the same physical unit can compare unequal, because equality must stay a valid `jax.jit` `static_argnames` key. Use `unxt.equivalent`/`is_equivalent` for physical-equivalence comparison instead.
- **`u.Q["length"]` doesn't check dimensions.** See above — that's `unxts.parametric.PQ`'s job, not `Quantity`'s.
- **Unit-system singletons (`si`, `cgs`, dimensionless) are deliberately shared, immutable objects** — code that used to be able to corrupt them by mutation was a bug (fixed in #704/#718); don't reintroduce mutable state on these.

## Which package do I need

| Need | Package |
| --- | --- |
| Runtime-checked, dimension-typed quantities (`PQ["length"]`) | `unxts.parametric` |
| Heterogeneous-unit matrices/vectors (`QuantityMatrix`/`QM`, `UnitsMatrix`) | `unxts.linalg` |
| `gala.units.UnitSystem` interop | `unxts.interop.gala` |
| Plotting `Quantity` with matplotlib | `unxts.interop.matplotlib` |
| xarray accessors for quantities | `unxts.interop.xarray` |
| Hypothesis strategies for property-based tests | `unxts.hypothesis` |
| Minimal-dependency abstract dispatch API only | `unxts.api` |

Each of these will get its own `skills/<pkg>/SKILL.md` in time; until then, their `docs/packages/<pkg>/` pages are the reference.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `'yr' (time) and 'km' (length) are not convertible` | Arithmetic/`uconvert` between incompatible dimensions — this is unxt working correctly; convert one side first or check you meant the operation. |
| `Physical type mismatch.` from `ParametricQuantity`/`PQ[...]` construction | The unit you passed doesn't match the type parameter's dimension. Only `PQ` checks this — `Quantity`/`u.Q` silently accepts it. |
| Ambiguous-dispatch / no-method error from `unit()`, `dimension()`, or `convert` | You passed a type with no registered conversion. Check `<func>.methods` to see what's registered, or convert to a supported type first (`str`, `AbstractUnit`, `astropy.units.Unit`, ...). |
| `u.Q["length"](2, "s")` returns a `Quantity` with the wrong dimension, no error | Expected — `Quantity` doesn't check dimensions on subscript construction. Use `unxts.parametric.PQ["length"]` if you need the guard. |
| A `Quantity` built via `_mk` behaves wrong downstream (wrong dtype, unit still a string, dispatch misses) | `_mk` was called with unnormalised input. Don't use `_mk` outside code that has already proven normalisation; use the checked constructor or `revalue`. |
| Doctest in a docstring/README/`docs/*.md` fails on dtype (`dtype=float32` vs `dtype=float`) | Sybil matches output exactly. Match the real dtype JAX produces, don't approximate it. |
| A warning you didn't expect fails the test suite | `filterwarnings = ["error", ...]` in `pyproject.toml` — either fix the cause or add a scoped, justified ignore; don't silence broadly. |

## Version notes

unxt v2.0 restructured the quantity hierarchy: `BareQuantity` is deprecated in favor of plain `Quantity`; dimension-parametrized quantities moved out to the separate `unxts.parametric` package (`PQ`); several previously-`unxt-*`-hyphenated packages have canonical `unxts.*` replacements (the hyphenated packages are now back-compat shims). See the [migration guide](https://unxt.readthedocs.io/en/latest/migration.html) for the full v1→v2 mapping before assuming an older code sample is still current.
