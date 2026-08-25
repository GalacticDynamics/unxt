# `unxts.api`

```{toctree}
:maxdepth: 1
:hidden:

tutorial-your-own-type
why-abstract-dispatch
extending
api
```

Abstract dispatch API for [unxt](https://github.com/GalacticDynamics/unxt).

:::{note}

`unxts.api` is the canonical package. The legacy `unxt-api` distribution remains available as a thin backward-compatible shim that re-exports this package, so existing `import unxt_api` code keeps working unchanged.

:::

{mod}`unxts.api` declares the abstract dispatch interfaces that {mod}`unxt` and other packages implement. It depends only on {mod}`plum` — not on {mod}`jax`, {mod}`numpy` or {mod}`astropy` — so a package can speak `unxt`'s API without pulling in its implementation.

## Install

::::{tab-set}

:::{tab-item} uv

```bash
uv add unxts.api
```

:::

:::{tab-item} pip

```bash
pip install unxts.api
```

:::

::::

## At a glance

`unxts.api` declares the function; `unxt` registers the implementation that runs.

```python
import unxt as u

q = u.Q(5, "m")
u.uconvert("km", q)
```

`u.uconvert` here is `unxt`'s registered implementation of the abstract `unxts.api.uconvert`. To make your own type work with the same call, you register an implementation of your own — see [Extending](extending).

## Pages

**Tutorial**

- [Teach unxt about your own type](tutorial-your-own-type) — start here: make `unit_of`, `dimension_of`, `ustrip` and `uconvert` work on a class of your own.

**Discussion**

- [Why an abstract dispatch API](why-abstract-dispatch) — what the separation between interface and implementation buys, and what it costs.

**How-to**

- [How to extend unxt with your own types](extending) — registering implementations, the common patterns, and debugging dispatch.

**Reference**

- [API](api) — every abstract function, its signature and its registered implementations in `unxt`.

## Public API

The abstract functions, by domain:

| Domain | Functions |
| --- | --- |
| Dimensions | {func}`~unxts.api.dimension`, {func}`~unxts.api.dimension_of` |
| Units | {func}`~unxts.api.unit`, {func}`~unxts.api.unit_of` |
| Quantities | {func}`~unxts.api.uconvert`, {func}`~unxts.api.uconvert_value`, {func}`~unxts.api.ustrip`, {func}`~unxts.api.is_unit_convertible`, {func}`~unxts.api.wrap_to` |
| Unit systems | {func}`~unxts.api.unitsystem_of` |

Every one is a [plum](https://beartype.github.io/plum/) dispatch function, so `f.methods` lists what is currently registered:

```python
import unxts.api as uapi

uapi.dimension.methods
uapi.unit_of.methods
uapi.uconvert.methods
uapi.uconvert_value.methods
```

## See also

- [unxt documentation](https://unxt.readthedocs.io/) — the concrete implementation.
- [plum documentation](https://beartype.github.io/plum/) — the dispatch library.
- [unxt on GitHub](https://github.com/GalacticDynamics/unxt)
