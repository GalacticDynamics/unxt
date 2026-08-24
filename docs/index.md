---
sd_hide_title: true
---

<h1> <code> unxt </code> </h1>

```{toctree}
:maxdepth: 1
:hidden:
:caption: 📖 Documentation

unxt <self>
tutorials/index
how-to/index
reference/index
explanation/index
about/index
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: 📦 Packages

unxts.api <packages/unxts.api/index>
unxts.hypothesis <packages/unxts.hypothesis/index>
unxts.parametric <packages/unxts.parametric/index>
unxts.linalg <packages/unxts.linalg/index>
unxts.interop.gala <packages/unxts.interop.gala/index>
unxts.interop.matplotlib <packages/unxts.interop.matplotlib/index>
unxts.interop.xarray <packages/unxts.interop.xarray/index>
```

# unxt

**Unitful quantities and calculations in [JAX][jax].**

`unxt` gives you arrays that carry their physical units, and keeps them carrying those units through everything JAX does: JIT compilation, vectorization, automatic differentiation, GPU and TPU. There are no unit-aware re-exports of JAX to learn. Your existing JAX code works, with one decorator.

```{code-block} python

>>> import unxt as u

>>> v = u.Q(25.0, "m/s")
>>> t = u.Q(3.0, "s")
>>> v * t
Quantity(Array(75., dtype=float32, weak_type=True), unit='m')

```

Install it with `pip install unxt` — see {doc}`how-to/install` for the other options.

---

## Where to start

<!-- prettier-ignore-start -->

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 🎓 I want to learn the basics
:link: tutorials/index
:link-type: doc

**Tutorials.** Guided lessons that build something from nothing and run it
through JAX. Start here if `unxt` is new to you.
:::

:::{grid-item-card} 🔧 I need to get something done
:link: how-to/index
:link-type: doc

**How-to guides.** Directions for one specific task: converting units, using JAX
functions, controlling display, migrating from v1, making it fast.
:::

:::{grid-item-card} 📇 I need to look something up
:link: reference/index
:link-type: doc

**Reference.** What every class, function and option is and does, plus the
generated API documentation and the glossary.
:::

:::{grid-item-card} 💡 I want to understand how it works
:link: explanation/index
:link-type: doc

**Discussion.** Why `unxt` is built the way it is, and where units and JAX
surprise people.
:::

::::

<!-- prettier-ignore-end -->

Looking for **`ParametricQuantity`**, unitful **linear algebra**, `hypothesis` strategies, or interop with `gala`, `matplotlib` or `xarray`? Those live in the separate `unxts.*` packages, listed under **Packages** in the sidebar.

Upgrading from v1? See {doc}`how-to/migrate-to-v2`.

---

For citation information and the surrounding ecosystem, see {doc}`about/index`.

<!-- LINKS -->

[jax]: https://jax.readthedocs.io/en/latest/
