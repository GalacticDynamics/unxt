---
jupytext:
  formats: md:myst
kernelspec:
  display_name: "Python 3"
  language: "python"
  name: "python3"
---

# How to optimize performance with unitful quantities

Quantities cost almost nothing inside a `jit`, and something measurable at its boundary. So the one move that matters is to put the `jit` boundary _outside_ the unit handling: accept raw arrays, build quantities inside the compiled function, and strip units off the result.

This guide measures that. Every timing below is produced by running the notebook, so the numbers are from the machine that built this page, not from a previous run. For _why_ a quantity costs anything at a boundary, see {doc}`../explanation/sharp-bits`.

## Set up

```{code-cell} ipython3
import functools as ft

import jax
import jax.numpy as jnp
import quax

import unxt as u
```

Thousand-element arrays, with and without units:

```{code-cell} ipython3
x = jnp.linspace(0.1, 10.0, 1000)
y = jnp.linspace(11, 100.0, 1000)

qx = u.Q(x, "m")
qy = u.Q(y, "m")
```

## Measure the baseline

Plain JAX on raw arrays, uncompiled and compiled. `%time` is the first call, which includes compilation; `%timeit` is the steady state.

```{code-cell} ipython3
def func(x, y):
    return jnp.sum((x ** 3 - y ** 3) / (x**2 + y**2))

%time jax.block_until_ready(func(x, y))
%timeit jax.block_until_ready(func(x, y))
```

```{code-cell} ipython3
jitted_func = jax.jit(func)

%time jax.block_until_ready(jitted_func(x, y))
%timeit jax.block_until_ready(jitted_func(x, y))
```

## Add units, and see what they cost

`quax.quaxify` lets the same function take quantities. Uncompiled, it unwraps each quantity, tracks the units and re-wraps the result on every call:

```{code-cell} ipython3
quax_func = quax.quaxify(func)

%time jax.block_until_ready(quax_func(qx, qy))
%timeit jax.block_until_ready(quax_func(qx, qy))
```

That is the overhead to avoid. Compiling it away recovers most of the gap, because `jit` traces through the dispatch and the wrapping:

```{code-cell} ipython3
jitted_quax_func = jax.jit(quax.quaxify(func))

%time jax.block_until_ready(jitted_quax_func(qx, qy))
%timeit jax.block_until_ready(jitted_quax_func(qx, qy))
```

A gap remains, and it is not the arithmetic. Quantities are JAX pytrees, so each one is decomposed on the way in and recomposed on the way out — work that happens at the boundary, where `jit` cannot reach it.

## Move the boundary outside the units

Close that gap by taking raw arrays at the outermost boundary and doing the unit handling inside the compiled function:

1. Accept raw arrays.
2. Build quantities from them, inside the `jit`.
3. Call the unitful function.
4. Strip the units off the result.

```{code-cell} ipython3
@jax.jit
def outer_func(x, y):
    qx = u.Q(x, "m")
    qy = u.Q(y, "m")
    out = jitted_quax_func(qx, qy)
    return out.ustrip("m")

%time jax.block_until_ready(outer_func(x, y))
%timeit jax.block_until_ready(outer_func(x, y))
```

That is back to near the raw-JAX baseline. Every pytree conversion is now inside the trace, so the compiler removes it along with the rest of the unit bookkeeping.

The inner function does not have to be a single call. Here it makes two and combines them; the outer wrapper still hoists every conversion inside one boundary, so the extra call costs nothing at the interface:

```{code-cell} ipython3
def func2(x, y):
    out1 = jitted_quax_func(x, y)
    out2 = jitted_quax_func(y, x)
    return out1 + 3 * out2

jitted_quax_func2 = jax.jit(quax.quaxify(func2))
```

## Choose the units at call time

To let one wrapper serve several unit choices, pass a {class}`~unxt.unitsystems.AbstractUnitSystem` as a static argument. Build the inputs with {meth}`u.Q.from_ <unxt.quantity.Quantity.from_>` and `usys["length"]`, and strip the result with `out.ustrip(usys)`:

```{code-cell} ipython3
@ft.partial(jax.jit, static_argnames=("usys",))
def outer_func3(x, y, *, usys):
    qx = u.Q.from_(x, usys["length"])
    qy = u.Q.from_(y, usys["length"])
    out = jitted_quax_func2(qx, qy)
    return out.ustrip(usys)
```

```{code-cell} ipython3
usys = u.unitsystems.si

%time jax.block_until_ready(outer_func3(x, y, usys=usys))
%timeit jax.block_until_ready(outer_func3(x, y, usys=usys))
```

The unit system is static, so each distinct one compiles its own specialization and then runs at the same near-baseline speed.

## Build the jitted wrapper once

Define the wrapper once, outside any loop or method, and call that same object every time. `jax.jit` caches on the _identity_ of the function it wraps, not on argument equality, so a wrapper rebuilt per call misses the cache every time — a full retrace and recompile, not a cheap re-trace:

```{code-cell} ipython3
def make_outer():
    @jax.jit
    def outer_func(x, y):
        qx = u.Q(x, "m")
        qy = u.Q(y, "m")
        return quax.quaxify(func)(qx, qy).ustrip("m")

    return outer_func


def rebuild_each_call(x, y):
    fn = make_outer()
    return jax.block_until_ready(fn(x, y))


%timeit rebuild_each_call(x, y)
%timeit jax.block_until_ready(outer_func(x, y))
```

Tens of milliseconds against low microseconds — several hundred times slower. Every jitted function in this guide is defined exactly once, which is why the timings above measure the compiled code rather than the compiler.

## When this does not matter

The boundary cost is fixed per outermost call, not per element. One call on a million-element array barely notices it; a million calls on scalars pay it a million times. So it is worth the wrapper on a hot outer function and not worth it anywhere else.

Write the correct version first. If units make the code clearer, use them — the rest of a codebase can stay unit-aware and unoptimized, and only the outermost layer of a measured hot path needs this pattern.
