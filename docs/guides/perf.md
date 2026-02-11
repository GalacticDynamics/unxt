---
jupytext:
  formats: md:myst
kernelspec:
  display_name: "Python 3"
  language: "python"
  name: "python3"
---

# Performance Optimization with Unitful Quantities

In this guide, we'll explore how to think about performance optimization when working with `unxt` Quantities in JAX. The key insight is understanding **where the overhead lives** and **when it matters**.

## Key Concepts

1. **Wrapper overhead**: Operations on Quantities have overhead compared to raw JAX arrays -- they're wrapped with unit information.
2. **JIT removes overhead**: JAX's JIT compiler can eliminate much of this wrapper overhead by tracing through the code.
3. **Pytree complexity**: Quantities are JAX pytrees, which adds cost when crossing JIT boundaries (converting between traced and non-traced values).
4. **Strategy**: The secret to performance is to **minimize pytree conversions at the boundary** between traced and non-traced code.

Let's start by importing the libraries we'll need and setting up some test data.

```{code-cell} ipython3
import functools as ft

import jax
import jax.numpy as jnp
import quax

import unxt as u
```

We'll create 1000-element arrays with physical units -- these will be our test data throughout this guide.

```{code-cell} ipython3
x = jnp.linspace(0.1, 10.0, 1000)
y = jnp.linspace(11, 100.0, 1000)

qx = u.Q(x, "m")
qy = u.Q(y, "m")
```

## Baseline: Raw JAX Performance

First, let's establish our baseline by measuring the performance of a plain JAX function with raw arrays. We'll time both:

1. **First call**: Includes JIT compilation overhead
2. **Repeated calls**: Shows the performance after compilation

```{code-cell} ipython3
def func(x, y):
    return jnp.sum((x ** 3 - y ** 3) / (x**2 + y**2))

%time jax.block_until_ready(func(x, y))
%timeit jax.block_until_ready(func(x, y))
```

### With JIT Compilation

Now let's compile the same function with `jax.jit`. Notice the dramatic speedup; JIT compilation converts Python loops and operations into optimized GPU/CPU kernels.

**Key insight**: JIT is almost always worth it. The first call takes longer due to compilation, but subsequent calls are much faster.

```{code-cell} ipython3
jitted_func = jax.jit(func)

%time jax.block_until_ready(jitted_func(x, y))
%timeit jax.block_until_ready(jitted_func(x, y))
```

### `quaxify` for unit support

Now let's apply the same function to Quantities using `quax.quaxify`. This wraps the function so it can handle Quantity inputs.

```{code-cell} ipython3
quax_func = quax.quaxify(func)

%time jax.block_until_ready(quax_func(qx, qy))
%timeit jax.block_until_ready(quax_func(qx, qy))
```

#### The Problem: Wrapper Overhead

Wow! That's much slower than the JIT'd version. This is the **wrapper overhead** in action. The `quaxify` decorator has to:

1. Unwrap the Quantities into arrays
2. Track unit information
3. Re-wrap the result back into a Quantity
4. Do all of this EVERY time the function is called, without JIT optimization

This is why JIT is a necessary ingredient -- let's see what happens when we add JIT:

#### Solution 1: JIT the Quaxified Function

By combining `jax.jit` with `quax.quaxify`, we eliminate much of the wrapper overhead. JIT compiles away the dynamic dispatch and wrapping logic.

```{code-cell} ipython3
jitted_quax_func = jax.jit(quax.quaxify(func))

%time jax.block_until_ready(jitted_quax_func(x, y))
%timeit jax.block_until_ready(jitted_quax_func(x, y))
```

Now let's pass actual Quantities to the jitted function. Notice there's some overhead compared to raw arrays, but it's much better than the non-JIT version!

```{code-cell} ipython3
%time jax.block_until_ready(jitted_quax_func(qx, qy))
%timeit jax.block_until_ready(jitted_quax_func(qx, qy))
```

#### Solution 2: Minimize Pytree Conversions at the Boundary

There's still a small overhead when passing Quantities across the JIT boundary. This is because **Quantities are JAX pytrees** -- they need to be decomposed before tracing and recomposed after.

The trick? Move the pytree conversion **inside** the JIT. Here's the key insight: we create a thin outer JIT'd wrapper that converts arrays to Quantities at the start, calls the inner unitful function, and extracts the result. This way, all the pytree overhead is inside the JIT boundary where it gets compiled away.

```{code-cell} ipython3
@jax.jit
def outer_func(x, y):
    qx = u.Q(x, "m")
    qy = u.Q(y, "m")

    # This calls the unitful function inside a jitted context
    out = jitted_quax_func(qx, qy)

    return out.ustrip("m")

%time jax.block_until_ready(outer_func(x, y))
%timeit jax.block_until_ready(outer_func(x, y))
```

```{code-cell} ipython3
def func2(x, y):
    out1 = jitted_quax_func(x, y)
    out2= jitted_quax_func(x, y)
    return out1 + 3 * out2

jitted_quax_func2 = jax.jit(quax.quaxify(func2))
```

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

### The Strategy: Outer Wrapper Pattern

Here's the optimal pattern:

1. **Accept raw arrays** at the outermost function boundary
2. **Create Quantities inside the JIT** by wrapping the arrays with units
3. **Call the inner unitful function** inside the JIT
4. **Extract the result** and return it as a raw array (or strip units if needed)

This way, all unit handling is compiled away, and you only pay the "cost of thinking in units" during actual computation -- not at function call boundaries.

### Results: Overhead Eliminated

Wow! Notice the dramatic speedup—we're nearly as fast as the raw JAX version!

**Why this works:**

- The outer JIT compiles away all the unit wrapping/unwrapping
- The `jitted_quax_func` runs inside the trace as a compiled operation
- The only overhead is JIT's normal pytree handling, which is minimal

**Important caveat:** This is a *fixed* cost that only appears once per outermost function call. If your function is called once with a million-element array, this optimization is huge. If your function is called a million times with scalar inputs, the overhead per element is negligible.

## Summary: How to Think About Performance

Here are the key takeaways for optimizing performance with `unxt` Quantities:

1. **Always use JIT for hot code** - The overhead of `quaxify` is negligible inside a JIT'd context
2. **Minimize pytree boundary crossings** - Use the outer wrapper pattern where you pass raw arrays to the outermost function
3. **Create Quantities inside JIT** - This lets the compiler optimize away unit handling
4. **It's a fixed cost per call** - The optimization matters more for functions that process large arrays or are called infrequently
5. **Don't microoptimize prematurely** - Write correct code first. If units make your code clearer, use them. Only optimize the outermost layer if profiling shows it's necessary.

The bottom line: **Use Quantities freely in your code—they're designed to work well with JAX.** When you need performance, apply the outer wrapper pattern to your hot functions. The rest of your codebase can stay clean and unit-aware.
