# Your first calculation with units

In this tutorial we will build a small projectile-range calculator in which every number carries its physical units, and then run it through JAX's compiler and autodifferentiation. By the end you will have a function you can re-run, on vectors of inputs, on any planet.

You need `unxt` installed and nothing else — no accounts, no data files, no network.

## Make a quantity

Everything in `unxt` starts with a `Quantity`: a number and a unit, together. Let's make one for the speed a projectile leaves the ground at.

```{code-block} python
>>> import unxt as u

>>> v0 = u.Q(25.0, "m/s")
>>> v0
Quantity(Array(25., dtype=float32, weak_type=True), unit='m / s')

```

`u.Q` is shorthand for `u.Quantity`. Notice the value became a JAX `Array` — that happened automatically, and it is what will let us hand this to JAX later. Notice too that the unit printed back as `'m / s'` rather than the `"m/s"` we typed: the string was parsed into a real unit object.

The two halves are available separately:

```{code-block} python
>>> v0.value
Array(25., dtype=float32, weak_type=True)

>>> v0.unit
Unit("m / s")

```

Now let's make a second quantity, for gravity:

```{code-block} python
>>> g = u.Q(9.81, "m/s2")
>>> g
Quantity(Array(9.81, dtype=float32, weak_type=True), unit='m / s2')

```

## Watch the units do arithmetic

Here is the part worth slowing down for. Divide the speed by the acceleration:

```{code-block} python
>>> v0 / g
Quantity(Array(2.54842, dtype=float32, weak_type=True), unit='s')

```

We never told `unxt` that speed divided by acceleration is a time. It worked that out from the units: `(m/s) / (m/s²)` cancels down to `s`. Try squaring the speed first:

```{code-block} python
>>> v0**2 / g
Quantity(Array(63.710495, dtype=float32, weak_type=True), unit='m')

```

That one came out in metres — which is a promising sign, because a range is a length.

The same bookkeeping catches mistakes. Adding a speed to an acceleration is meaningless, and `unxt` says so instead of quietly returning a number:

```{code-block} python
>>> try: v0 + g
... except Exception as e: print(e)
'm / s2' (acceleration) and 'm / s' (speed/velocity) are not convertible

```

## Add an angle

Launch angle needs a unit too. `unxt` has a dedicated `Angle` class:

```{code-block} python
>>> theta = u.Angle(45.0, "deg")
>>> theta
Angle(Array(45., dtype=float32, weak_type=True), unit='deg')

```

Trigonometric functions want radians, so let's convert. `uconvert` returns a new quantity in the units you name:

```{code-block} python
>>> theta_rad = theta.uconvert("rad")
>>> theta_rad
Angle(Array(0.7853982, dtype=float32, weak_type=True), unit='rad')

```

The value changed from `45.` to `0.7853982` and the unit label changed with it. That pairing is the whole idea: a quantity's number is only ever meaningful together with its unit, so `unxt` never changes one without the other.

## Write the calculator

The range of a projectile launched at speed $v_0$ and angle $\theta$ under gravity $g$ is $v_0^2 \sin(2\theta) / g$. We will write it with `quaxed.numpy`, which is JAX's `numpy` with quantity support already switched on:

```{code-block} python
>>> import quaxed.numpy as jnp

>>> def projectile_range(v0, theta, g):
...     return v0**2 * jnp.sin(2 * theta) / g

```

Call it:

```{code-block} python
>>> r = projectile_range(v0, theta_rad, g)
>>> r
Quantity(Array(63.710495, dtype=float32, weak_type=True), unit='m')

```

About 64 metres. Notice we never wrote `"m"` anywhere in the function — the metres came out of the arithmetic, and they are the units the formula actually produces.

Ask for it in kilometres instead:

```{code-block} python
>>> r.uconvert("km")
Quantity(Array(0.0637105, dtype=float32, weak_type=True), unit='km')

```

Or, when you want to hand a plain number to something that does not speak units, `ustrip` converts and unwraps in one step:

```{code-block} python
>>> r.ustrip("km")
Array(0.0637105, dtype=float32, weak_type=True)

```

## Compile it

Our function is ordinary JAX, so `jax.jit` compiles it:

```{code-block} python
>>> import jax

>>> fast_range = jax.jit(projectile_range)
>>> fast_range(v0, theta_rad, g)
Quantity(Array(63.710495, dtype=float32, weak_type=True), unit='m')

```

Same answer, same units. The first call did the compiling; later calls reuse it.

## Differentiate it

Now let's ask a question the units answer for us: how much further does the projectile go per extra metre-per-second of launch speed? That is the derivative of the range with respect to `v0`. `quaxed` provides `grad` with quantity support already applied:

```{code-block} python
>>> import quaxed as qjax

>>> qjax.grad(projectile_range)(v0, theta_rad, g)
Quantity(Array(5.09684, dtype=float32, weak_type=True), unit='s')

```

Look at the unit: **seconds**. We differentiated metres with respect to metres-per-second, and `unxt` carried that through to `m / (m/s) = s` without being told. The number says that at 25 m/s, each extra metre-per-second buys about 5.1 more metres of range.

## Run it on many inputs, and on the Moon

The function was never written for a single value. Hand it an array of speeds:

```{code-block} python
>>> speeds = u.Q(jnp.asarray([10.0, 25.0, 40.0]), "m/s")
>>> projectile_range(speeds, theta_rad, g)
Quantity(Array([ 10.19368 ,  63.710495, 163.09888 ], dtype=float32), unit='m')

```

And because gravity is a parameter rather than a constant baked into the formula, the same function works elsewhere. Lunar surface gravity is about 1.62 m/s²:

```{code-block} python
>>> projectile_range(v0, theta_rad, u.Q(1.62, "m/s2"))
Quantity(Array(385.80246, dtype=float32, weak_type=True), unit='m')

```

Roughly six times further, which is about what you would expect from gravity being about six times weaker.

## What we built

You now have a unit-aware function that compiles, differentiates and vectorises, and in which no number ever lost track of what it measures. You wrote no unit conversions inside it, and no unit bookkeeping at all — `unxt` did that, and would have raised if the physics had not lined up.

## Where to go next

- {doc}`../how-to/convert-units` — the full conversion surface, including conversion without wrapping.
- {doc}`../how-to/use-jax-functions` — using `unxt` with JAX beyond `quaxed`.
- {doc}`../reference/quantity` — every method on `Quantity`.
- {doc}`../explanation/sharp-bits` — the places where units and JAX surprise you.
