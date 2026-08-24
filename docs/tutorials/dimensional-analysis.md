# Check a formula before you run it

In this tutorial we will catch mistakes in physics formulas without computing anything. No arrays, no units, no numbers — just dimensions. By the end you will have a small consistency checker you can point at your own expressions, and you will have used it to derive a result rather than merely verify one.

You need `unxt` installed and nothing else.

## Set up

```{code-block} python
>>> import unxt as u

```

## A dimension is something you can compute with

`unxt.dimension` turns a name into a dimension object:

```{code-block} python
>>> u.dimension("length")
PhysicalType('length')

```

The useful part is that it also reads expressions:

```{code-block} python
>>> u.dimension("length / time")
PhysicalType({'speed', 'velocity'})

```

We asked for length over time and got back _speed_. `unxt` did the algebra and then recognised the answer. Nothing here is a measurement — there is no number anywhere on this page yet.

## Check a formula you believe

Kinetic energy is $\frac{1}{2} m v^2$. Let's ask what that combination _is_, dimensionally. The $\frac{1}{2}$ is a pure number and cannot affect dimensions, so we leave it out:

```{code-block} python
>>> u.dimension("mass * (length/time)**2")
PhysicalType({'energy', 'torque', 'work'})

```

Energy, as promised. We can assert it rather than read it:

```{code-block} python
>>> u.dimension("mass * (length/time)**2") == u.dimension("energy")
True

```

Notice that the answer came back as a _set_ — `energy`, `torque` and `work` all share these dimensions, because dimensions cannot tell them apart. Nothing can: that is a fact about physics, not a limitation of `unxt`.

## Now catch one that is wrong

Suppose a colleague writes momentum where they meant energy — $mv$ instead of $mv^2$. Dimensionally:

```{code-block} python
>>> u.dimension("mass * length / time")
PhysicalType({'impulse', 'momentum'})

```

Momentum, not energy. The error is visible without running a single line of the simulation that would have used it.

And when a combination is not any known physical quantity, you get told:

```{code-block} python
>>> u.dimension("mass * length")
PhysicalType('unknown')

```

`unknown` is the answer you want to be alarmed by. It means the expression is dimensionally coherent but corresponds to nothing anybody has named — usually a sign that a term is missing.

## Derive something you do not know

Verification is the easy half. Let's use dimensions to _find_ a result.

How does the period of a pendulum depend on its length $L$ and gravity $g$? We do not know the formula. We do know the period is a time, and that only $L$ and $g$ are available. So ask what $L/g$ is:

```{code-block} python
>>> u.dimension("length / acceleration")
PhysicalType('unknown')

```

Not a named quantity — but look at its square root:

```{code-block} python
>>> u.dimension("(length / acceleration)**0.5")
PhysicalType('time')

```

A time. So the period must go as $\sqrt{L/g}$, and we have recovered the shape of the pendulum formula from nothing but dimensions:

```{code-block} python
>>> u.dimension("(length / acceleration)**0.5") == u.dimension("time")
True

```

Dimensional analysis cannot give you the $2\pi$ out front — no pure number has dimensions, so no dimensional argument can ever produce one. It gives you everything else.

## Point it at real objects

`dimension_of` answers the same question about things you already have. A unit:

```{code-block} python
>>> u.dimension_of(u.unit("kg m2 / s2"))
PhysicalType({'energy', 'torque', 'work'})

```

or a quantity:

```{code-block} python
>>> u.dimension_of(u.Q(1.0, "J"))
PhysicalType({'energy', 'torque', 'work'})

```

Same answer both times, and the same answer we derived from `mass * (length/time)**2` above — three routes to one dimension.

## Build the checker

Let's wrap this into something reusable: given an expression and what you expect it to be, say whether they agree.

```{code-block} python
>>> def check(expression, expected):
...     got = u.dimension(expression)
...     return got == u.dimension(expected)

```

Try it on the formulas from earlier:

```{code-block} python
>>> check("mass * (length/time)**2", "energy")
True

>>> check("mass * length / time", "energy")
False

>>> check("force / area", "pressure")
True

```

That last one we had not tested before, and it passed — force per unit area is a pressure:

```{code-block} python
>>> u.dimension("mass / (length * time**2)")
PhysicalType({'energy density', 'pressure', 'stress'})

```

## What we built

You have a `check` function that validates a physics expression against what you expect it to be, and you used the same tool to derive the pendulum period's form without knowing it in advance. None of it required a single measurement — which is the point: dimensional errors are findable before the code runs, and this is how you find them.

## Where to go next

- {doc}`../reference/dimensions` — every input `dimension` accepts, and the full expression syntax including multi-word names.
- {doc}`mars-lander` — what happens when a unit error _does_ reach the numbers.
- The `unxts.parametric` package's [dimension checking](../packages/unxts.parametric/index), for enforcing a dimension on a function argument at runtime.
