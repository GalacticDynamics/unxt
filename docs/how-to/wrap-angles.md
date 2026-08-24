# How to wrap angles to a range

Angles have branch cuts: a right ascension of 370° is the same direction as 10°, and a longitude of −200° is the same as 160°. `wrap_to` maps an angle into the half-open range you name, so downstream comparisons and plots behave.

```{code-block} python
>>> import unxt as u

```

## Wrap to 0–360°

Pass the lower and upper bounds as quantities:

```{code-block} python
>>> u.Angle(370.0, "deg").wrap_to(u.Q(0, "deg"), u.Q(360, "deg"))
Angle(Array(10., dtype=float32...), unit='deg')

```

The range is half-open — the lower bound is included, the upper is not, so 360° comes back as 0°.

## Wrap to −180…180°

The usual convention for longitudes and for anything centred on zero:

```{code-block} python
>>> u.Angle(200.0, "deg").wrap_to(u.Q(-180, "deg"), u.Q(180, "deg"))
Angle(Array(-160., dtype=float32...), unit='deg')

>>> u.Angle(-200.0, "deg").wrap_to(u.Q(-180, "deg"), u.Q(180, "deg"))
Angle(Array(160., dtype=float32...), unit='deg')

```

## Wrap an array

Whole arrays wrap element-wise, which is the common case for a catalogue column:

```{code-block} python
>>> import jax.numpy as jnp

>>> u.Angle(jnp.asarray([-10.0, 370.0, 720.0]), "deg").wrap_to(
...     u.Q(0, "deg"), u.Q(360, "deg")
... )
Angle(Array([350.,  10.,   0.], dtype=float32), unit='deg')

```

## Give the bounds in whatever units you like

The bounds do not have to match the angle's unit — they are converted for you, and the result keeps the _input's_ unit:

```{code-block} python
>>> import math

>>> u.Angle(370.0, "deg").wrap_to(u.Q(0.0, "rad"), u.Q(2 * math.pi, "rad"))
Angle(Array(10., dtype=float32...), unit='deg')

```

To get the result in radians instead, convert after wrapping:

```{code-block} python
>>> u.Angle(7.0, "rad").wrap_to(u.Q(0.0, "rad"), u.Q(2 * math.pi, "rad"))
Angle(Array(0.7168145, dtype=float32...), unit='rad')

```

## Use the function form

If you prefer the functional style, or are working with a plain `Quantity` rather than an `Angle`, `unxt.quantity.wrap_to` takes the object first:

```{code-block} python
>>> u.quantity.wrap_to(u.Q(370.0, "deg"), u.Q(0, "deg"), u.Q(360, "deg"))
Quantity(Array(10., dtype=float32...), unit='deg')

```

Note the result is a `Quantity`, not an `Angle` — `wrap_to` preserves the class it was given. If you want the angular-dimension guarantee, construct an {class}`~unxt.quantity.Angle` first; it rejects non-angular units at construction.

## See also

- {doc}`../reference/quantity` — the `Angle` class and its enforced dimensionality.
- {doc}`../explanation/sharp-bits` — why `jnp.deg2rad` rescales the value but not the unit label, and what to use instead.
