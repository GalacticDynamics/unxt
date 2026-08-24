# Write a function others can call

Everything so far has had you writing scripts. This lesson is about writing a _function_ — one other people call, with units you did not choose. We will start from a version that fails silently, and end with one that cannot be called wrong by accident.

The worked example is escape velocity, $v = \sqrt{2GM/r}$, but the decisions are the same for anything you expose.

You need `unxt` installed and nothing else.

## Set up

```{code-block} python
>>> import math

>>> import quaxed.numpy as jnp
>>> import unxt as u
>>> from astropy import constants as const

```

## Start with the version everyone writes first

Take plain numbers, and say in the docstring which units you meant:

```{code-block} python
>>> G_SI = const.G.value

>>> def escape_velocity_naive(mass_kg, radius_m):
...     """Escape velocity in m/s. mass_kg in kg, radius_m in metres."""
...     return math.sqrt(2 * G_SI * mass_kg / radius_m)

```

For Earth it is right — about 11.2 km/s:

```{code-block} python
>>> round(escape_velocity_naive(5.972e24, 6.371e6) / 1000, 2)
11.19

```

Now a caller who works in solar masses reads "mass" and passes one:

```{code-block} python
>>> round(escape_velocity_naive(1.0, 6.371e6), 6)
0.0

```

Zero. Not an error, not a warning — a number, of the right type, in the right range for _something_, and wrong by fifteen orders of magnitude. The docstring was the only thing standing between the caller and this, and docstrings are not enforced.

## Take quantities instead

Change the signature to accept quantities, and let the arithmetic carry the units. Note `G` now carries its own:

```{code-block} python
>>> G = u.Q(const.G.value, "m3 / (kg s2)")

>>> def escape_velocity(mass, radius):
...     return jnp.sqrt(2 * G * mass / radius)

```

The caller's units no longer have to match yours. Earth, in SI:

```{code-block} python
>>> escape_velocity(u.Q(5.972e24, "kg"), u.Q(6.371e6, "m")).uconvert("km/s")
Quantity(Array(11.185978, dtype=float32), unit='km / s')

```

and the Sun, in solar masses and kilometres — units your function never mentions:

```{code-block} python
>>> escape_velocity(u.Q(1.0, "solMass"), u.Q(696340.0, "km")).uconvert("km/s")
Quantity(Array(617.3908, dtype=float32), unit='km / s')

```

617 km/s, the textbook value. The caller's mistake from a moment ago is now impossible: a solar mass _is_ a mass, and the conversion happens inside the arithmetic.

## Find out what units still do not protect you from

So far so good. Now pass a **time** where the mass goes:

```{code-block} python
>>> wrong = escape_velocity(u.Q(1.0, "s"), u.Q(696340.0, "km"))
>>> wrong
Quantity(Array(1.3845454e-08, dtype=float32), unit='m(3/2) / (kg(1/2) km(1/2) s(1/2))')

```

No error. The unit algebra had no opinion — you can take the square root of anything — so it computed something and gave it an honest, unreadable unit.

Ask what that unit _is_ and you get the tell:

```{code-block} python
>>> u.dimension_of(wrong)
PhysicalType('unknown')

```

And the error, when it finally comes, comes somewhere else entirely — wherever the caller tries to use the result:

```{code-block} python
>>> try:
...     wrong.uconvert("km/s")
... except Exception as e:
...     print(type(e).__name__)
UnitConversionError

```

That is a bug report about the wrong line of code. Units caught it, but far from where it happened.

## Guard the boundary

So check the arguments where they arrive. `is_unit_convertible` asks the question without attempting the conversion, so you can raise your own error:

```{code-block} python
>>> def escape_velocity(mass, radius):
...     if not u.is_unit_convertible("kg", mass):
...         msg = f"mass must have dimensions of mass, got {u.unit_of(mass)}"
...         raise ValueError(msg)
...     if not u.is_unit_convertible("m", radius):
...         msg = f"radius must have dimensions of length, got {u.unit_of(radius)}"
...         raise ValueError(msg)
...     return jnp.sqrt(2 * G * mass / radius)

```

Now the same mistake is caught at the call, and the message names your parameter:

```{code-block} python
>>> try:
...     escape_velocity(u.Q(1.0, "s"), u.Q(696340.0, "km"))
... except ValueError as e:
...     print(e)
mass must have dimensions of mass, got s

```

and so does the other one:

```{code-block} python
>>> try:
...     escape_velocity(u.Q(1.0, "solMass"), u.Q(1.0, "kg"))
... except ValueError as e:
...     print(e)
radius must have dimensions of length, got kg

```

The correct call is unaffected:

```{code-block} python
>>> escape_velocity(u.Q(1.0, "solMass"), u.Q(696340.0, "km")).uconvert("km/s")
Quantity(Array(617.3908, dtype=float32), unit='km / s')

```

## Decide what a bare number means

Some callers will not use `unxt` at all and will hand you a float. You can accept that, provided you are explicit about what you assume it is. `AllowValue` lets `ustrip` pass a bare value through untouched, taken to be already in the unit you named:

```{code-block} python
>>> def escape_velocity_lenient(mass, radius):
...     m = u.Q(u.ustrip(u.quantity.AllowValue, "kg", mass), "kg")
...     r = u.Q(u.ustrip(u.quantity.AllowValue, "m", radius), "m")
...     return jnp.sqrt(2 * G * m / r)

```

A quantity works as before, in any units:

```{code-block} python
>>> escape_velocity_lenient(u.Q(1.0, "solMass"), u.Q(696340.0, "km")).uconvert("km/s")
Quantity(Array(617.39087, dtype=float32), unit='km / s')

```

and a bare pair of SI numbers now works too:

```{code-block} python
>>> escape_velocity_lenient(1.989e30, 6.9634e8).uconvert("km/s")
Quantity(Array(617.4825, dtype=float32), unit='km / s')

```

You have traded a guarantee for convenience, and the trade is explicit and documented in one place instead of implied by a docstring.

## Return what the algebra gives you

Notice every example above ended with the _caller_ writing `.uconvert("km/s")`. That is deliberate. Your function returns a quantity whose unit falls out of the arithmetic, and the caller asks for the unit they want. Converting on their behalf means guessing, and a guess in a return value is the same class of mistake the naive version made in its arguments.

## What we built

A function that accepts any units for its arguments, rejects arguments of the wrong dimension at the point of call with a message naming the parameter, has a documented answer for callers who pass bare numbers, and hands back a quantity rather than a number that means something only if you read the docstring.

## Where to go next

- {doc}`../how-to/compare-quantities` — `is_unit_convertible`, `equivalent` and `==`.
- {doc}`../how-to/check-types-at-runtime` — enforcing dtype and shape as well as dimension.
- [`unxts.parametric`](../packages/unxts.parametric/index) — putting the dimension in the _type_, so the check is part of the signature rather than the body.
- {doc}`../reference/quantity` — `AllowValue` and the rest of the surface.
