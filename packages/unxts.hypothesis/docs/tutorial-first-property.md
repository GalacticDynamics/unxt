# Write your first property test

In this tutorial we will test a unitful function without writing a single example by hand. Instead of picking a few inputs and asserting the answers, we will state a _property_ that should hold for every input, and let [Hypothesis](https://hypothesis.readthedocs.io/) go looking for one where it does not.

You need `unxt` and `unxts.hypothesis` installed and nothing else.

## Set up

```{code-block} python
>>> import jax.numpy as jnp
>>> import unxt as u
>>> import unxts.hypothesis as ust
>>> from hypothesis import given, settings, strategies as st

```

We import `settings` too, and every test below carries `@settings(deadline=None, max_examples=25)`. Hypothesis times each example and fails anything over 200 ms; a test that calls into JAX will sometimes trip a fresh trace or compile and blow past that, through no fault of the code under test. Turning the deadline off is the standard thing to do for JAX tests.

## State a property

Here is one that should be true of any length: convert it to kilometres, convert it back, and you should have what you started with.

```{code-block} python
>>> @settings(deadline=None, max_examples=25)
... @given(q=ust.quantities(unit="m"))
... def test_roundtrip(q):
...     back = q.uconvert("km").uconvert("m")
...     assert bool(jnp.allclose(back.ustrip("m"), q.ustrip("m"),
...                              rtol=1e-4, atol=1e-6))

```

`ust.quantities(unit="m")` is a _strategy_: a recipe for generating quantities, not a quantity. `@given` hands one to the test at a time.

Now run it. A property test is an ordinary function — just call it:

```{code-block} python
>>> test_roundtrip()

```

Nothing printed, which means every generated case passed. That silence covered rather more than one input.

## See how much it covered

Let's watch what actually goes in. We will collect the generated values instead of asserting anything:

```{code-block} python
>>> seen = []

>>> @settings(deadline=None, max_examples=25)
... @given(q=ust.quantities(unit="m", shape=(3,)))
... def collect(q):
...     seen.append(q)

>>> collect()
>>> len(seen)
25

```

Twenty-five runs of one test body, each with a different array — the number we asked for with `max_examples`. Every one had the unit and shape we specified:

```{code-block} python
>>> seen[0].unit
Unit("m")

>>> seen[0].shape
(3,)

```

This is the trade you are making. You gave up choosing the inputs, and in exchange you got far more of them than you would have written out — including zeros, tiny values, and huge ones, which is where unit code tends to go wrong.

## Let the units vary too

`unit=` accepts a strategy, not just a fixed unit. Ask for any unit of length and the generated quantities will span several:

```{code-block} python
>>> units = set()

>>> @settings(deadline=None, max_examples=25)
... @given(q=ust.quantities(unit=ust.units("length")))
... def collect_units(q):
...     units.add(str(q.unit))

>>> collect_units()
>>> len(units) > 1
True

```

Push it further and generate quantities of _any_ physical dimension at all, then assert something that must hold regardless — adding a quantity to itself cannot change its unit:

```{code-block} python
>>> @settings(deadline=None, max_examples=25)
... @given(q=ust.quantities(unit=ust.units(ust.named_dimensions())))
... def test_add_keeps_unit(q):
...     assert (q + q).unit == q.unit

>>> test_add_keeps_unit()

```

Silence again — and that ran across metres, seconds, joules, teslas and whatever else the dimension catalogue offered.

## Test a function of your own

Now let's point this at real code. Here is a function that computes a speed:

```{code-block} python
>>> def speed(d, t):
...     return d / t

```

We could assert what it returns for one distance and one time. The stronger claim is about its _dimension_: give it a length and a time, and the result must be a speed — whatever units the caller happened to use.

```{code-block} python
>>> @settings(deadline=None, max_examples=25)
... @given(d=ust.quantities(unit="km"), t=ust.quantities(unit="s"))
... def test_speed_dimension(d, t):
...     assert u.dimension_of(speed(d, t)) == u.dimension("speed")

>>> test_speed_dimension()

```

Passed. And notice we never converted anything by hand — `unxt` did the unit algebra, and the property checked that it came out right:

```{code-block} python
>>> speed(u.Q(3.0, "km"), u.Q(2.0, "s"))
Quantity(Array(1.5, dtype=float32, weak_type=True), unit='km / s')

```

## Generate angles that obey a range

One more strategy worth meeting, because angles are where wrapping bugs live. `ust.angles` can generate angles already wrapped into a range, so you can assert that whatever consumes them stays in that range:

```{code-block} python
>>> @settings(deadline=None, max_examples=25)
... @given(a=ust.angles(wrap_to=(u.Q(0, "deg"), u.Q(360, "deg"))))
... def test_angle_range(a):
...     deg = a.ustrip("deg")
...     assert bool(jnp.all(deg >= 0)) and bool(jnp.all(deg < 360))

>>> test_angle_range()

```

## What we built

You have four property tests covering unit round-trips, unit-preserving arithmetic across every physical dimension, the dimensional correctness of a function you wrote, and angle ranges — and you chose not one input value. Drop these in a `test_*.py` and `pytest` will run them like any other test.

## Where to go next

- [How to write property-based tests](testing-guide) — assumptions, shrinking, `st.from_type()`, and reproducing a failure once you get one.
- [Strategies](strategies) — every strategy and what its parameters do.
- [How to combine strategies](recipes) — composing them for a specific domain.
