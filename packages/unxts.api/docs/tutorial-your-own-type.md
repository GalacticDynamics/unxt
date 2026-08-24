# Teach unxt about your own type

Take a class that `unxt` has never heard of. By the end of this page, `unxt`'s own functions will work on it — `unit_of`, `dimension_of`, `ustrip`, `uconvert` — with nothing subclassed and not a line of `unxt` touched. A function written against `unxt`'s API will accept our class as readily as it accepts a `Quantity`.

You need `unxt` installed and nothing else. (`unxts.api` comes with it.)

## Set up

```{code-block} python
>>> from plum import dispatch
>>> import unxt as u

```

## Start with a type unxt does not know

Here is a small class standing in for whatever your project already has — a raw instrument reading, a number with a unit name attached:

```{code-block} python
>>> class Reading:
...     """A raw instrument reading: a number and a unit name."""
...     def __init__(self, value, unit):
...         self.value = value
...         self.unit = unit
...     def __repr__(self):
...         return f"Reading({self.value}, {self.unit!r})"

>>> r = Reading(2000.0, "m")

```

Ask `unxt` about it:

```{code-block} python
>>> print(u.unit_of(r))
None

```

`None` — not an error. `unit_of` has a fallback for objects it does not recognise, and as far as it is concerned our `Reading` is just some object. The `"m"` sitting right there in the attribute means nothing to it yet.

## Register the first implementation

Now we tell `unxt` how to find the unit. Define a function named `unit_of`, annotate the argument with our type, and decorate it with `plum`'s `dispatch`:

```{code-block} python
>>> @dispatch
... def unit_of(obj: Reading, /):
...     return u.unit(obj.unit)

```

That is the entire registration. Ask again:

```{code-block} python
>>> u.unit_of(r)
Unit("m")

```

Look carefully at what just happened. We called `u.unit_of` — `unxt`'s function, the one we did not modify — and it dispatched into the implementation we wrote a moment ago. A bare `@dispatch` on a function of the same name extends the existing dispatch function rather than shadowing it, so `unit_of` and `u.unit_of` are now literally the same object with one more method on it.

## Add the dimension

Same pattern. Once `unxt` can get the unit, the dimension follows from it:

```{code-block} python
>>> @dispatch
... def dimension_of(obj: Reading, /):
...     return u.dimension_of(u.unit_of(obj))

>>> u.dimension_of(r)
PhysicalType('length')

```

Notice our implementation called `u.unit_of(obj)` — the function we had just extended. Registrations compose; each one you add makes the next one shorter.

## Watch generic code accept it

Here is the payoff. Write a function that knows nothing about `Reading` and only uses `unxt`'s API:

```{code-block} python
>>> def describe(x):
...     return f"{u.dimension_of(x)} measured in {u.unit_of(x)}"

```

Hand it a `Quantity`:

```{code-block} python
>>> describe(u.Q(5.0, "m"))
'length measured in m'

```

and hand it our `Reading`:

```{code-block} python
>>> describe(r)
'length measured in m'

```

The same function, unchanged, for both. `describe` could have been written by someone else, in another library, before our class existed — this is what the abstract API is for.

## Make it convert

Two more registrations and `Reading` supports unit conversion. `ustrip` returns the bare number in the requested unit:

```{code-block} python
>>> @dispatch
... def ustrip(to: u.AbstractUnit, obj: Reading, /):
...     return u.ustrip(to, u.Q(obj.value, obj.unit))

>>> u.ustrip(u.unit("km"), r)
Array(2., dtype=float32, weak_type=True)

```

and `uconvert` returns a converted object of our own type:

```{code-block} python
>>> @dispatch
... def uconvert(to: u.AbstractUnit, obj: Reading, /):
...     return Reading(float(u.ustrip(to, obj)), str(to))

>>> u.uconvert(u.unit("km"), r)
Reading(2.0, 'km')

```

2000 metres came back as 2.0 kilometres, still a `Reading`. We borrowed `unxt`'s conversion machinery for the arithmetic and kept our own type on the outside.

## What we built

`Reading` now answers to four of `unxt`'s functions, and generic unit-aware code works on it. We never subclassed an `unxt` class, never edited `unxt`, and never imported `jax` — which is the point of `unxts.api` being a separate package: a library can speak this API while leaving the choice of whether to install `unxt` itself to its users.

## Where to go next

- [How to extend unxt with your own types](extending) — the rest of the patterns: conversion in both directions, fallbacks, and debugging dispatch.
- [API](api) — every abstract function and what it promises.
- [Why an abstract dispatch API](why-abstract-dispatch) — and what the approach costs.
