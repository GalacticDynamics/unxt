# 📜 Conventions

## Naming Conventions

`unxt` uses a few conventions to make the code more readable and to avoid
verbosity. Many of these are also found in the [Glossary](glossary.md).

- `Abstract...`: a class that is not meant to be instantiated directly, but
  rather to be subclassed. Abstract classes are prefixed with 'Abstract'.
  Concrete (or 'final') classes are not so prefixed. As a further rule, no
  abstract class inherits from a concrete class and no concrete class inherits
  from any other concrete class.
- `USys`: a shorthand for "unit system", used in class names for concision.
- `Sim`: a shorthand for "simulation", used in class names for concision.

## Functional vs Object-Oriented APIs

As `JAX` is function-oriented, but Python is generally object-oriented, `unxt`
provides both functional and object-oriented APIs. The functional APIs are the
primary APIs, but the object-oriented APIs are easy to use and call the
functional APIs, so lose none of the power.

As an example, consider the following code snippets:

```{code-block} python

>>> import unxt as u

>>> q = u.Q(1, 'm')
>>> q
Quantity(Array(1, dtype=int32, weak_type=True), unit='m')
```

First we'll show the object-oriented API:

```{code-block} python

>>> q.uconvert('cm')
Quantity(Array(100., dtype=float32, weak_type=True), unit='cm')
```

And now the function-oriented API:

```{code-block} python

>>> u.uconvert("cm", q)
Quantity(Array(100., dtype=float32, weak_type=True), unit='cm')
```

## Argument Order of Functional APIs

The functional APIs in `unxt` are inspired by the `Unitful.jl` library. The way
to remember the order of arguments is to think of the function as constructing
an operator that is then applied to the quantity.

For example, to convert a quantity `q` to centimeters, we use the `uconvert`
function with the unit as the first argument and the quantity as the second:

```{code-block} python

>>> u.uconvert("cm", q)  # convert[to_unit](quantity)
Quantity(Array(100., dtype=float32, weak_type=True), unit='cm')
```

One of the reasons for this order is because it works very well with a
multiple-dispatch system, where many variants of the same function can be
defined based on the types of the arguments. The arguments for "operator" part
of the function are the first arguments, and the arguments for the "operand" are
the last arguments.

## Multiple Dispatch

`unxt` uses [multiple dispatch](https://beartype.github.io/plum/) to hook into
`quax`'s flexible and extensible system to enable custom array-ish objects, like
`Quantity`, in `JAX`. Also, `unxt` uses multiple dispatch to enable deep
interoperability between `unxt` and other libraries, like `astropy`, `gala` (,
and anything user-defined).

For example, `unxt` provides a `Quantity.from_` method that can convert an
`astropy.Quantity` to a `unxt.Quantity`:

```{code-block} python

>>> import astropy.units as apyu
>>> import unxt as u

>>> aq = apyu.Quantity(1, 'm')  # Astropy Quantity
>>> aq
<Quantity 1. m>

>>> xq = u.Quantity.from_(aq)  # unxt Quantity
>>> xq
Quantity(Array(1., dtype=float32), unit='m')

```

This easy interoperability is enabled by multiple dispatch, which allows the
`Quantity.from_` method to dispatch to the correct implementation based on the
types of the arguments.

For more information on multiple dispatch, see the
[plum documentation](https://beartype.github.io/plum/).
