# Dimensions

A dimension refers to a measurable extent of a physical quantity, such as
length, time, or mass. `unxt` has a sub-module for working with dimensions:
`unxt.dims`. The dimensions module provides two functions: `dimension` and
`dimension_of`.

```{code-block} python
>>> from unxt.dims import dimension, dimension_of
```

The function `dimension` is for creating a dimension, while `dimension_of` is
for getting the dimension of an object.

Note that both functions are also available in the `unxt` namespace.

```{code-block} python
>>> import unxt as u
>>> u.dimension is dimension
True

>>> u.dimension_of is dimension_of
True
```

## Creating Dimensions

The `dimension()` function can accept many types of inputs, and more
types can be registered via multiple dispatch.
A longer list of supported inputs can be found in the API documentation.
The full list can be found dynamically by running:

```{code-block} python
>>> u.dimension.methods  # doctest: +SKIP
List of 2 method(s):
    [0] dimension(obj: ...
    [1] dimension(obj: ...
```

### 1. From Dimension Objects

If you already have a dimension object (from {mod}`astropy`), you can pass it
directly. The function will return the same object unchanged.

```{code-block} python
>>> import astropy.units as apyu

>>> dim = apyu.get_physical_type("length")
>>> u.dimension(dim) is dim
True

```

### 2. From Strings

You can create dimensions from strings in several ways:

#### Simple Dimension Names

The simplest approach is to use a dimension name as a string:

```{code-block} python
>>> u.dimension("length")
PhysicalType('length')

>>> u.dimension("time")
PhysicalType('time')

>>> u.dimension("mass")
PhysicalType('mass')

```

Some dimension names have spaces, such as "amount of substance" and
"absement". You can use these names directly:

```{code-block} python
>>> u.dimension("amount of substance")
PhysicalType('amount of substance')
```

#### Mathematical Expressions

You can construct derived dimensions using mathematical expressions. Supported
operators are:

- `*` : Multiplication
- `/` : Division
- `**` : Exponentiation (power)

Expressions follow standard operator precedence (PEMDAS):

```{code-block} python
>>> # Division: length / time gives speed
>>> u.dimension("length / time")
PhysicalType({'speed', 'velocity'})

>>> # Exponentiation: length**2 gives area
>>> u.dimension("length**2")
PhysicalType('area')

>>> # Complex expression: force = mass * length / time**2
>>> u.dimension("mass * length / time**2")
PhysicalType('force')

```

Parentheses can be used to group operations or to disambiguate multi-word
dimension names in expressions. When you have multi-word dimension names, you
**must** use parentheses:

```{code-block} python
>>> # Multi-word names require parentheses in expressions
>>> u.dimension("(angular speed) / (angular acceleration)")
PhysicalType('time')

>>> # Single-word names don't require parentheses
>>> u.dimension("length / time")
PhysicalType({'speed', 'velocity'})

>>> # But can use them optionally for clarity
>>> u.dimension("(length) / (time)")
PhysicalType({'speed', 'velocity'})

```

You can freely mix parenthesized multi-word names with unparenthesized
single-word names in the same expression:

```{code-block} python
>>> # Multi-word name with single-word names
>>> u.dimension("length * (amount of substance)")
PhysicalType('unknown')

>>> u.dimension("(absement) / time")
PhysicalType('length')

```

#### Whitespace Handling

Whitespace is flexible and doesn't affect parsing:

```{code-block} python
>>> u.dimension("length / time ** 2")
PhysicalType('acceleration')

>>> u.dimension("length/time**2")
PhysicalType('acceleration')

```

## Getting Dimensions from Objects

Now let's get the dimension from various objects:

```{code-block} python
>>> print(u.dimension_of("length"))  # strings have no dimensions
None

>>> dim = u.dimension("length")
>>> u.dimension_of(dim)  # dimensions return themselves
PhysicalType('length')

>>> unit = u.unit('m')  # units have dimensions
>>> u.dimension_of(unit)
PhysicalType('length')

>>> q = u.Q(5, 'm')  # quantities have dimensions
>>> u.dimension_of(q)
PhysicalType('length')

>>> u.dimension_of(u.Q["length"])  # so do parameterized Quantity classes
PhysicalType('length')

>>> try: u.dimension_of(u.Q)  # unparameterized Quantity will raise an error
... except Exception as e: print(e)
can only get dimensions from parametrized Quantity -- Quantity[dim].

>>> angle = u.Angle(30, 'deg')  # angles always have dimension 'angle'
>>> u.dimension_of(angle)
PhysicalType('angle')

>>> u.dimension_of(u.Angle)  # and Angle class itself is dimensionful
PhysicalType('angle')

```

## Important Notes

- **Unsupported operators**: The `+` and `-` symbols are **not** supported as
  mathematical operators since dimensions are invariant under addition and subtraction.
  Also, they're reserved for dimension names like "electric-dipole moment".
  If you need to add or subtract dimensions, what are you even doing?

- **Derived dimensions**: When you create a derived dimension via an expression,
  Astropy will attempt to simplify it to a known dimension name if possible.
  For example, "length / time" becomes "speed/velocity".

- **Dimension names from Astropy**: All dimension names are drawn from Astropy's
  physical type catalogue, which provides a comprehensive set of physical
  dimensions used in science and engineering.

  ```{code-block} python
  >>> u.dimension("length**3")
  PhysicalType('volume')

  >>> u.dimension("mass / length**3")
  PhysicalType('mass density')
  ```


:::{seealso}

[API Documentation for Dimensions](../api/dims.md)

:::
