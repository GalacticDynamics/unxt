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

The `dimension()` function accepts two types of inputs:

### 1. From Dimension Objects

If you already have a dimension object (from Astropy), you can pass it directly.
The function will return the same object unchanged.

```{code-block} python
>>> from unxt.dims import dimension
>>> import astropy.units as apyu

>>> dim = apyu.get_physical_type("length")
>>> dimension(dim) is dim
True

```

### 2. From Strings

You can create dimensions from strings in several ways:

#### Simple Dimension Names

The simplest approach is to use a dimension name as a string:

```{code-block} python
>>> from unxt.dims import dimension

>>> dimension("length")
PhysicalType('length')

>>> dimension("time")
PhysicalType('time')

>>> dimension("mass")
PhysicalType('mass')

```

#### Multi-word Dimension Names

Astropy supports dimension names with spaces, such as "amount of substance" and
"absement". You can use these names directly:

```{code-block} python
>>> dimension("amount of substance")
PhysicalType('amount of substance')

>>> dimension("absement")
PhysicalType('absement')

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
>>> dimension("length / time")
PhysicalType({'speed', 'velocity'})

>>> # Exponentiation: length**2 gives area
>>> dimension("length**2")
PhysicalType('area')

>>> # Complex expression: force = mass * length / time**2
>>> dimension("mass * length / time**2")
PhysicalType('force')

```

#### Using Parentheses with Expressions

Parentheses can be used to group operations or to disambiguate multi-word
dimension names in expressions. When you have multi-word dimension names, you
**must** use parentheses:

```{code-block} python
>>> # Multi-word names require parentheses in expressions
>>> dimension("(angular speed) / (angular acceleration)")
PhysicalType('time')

>>> # Single-word names don't require parentheses
>>> dimension("length / time")
PhysicalType({'speed', 'velocity'})

>>> # But can use them optionally for clarity
>>> dimension("(length) / (time)")
PhysicalType({'speed', 'velocity'})

```

You can freely mix parenthesized multi-word names with unparenthesized
single-word names in the same expression:

```{code-block} python
>>> # Multi-word name with single-word names
>>> dimension("length * (amount of substance)")
PhysicalType('unknown')

>>> dimension("(absement) / time")
PhysicalType('length')

```

#### Whitespace Handling

Whitespace is flexible and doesn't affect parsing:

```{code-block} python
>>> # All of these are equivalent
>>> dimension("length / time")
PhysicalType({'speed', 'velocity'})

>>> dimension("length/time")
PhysicalType({'speed', 'velocity'})

>>> dimension("length / time ** 2")
PhysicalType('acceleration')

>>> dimension("length/time**2")
PhysicalType('acceleration')

```

## Getting Dimensions from Objects

Now let's get the dimension from various objects:

```{code-block} python
>>> from unxt.dims import dimension_of

>>> print(dimension_of("length"))  # strings have no dimensions
None

>>> dim = dimension("length")
>>> dimension_of(dim)  # dimensions return themselves
PhysicalType('length')

>>> q = u.Quantity(5, 'm')  # quantities have dimensions
>>> dimension_of(q)
PhysicalType('length')

```

## Important Notes

- **Unsupported operators**: The `+` and `-` symbols are **not** supported as
  mathematical operators (they're reserved for dimension names like
  "electric-dipole moment"). If you need to add or subtract dimensions, that
  doesn't make physical sense anyway!

- **Derived dimensions**: When you create a derived dimension via expression,
  Astropy will attempt to simplify it to a known dimension name if possible.
  For example, "length / time" becomes "speed/velocity".

- **Dimension names from Astropy**: All dimension names are drawn from Astropy's
  physical type catalogue, which provides a comprehensive set of physical
  dimensions used in science and engineering.

```{code-block} python
>>> # Examples of derived dimensions being simplified
>>> dimension("length**3")
PhysicalType('volume')

>>> dimension("mass / length**3")
PhysicalType('mass density')
```


:::{seealso}

[API Documentation for Dimensions](../api/dims.md)

:::
