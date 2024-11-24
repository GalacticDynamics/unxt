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

First let's create a some dimensions:

```{code-block} python
>>> from unxt.dims import dimension

>>> dim = dimension('length')  # from a str
>>> dim
PhysicalType('length')

>>> dimension(dim)  # from a dimension object
PhysicalType('length')

```

Now let's get the dimensions from objects:

```{code-block} python
>>> from unxt.dims import dimension_of
>>> from unxt import Quantity

>>> print(dimension_of("length"))  # str have no dimensions
None

>>> dimension_of(dim)  # from a dimension object
PhysicalType('length')

>>> q = Quantity(5, 'm')  # from a Quantity
>>> dimension_of(q)
PhysicalType('length')

```

:::{seealso}

[API Documentation for Dimensions](../api/dims.md)

:::
