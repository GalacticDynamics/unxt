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

First let's create a dimension:

```{code-block} python
>>> from unxt.dims import dimension  # also in `unxt` namespace

>>> dim = dimension('length')  # from a str
>>> dim
PhysicalType('length')

>>> dimension(dim)  # from a dimension object
PhysicalType('length')

```

Now let's get the dimension from various objects:

```{code-block} python
>>> from unxt.dims import dimension_of  # also in `unxt` namespace

>>> print(dimension_of("length"))  # str have no dimensions
None

>>> dimension_of(dim)  # from a dimension object
PhysicalType('length')

>>> q = u.Q(5, 'm')  # from a Quantity
>>> dimension_of(q)
PhysicalType('length')

```

:::{seealso}

[API Documentation for Dimensions](../api/dims.md)

:::
