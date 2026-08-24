# Units

Units are measures of dimensions: metres for length, seconds for time, joules for energy. `unxt.units` provides two functions, both re-exported on the top-level `unxt` namespace.

| Function       | Purpose                                                 |
| -------------- | ------------------------------------------------------- |
| `unxt.unit`    | Construct a unit.                                       |
| `unxt.unit_of` | Return the unit of an object, or `None` if it has none. |

Both are multiple-dispatch functions; the registered signatures are listed in the {doc}`API reference <api/units>`.

```{code-block} python
>>> import unxt as u

>>> u.unit
<multiple-dispatch function unit (...)>

>>> u.unit_of
<multiple-dispatch function unit_of (...)>
```

## `unit`

Accepts a string or an existing unit object; the unit backend is [`astropy.units`](https://docs.astropy.org/en/stable/units/index.html), so the returned object is an `astropy` `Unit`.

```{code-block} python
>>> m = u.unit('m')  # from a str
>>> m
Unit("m")

>>> u.unit(m)  # from a unit object
Unit("m")

```

Units compose arithmetically:

```{code-block} python
>>> u.unit("km") / u.unit("h")
Unit("km / h")

```

## `unit_of`

Returns the unit of an object. Objects that carry no unit — strings, for example — return `None`.

```{code-block} python
>>> print(u.unit_of("m"))  # str have no units
None

>>> u.unit_of(m)  # from a unit object
Unit("m")

>>> q = u.Q(5, 'm')  # from a Quantity
>>> u.unit_of(q)
Unit("m")

```

## See also

- {doc}`unitsystems` — collections of base units used together.
- {doc}`dimensions` — the physical nature a unit measures.
- {doc}`API documentation for units <api/units>`
