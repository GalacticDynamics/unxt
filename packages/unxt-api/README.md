# unxt-api

Abstract dispatch API definitions for unxt.

This package defines the abstract dispatch interfaces that unxt and other
packages can implement. It provides a minimal dependency set for packages that
want to define or use unxt's dispatch-based API without pulling in the full unxt
implementation.

## Installation

```bash
pip install unxt-api
```

## Usage

This package is typically used as a dependency by unxt and related packages. It
defines abstract dispatch signatures using plum-dispatch that concrete
implementations register against.

For comprehensive documentation, examples, and extension guides, see:

- [unxt-api API Reference](https://unxt.readthedocs.io/en/latest/api/unxt-api.html)
- [Extending unxt Guide](https://unxt.readthedocs.io/en/latest/guides/extending.html)
- [Main unxt Documentation](https://unxt.readthedocs.io/)

## Quick Example

```python
from plum import dispatch
import unxt as u


class Temperature:
    def __init__(self, value, unit="K"):
        self.value = value
        self.unit_str = unit


@dispatch
def unit_of(obj: Temperature, /):
    return u.unit(obj.unit_str)


@dispatch
def dimension_of(obj: Temperature, /):
    return u.dimension("temperature")


# Now Temperature works with unxt!
temp = Temperature(300, "K")
u.unit_of(temp)  # Unit("K")
u.dimension_of(temp)  # PhysicalType('temperature')
```
