# unxt-api

Abstract dispatch API for [unxt](https://github.com/GalacticDynamics/unxt).

`unxt-api` defines the abstract dispatch interfaces that `unxt` and other
packages implement. It provides a minimal dependency foundation for packages
that want to define or use `unxt`'s multiple-dispatch-based API without pulling
in the full `unxt` implementation.

The `unxt-api` package serves several important purposes:

1. **Minimal Dependencies**: Depends only on `plum-dispatch`, not on JAX, NumPy,
   or Astropy
2. **Extensibility**: Allows third-party packages to register their own
   implementations
3. **Type Safety**: Provides a clear contract for what functions exist and what
   they should do
4. **Separation of Concerns**: API definitions are separate from implementation
   details

## Installation

```bash
pip install unxt-api
```

Or with uv:

```bash
uv add unxt-api
```

## Core API

The `unxt-api` package defines abstract dispatch functions organized by domain:

- **Dimensions** (`dimension`, `dimension_of`) - Working with physical
  dimensions
- **Units** (`unit`, `unit_of`) - Constructing and inspecting units
- **Quantities** (`uconvert`, `ustrip`, `is_unit_convertible`, `wrap_to`) - Unit
  conversion and quantity operations
- **Unit Systems** (`unitsystem_of`) - Inspecting unit systems

## Using Multiple Dispatch

All functions in `unxt-api` use [plum](https://beartype.github.io/plum/) for
multiple dispatch. This means:

1. **Functions can have multiple implementations** based on argument types
2. **You can register your own implementations** for custom types
3. **Type annotations drive dispatch** - the runtime types of arguments
   determine which implementation runs

### Viewing All Implementations

To see all registered implementations of a function:

```python
import unxt as u

# View all dimension() implementations
u.dimension.methods

# View all uconvert() implementations
u.uconvert.methods

# View all unit_of() implementations
u.unit_of.methods
```

### Registering Custom Implementations

You can extend `unxt-api`'s dispatch system with your own types:

```python
from plum import dispatch
import unxt_api as u


class MyCustomQuantity:
    def __init__(self, value, unit_str):
        self.value = value
        self.unit_str = unit_str


# Register implementation for your type
@dispatch
def unit_of(obj: MyCustomQuantity, /):
    """Get unit from MyCustomQuantity."""
    return u.unit(obj.unit_str)


# Now it works with the dispatch system
my_q = MyCustomQuantity(5.0, "m")
u.unit_of(my_q)  # Unit("m")
```

## Integration with `unxt`

The `unxt` package provides the concrete implementations of all `unxt-api`
functions. When you use:

```python
import unxt as u

q = u.Q(5, "m")
u.uconvert("km", q)
```

The `u.uconvert` function is the implementation registered by `unxt` for the
abstract `unxt_api.uconvert` function.

## Documentation

For comprehensive documentation, examples, and extension guides, see:

- [unxt-api API Reference](https://unxt.readthedocs.io/en/latest/api/unxt-api.html)
- [Extending unxt Guide](https://unxt.readthedocs.io/en/latest/guides/extending.html)
- [Main unxt Documentation](https://unxt.readthedocs.io/)
- [plum Documentation](https://beartype.github.io/plum/) - Multiple dispatch
  library

## Contributing

Contributions are welcome! Please see the main
[unxt repository](https://github.com/GalacticDynamics/unxt) for contributing
guidelines.
