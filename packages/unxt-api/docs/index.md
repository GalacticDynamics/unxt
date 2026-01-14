# `unxt-api`

```{toctree}
:maxdepth: 1
:hidden:

api
extending
```

Abstract dispatch API for [unxt](https://github.com/GalacticDynamics/unxt).

{mod}`unxt-api` defines the abstract dispatch interfaces that {mod}`unxt` and
other packages implement. It provides a minimal dependency foundation for
packages that want to define or use {mod}`unxt`'s multiple-dispatch-based API
without pulling in the full {mod}`unxt` implementation.

The {mod}`unxt-api` package serves several important purposes:

1. **Minimal Dependencies**: Depends only on {mod}`plum-dispatch`, not on
   {mod}`jax`, {mod}`numpy`, or {mod}`astropy`
2. **Extensibility**: Allows third-party packages to register their own
   implementations
3. **Type Safety**: Provides a clear contract for what functions exist and what
   they should do
4. **Separation of Concerns**: API definitions are separate from implementation
   details

## Installation

::::{tab-set}

:::{tab-item} pip

```bash
pip install unxt-api
```

:::

:::{tab-item} uv

```bash
uv add unxt-api
```

:::

::::

## Core API

The {mod}`unxt-api` package defines abstract dispatch functions organized by
domain:

- **Dimensions** ({func}`~unxt_api.dimension`, {func}`~unxt_api.dimension_of`) -
  Working with physical dimensions
- **Units** ({func}`~unxt_api.unit`, {func}`~unxt_api.unit_of`) - Constructing
  and inspecting units
- **Quantities** ({func}`~unxt_api.uconvert`, {func}`~unxt_api.uconvert_value`,
  {func}`~unxt_api.ustrip`, {func}`~unxt_api.is_unit_convertible`,
  {func}`~unxt_api.wrap_to`) - Unit conversion and quantity operations
- **Unit Systems** ({func}`~unxt_api.unitsystem_of`) - Inspecting unit systems

## Using Multiple Dispatch

All functions in {mod}`unxt-api` use [plum](https://beartype.github.io/plum/)
for multiple dispatch. This means:

1. **Functions can have multiple implementations** based on argument types
2. **You can register your own implementations** for custom types
3. **Type annotations drive dispatch** - the runtime types of arguments
   determine which implementation runs

### Viewing All Implementations

To see all registered implementations of a function:

```python
import unxt_api as uapi

# View all dimension() implementations
uapi.dimension.methods

# View all uconvert() implementations
uapi.uconvert.methods

# View all unit_of() implementations
uapi.unit_of.methods
```

### Registering Custom Implementations

You can extend `unxt-api`'s dispatch system with your own types:

```python
from plum import dispatch
import unxt_api as uapi


class MyCustomQuantity:
    def __init__(self, value, unit_str):
        self.value = value
        self.unit_str = unit_str


# Register implementation for your type
@dispatch
def unit_of(obj: MyCustomQuantity, /):
    """Get unit from MyCustomQuantity."""
    return uapi.unit(obj.unit_str)


# Now it works with the dispatch system
my_q = MyCustomQuantity(5.0, "m")
uapi.unit_of(my_q)  # Unit("m")
```

<!-- ## Design Philosophy

### Why Abstract Dispatch?

The abstract dispatch pattern used in `unxt-api` provides several benefits:

1. **Extensibility**: Third-party packages can add support for their types
   without modifying unxt
2. **Type Safety**: The abstract signatures document what the API expects
3. **Flexibility**: Multiple implementations can coexist, selected automatically
   by type
4. **Loose Coupling**: Packages can depend on the API without the full
   implementation

### API vs Implementation

- **`unxt-api`**: Defines _what_ functions exist and their contracts (this
  package)
- **`unxt`**: Provides _how_ those functions work with JAX arrays and physical
  units

This separation allows:

- Lightweight packages to depend only on the API
- Multiple implementations of the same API
- Clear boundaries between interface and implementation -->

## Integration with `unxt`

The {mod}`unxt` package provides the concrete implementations of all
{mod}`unxt-api` functions. When you use:

```python
import unxt as u

q = u.Q(5, "m")
u.uconvert("km", q)
```

The `u.uconvert` function is the implementation registered by {mod}`unxt` for
the abstract `unxt_api.uconvert` function.

## For Package Authors

If you're writing a package that works with physical quantities:

### Minimal Dependency Approach

Depend on {mod}`unxt-api` to use the dispatch system without pulling in
{mod}`JAX`:

```toml
# pyproject.toml
[project]
dependencies = ["unxt-api>=X.Y.Z"]
```

Then register your implementations:

```python
from plum import dispatch


# Define a custom type (example)
class YourType:
    def __init__(self, value, unit_str, dim_str):
        self.value = value
        self.unit = unit_str
        self.dimension = dim_str


@dispatch
def unit_of(obj: YourType, /):
    """Get unit from your type."""
    return obj.unit


@dispatch
def dimension_of(obj: YourType, /):
    """Get dimension from your type."""
    return obj.dimension
```

### Full Integration Approach

If you need JAX and want full {mod}`unxt` functionality:

```toml
# pyproject.toml
[project]
dependencies = [
    "unxt>=X.Y.Z",  # Includes unxt-api transitively
]
```

Then use {mod}`unxt`'s types directly:

```python
import unxt as u


@dispatch
def your_function(q: u.Quantity, /):
    """Work with unxt quantities."""
    return q * 2
```

## See Also

- [unxt Documentation](https://unxt.readthedocs.io/) - Full implementation with
  examples
- [plum Documentation](https://beartype.github.io/plum/) - Multiple dispatch
  library
- [unxt on GitHub](https://github.com/GalacticDynamics/unxt)

## License

BSD 3-Clause License. See
[LICENSE](https://github.com/GalacticDynamics/unxt/blob/main/LICENSE) for
details.

## Contributing

Contributions are welcome! Please see the main
[unxt repository](https://github.com/GalacticDynamics/unxt) for contributing
guidelines.
