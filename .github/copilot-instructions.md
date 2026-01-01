# Project Overview

This is a UV workspace repository containing multiple packages:

- **unxt**: Main library for unitful quantities in JAX with support for JIT
  compilation, auto-differentiation, vectorization, and GPU/TPU acceleration
- **unxt-api**: Abstract dispatch API that defines the multiple-dispatch
  interfaces implemented by `unxt` and other packages. Minimal dependencies
  (only `plum-dispatch`).
- **unxt-hypothesis**: Hypothesis strategies for property-based testing with
  `unxt`

## Main Package: unxt

- **Language**: Python 3.11+
- **Main API**: `Quantity`, `Angle`, unit conversion and manipulation
  - `Quantity`: Main parametric class with dimension checking
  - `Angle`: Specialized type with wrapping support
  - `unit()`, `dimension()`, unit systems
- **Design goals**: JAX-compatible quantities, dimension checking, seamless
  integration with existing JAX code via Quax
- **JAX integration**: Objects are PyTrees via Equinox. Use `quaxed` for
  pre-quaxified JAX functions. Performant with JIT, vmap, grad.

## Architecture & Core Components

- **Quantity types** (hierarchical):
  - `AbstractQuantity`: Base class using Quax's `ArrayValue` for JAX integration
  - `AbstractParametricQuantity`: Enables dimension parametrization
    (`Quantity["length"]`)
  - `Quantity`: Main parametric class with runtime dimension checking
  - `BareQuantity`: Lightweight variants without dimension checks
  - `Angle`: Specialized type with wrapping support
- **Units system**: Wraps Astropy units, provides `unit()`, `unit_of()`
  constructors
- **Dimensions**: Physical type checking via `dimension()`, `dimension_of()`
- **Unit systems**: `AbstractUnitSystem` for consistent unit sets
- **Interop modules**: Optional integration with Astropy, Gala, Matplotlib (in
  `_src/_interop/`)

## Folder Structure

### Root Level (UV Workspace)

- `/src/unxt/`: Main package public API with re-exports
- `/packages/`: Workspace packages
  - `unxt-api/`: Abstract dispatch API package
  - `unxt-hypothesis/`: Hypothesis strategies package
- `/tests/`: Main package tests, organized into `unit/`, `integration/`,
  `benchmark/`
- `README.md`: Main package documentation, tested via Sybil (all Python code
  blocks are doctests)
- `conftest.py`: Pytest config, Sybil setup, optional dependency handling
- `noxfile.py`: Task automation with dependency groups
- `pyproject.toml`: Root workspace configuration with `[tool.uv.workspace]`

### Main Package Structure (`/src/unxt/`)

- `_src/`: Private implementation code
  - `quantity/`: Quantity classes and operations
  - `units/`: Unit handling and wrapping
  - `dimensions/`: Physical dimension types
  - `unitsystems/`: Unit system definitions
- `_interop/`: Optional dependency integrations

## Coding Style

- Always use type hints (standard typing, `jaxtyping.Array`, `ArrayLike`, shape
  annotations)
- Extensive use of Plum multiple dispatch - check `.methods` on any function to
  see all dispatches
- `@parametric` decorator enables dimension parametrization:
  `Quantity["length"]`
- Runtime type checking controlled by `UNXT_ENABLE_RUNTIME_TYPECHECKING` env var
  (defaults to `False`, set to `"beartype.beartype"` in tests)
- Immutability is a core constraint: methods return new objects, never mutate
- Keep dependencies minimal; the core dependencies are listed in
  `pyproject.toml`
- Docstrings should be concise and include testable usage examples
- `__all__` should always be a tuple unless it needs to be modified (e.g., with
  `+=`), in which case use a list - prefer immutable by default

### Multiple Dispatch with Plum

This project heavily relies on `plum-dispatch` for multiple dispatch, which
allows different implementations of the same function based on argument types.
Understanding how plum works is critical for working with this codebase.

#### Multiple Dispatch Mechanism

- **Single-dispatch vs Multiple-dispatch**: Unlike single dispatch (e.g.,
  `functools.singledispatch`), plum selects implementations based on ALL
  argument types, not just the first one
- **Type-based routing**: Plum examines the runtime types of all arguments and
  selects the most specific matching implementation
- **Dispatch decorator**: Use `@dispatch` to register multiple implementations
  of the same function name

Example:

```python
from plum import dispatch


@dispatch
def process(x: int) -> str:
    return f"integer: {x}"


@dispatch
def process(x: float) -> str:
    return f"float: {x}"


@dispatch
def process(x: int, y: int) -> str:
    return f"two integers: {x}, {y}"
```

#### Finding All Dispatches

**CRITICAL**: When working with dispatched functions, you MUST check all
registered implementations. A function may have dozens of overloads.

**Two methods to find all dispatches:**

1. **Use `.methods` attribute** (preferred in Python REPL/notebooks):

   ```python
   from unxt import ustrip

   print(ustrip.methods)  # Shows all registered dispatch signatures
   ```

2. **Search the codebase** (preferred when coding):
   - Search for `@dispatch` followed by the function name
   - Look for all `def function_name(...)` definitions with `@dispatch`
   - Example: searching for `@dispatch\ndef ustrip` finds all ustrip overloads

**Why this matters:**

- You might find a more specific dispatch that handles your exact case
- Prevents accidentally adding duplicate dispatches
- Reveals the complete API surface and supported type combinations
- Essential for understanding how different types interact

#### Parametric Classes

Plum's `@parametric` decorator enables type parametrization, creating distinct
types for different parameters:

```python
from plum import parametric


@parametric
class Container(type_parameter):
    def __init__(self, value):
        self.value = value


# Creates distinct types:
IntContainer = Container[int]
FloatContainer = Container[float]
```

**In this codebase:**

- `Quantity` is parametric by dimension: `Quantity["length"]`,
  `Quantity["mass"]`
- Each parametrization creates a distinct type for dispatch
- Enables dimension-aware multiple dispatch:

  ```python
  @dispatch
  def divide(x: Quantity["length"], y: Quantity["time"]) -> Quantity["speed"]:
      ...  # Returns speed dimension
  ```

**Key properties:**

- Parametric types are cached (same parameters = same type object)
- Type parameters can be strings, tuples, or other hashable objects
- Use `get_type_parameter(obj)` to retrieve the parameter from an instance
- Parametric classes enable dimension checking at dispatch time

#### Type Promotion with `plum.promote`

`plum.promote` implements automatic type promotion for mixed-type operations:

```python
from plum import dispatch, promote


@dispatch
def add(x: int, y: int) -> int:
    return x + y


@dispatch
def add(x: float, y: float) -> float:
    return x + y


# Without promotion:
add(1, 2.5)  # Error: no dispatch for (int, float)


# With promotion (defined separately):
@dispatch
def add(x: promote(int, float), y: float) -> float:
    return add(float(x), y)
```

**In this codebase:**

- `plum.promote` is used to convert between quantity types
- Common pattern: promote bare numbers to quantities
- Enables natural operations like `quantity + 5.0`

**Usage pattern:**

1. Define core implementations for specific types
2. Add promotion dispatches to handle mixed types
3. Promotion dispatches typically convert arguments and redispatch

**Important notes:**

- Promotion order matters: `promote(int, float)` != `promote(float, int)`
- Keep promotion logic explicit and minimal
- Prefer concrete dispatches over heavy promotion use
- Document promotion behavior when it's non-obvious

### JAX Integration via Quax

- Quantities are `ArrayValue` subclasses (Quax protocol)
- PyTree registration handled automatically via Equinox
- Use `quaxed` library (pre-quaxified JAX) for convenience, or manually apply
  `quax.quaxify` decorator
- Mixins from `quax-blocks` provide operator overloading (`NumpyBinaryOpsMixin`,
  etc.)

### Immutability

- All quantity operations return new instances
- Use `dataclassish.replace()` for attribute updates
- Follow Equinox patterns for JAX compatibility

### Import Hook

- `setup_package.py` installs jaxtyping import hook for runtime checking
- Not required for normal usage but enables beartype integration during tests

## Tooling

- This repo uses `uv` for dependency and environment management
- This repo uses `nox` for all development tasks
- Before committing, run full checks:
  ```bash
  uv run nox -s all
  ```
- Common sessions:
  - `nox -s lint`: pre-commit + pylint
  - `nox -s test`: pytest suite
  - `nox -s docs`: build documentation (add `--serve` to preview)
  - `nox -s pytest_benchmark`: run CodSpeed benchmarks

**IMPORTANT**: Never write temporary files outside the repository (e.g., to
`/tmp/` or other system directories). Always use paths within the repository for
any file operations, including temporary or scratch files.

## Testing

- Use `pytest` for all test suites with Sybil for doctests in code and markdown
- Add unit tests for every new function or class
- Test organization: `unit/`, `integration/`, `benchmark/`
- **All tests must actually test something**: Every test function must include
  `assert` statements or return values that pytest can validate. Empty test
  bodies or tests that only call functions without verification are not valid.
- Optional dependencies handled via
  `optional_dependencies.OptionalDependencyEnum`
  - Tests requiring optional deps auto-skip if not installed
  - `conftest.py` manages `collect_ignore_glob` for missing deps
- For JAX-related behavior:
  - Confirm PyTree registration works correctly (flatten/unflatten)
  - Verify compatibility with transformations like `jit`, `vmap`, and `grad`
  - Test numerical accuracy where applicable
  - Tests should run on CPU by default; no accelerators required
- Hypothesis for property-based testing of quantity laws

## Optional Dependencies

Three optional interop groups:

- `backend-astropy`: Enhanced Astropy integration
- `interop-gala`: Galactic dynamics library support
- `interop-mpl`: Matplotlib quantity plotting

Install with: `uv add unxt --extra all` or specific groups

## Workspace Packages

This repository uses a UV workspace structure with multiple packages (e.g.,
`unxt`, `unxt-api`, `unxt-hypothesis`). When creating new workspace packages,
use this versioning setup pattern:

```toml
[build-system]
build-backend = "hatchling.build"
requires      = ["hatch-vcs", "hatchling"]

[tool.hatch.version]
raw-options = { root = "../..", search_parent_directories = true, git_describe_command = "git describe --dirty --tags --long --match '<package-name>-v*'", local_scheme = "no-local-version" }
source      = "vcs"

[tool.hatch.build.hooks.vcs]
version-file = "src/<package_name>/_version.py"
version-file-template = """\
version: str = {version!r}
version_tuple: tuple[int, int, int] | tuple[int, int, int, str, str]
version_tuple = {version_tuple!r}
"""

[tool.uv.sources]
unxt = { workspace = true }
```

Replace `<package-name>` with the actual package name (e.g.,
`unxt-hypothesis-v*`) and `<package_name>` with the Python module name (e.g.,
`unxt_hypothesis`). This enables automatic versioning from git tags.

## Final Notes

Preserve JAX compatibility and immutability above all. When extending quantity
operations, ensure dimension checking is correct and test with JAX
transformations. Follow Equinox/Quax patterns for custom array types.
Documentation examples must be executable (they're tested).
