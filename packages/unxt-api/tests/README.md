# unxt-api Test Suite

Comprehensive test suite for the `unxt-api` package, covering both unit tests
and usage examples.

## Test Structure

The test suite is organized into several modules:

### Unit Tests

- **`test_dimensions.py`**: Tests for dimension-related functions (`dimension`,
  `dimension_of`)
  - Abstract dispatch behavior
  - Custom dispatch registration
  - API consistency checks

- **`test_units.py`**: Tests for unit-related functions (`unit`, `unit_of`)
  - Abstract dispatch behavior
  - Custom dispatch registration
  - API consistency checks

- **`test_quantity.py`**: Tests for quantity operations (`uconvert`, `ustrip`,
  `is_unit_convertible`, `wrap_to`)
  - Abstract dispatch behavior
  - Custom dispatch registration
  - Varargs handling
  - Default implementations

- **`test_unitsystems.py`**: Tests for unit system functions (`unitsystem_of`)
  - Abstract dispatch behavior
  - Custom dispatch registration

- **`test_package.py`**: Package-level API tests
  - Module structure verification
  - Export validation
  - Import path checks
  - Documentation completeness

- **`test_advanced.py`**: Advanced scenarios and edge cases
  - Dispatch method inspection
  - Type annotations
  - Concurrent dispatch registration
  - Complex type hierarchies
  - Default behaviors
  - Edge cases (None, zeros, special floats)
  - Dispatch resolution order

### Usage Tests

- **`test_usage.py`**: Real-world usage patterns and integration examples
  - Custom type integration
  - Dimension-aware types
  - Unit conversion implementations
  - Unit stripping implementations
  - Angle wrapping
  - Unit system integration
  - Complete package integration
  - Dispatch priority
  - Error handling

## Running Tests

Run all tests (including doctests):

```bash
uv run pytest packages/unxt-api
```

Run only unit tests:

```bash
uv run pytest packages/unxt-api/tests
```

Run tests with verbose output:

```bash
uv run pytest packages/unxt-api/tests -v
```

Run a specific test file:

```bash
uv run pytest packages/unxt-api/tests/test_usage.py
```

Run a specific test class or function:

```bash
uv run pytest packages/unxt-api/tests/test_usage.py::TestCustomTypeIntegration::test_simple_quantity_type
```

## Test Coverage

The test suite covers:

1. **Abstract Dispatch Behavior**
   - All functions are proper plum dispatch functions
   - Accept appropriate type signatures
   - Support custom dispatch registration

2. **API Consistency**
   - All expected functions are exported
   - Functions are listed in `__all__`
   - Independent dispatch registries
   - Proper module structure

3. **Custom Type Integration**
   - Simple quantity types
   - Dimension-aware types
   - Unit conversion
   - Unit stripping
   - Angle wrapping
   - Unit systems

4. **Advanced Scenarios**
   - Type hierarchies and inheritance
   - Multiple dispatch coexistence
   - Ambiguous dispatch handling
   - Edge cases and boundary conditions

5. **Real-World Usage**
   - Complete package integration examples
   - Multiple packages using the same dispatch system
   - Error handling patterns

## Notes for Test Maintainers

- **Dispatch Registry Pollution**: Some tests account for the fact that when
  `unxt` is imported (e.g., during doctest execution), it registers default
  implementations for certain functions. Tests that check for "no default
  implementation" are written to handle both scenarios.

- **Custom Types in Fixtures**: The `conftest.py` file provides fixtures for
  creating custom types used in tests. This keeps test code DRY and ensures
  consistency.

- **Plum Dispatch Testing**: Tests verify plum dispatch behavior without
  depending on implementation details. They check that:
  - Functions are dispatch functions
  - Custom dispatches can be registered
  - Dispatch resolution works as expected
  - Appropriate errors are raised

## Adding New Tests

When adding new API functions to `unxt-api`:

1. Add unit tests in the appropriate file (e.g., `test_dimensions.py` for
   dimension-related functions)
2. Add usage examples in `test_usage.py` showing real-world integration
3. Update `test_package.py` to include the new function in export checks
4. Consider edge cases and add them to `test_advanced.py` if needed

## Test Philosophy

The tests follow these principles:

1. **Test the interface, not the implementation**: Tests verify that the
   dispatch system works correctly, not how it's implemented internally.

2. **Realistic usage patterns**: Usage tests show how third-party packages would
   actually use the API.

3. **Comprehensive coverage**: Both happy paths and edge cases are tested.

4. **Documentation through tests**: Tests serve as examples of how to use the
   API correctly.

5. **Isolation where possible**: Each test should be independent and not rely on
   side effects from other tests.
