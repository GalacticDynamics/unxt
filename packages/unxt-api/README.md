# unxt-api

> **Note:** As of v2.0.0, `unxt-api` is a backward-compatible shim for [`unxts.api`](https://pypi.org/p/unxts.api) — the new canonical location for this package. No code changes are required: `import unxt_api` continues to work exactly as before. New projects should prefer `pip install unxts.api` and `import unxts.api`.

Abstract dispatch API for [unxt](https://github.com/GalacticDynamics/unxt).

`unxt-api` defines the abstract dispatch interfaces that `unxt` and other packages implement. It provides a minimal dependency foundation for packages that want to define or use `unxt`'s multiple-dispatch-based API without pulling in the full `unxt` implementation.

## Installation

```bash
pip install unxt-api          # legacy name — continues to work long-term
pip install unxts.api         # canonical name going forward
```

## Migration

If you currently depend on `unxt-api`, you have two options:

1. **Do nothing** — `unxt-api` re-exports the full public API from `unxts.api` and will be maintained long-term. Existing code requires no changes.

2. **Migrate to `unxts.api`** — change your dependency and imports:

   ```diff
   - pip install unxt-api
   + pip install unxts.api
   ```

   ```diff
   - import unxt_api as uapi
   + import unxts.api as uapi
   ```

   All public symbols (`uconvert`, `ustrip`, `unit`, `dimension`, etc.) are identical — the canonical package is the source of truth and the shim simply re-exports from it.

## Core API

- **Dimensions** — `dimension`, `dimension_of`
- **Units** — `unit`, `unit_of`
- **Quantities** — `uconvert`, `uconvert_value`, `ustrip`, `is_unit_convertible`, `wrap_to`
- **Unit Systems** — `unitsystem_of`

## Documentation

- [unxts.api documentation](https://unxt.readthedocs.io/en/latest/packages/unxts.api/)
- [unxt documentation](https://unxt.readthedocs.io/)
