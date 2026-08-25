# unxt — Agent Instructions

`unxt` is unitful quantities and calculations in JAX, built on [Equinox](https://github.com/patrick-kidger/equinox), [Quax](https://github.com/nstarman/quax), [quaxed](https://github.com/GalacticDynamics/quaxed), and [quax-blocks](https://github.com/GalacticDynamics/quax-blocks), with [plum](https://github.com/beartype/plum) multiple dispatch throughout. It's the foundation quantity library other GalacticDynamics packages (coordinax, galax) build on.

For _using_ `unxt` correctly — `Quantity` vs `ParametricQuantity`, dispatch gotchas, the `_mk` hazard — read [skills/unxt/SKILL.md](skills/unxt/SKILL.md). This file is for working _inside_ this repo.

## Essential commands

```bash
uv sync --group dev --extra all      # install, all extras + dev tooling
uv run nox -s all                    # the full gate: lint -> test -> docs
uv run nox -s lint                   # pre-commit (incl. pyright/ty/mypy) + pylint
uv run nox -s test                   # pytest, every workspace package
uv run nox -s pytest -- unxt         # pytest for one package only (see PackageEnum)
uv run nox -s docs -- --serve        # build + preview the Sphinx site
uv run nox -s docs -- -b linkcheck   # check doc links
uv run nox -s pytest_benchmark       # CodSpeed benchmarks (also gated by the `run-benchmarks` PR label)
```

Always go through `uv run`/`nox` — never bare `python`/`pytest`/`ruff`. Sync first if `uv.lock` moved.

## Workspace layout

Root `unxt` (`src/unxt/`) plus a `uv` workspace at `packages/*`. Two package families, same functionality, **different names — don't confuse them**:

| Family | Packages | Status |
| --- | --- | --- |
| **canonical** (dotted, `unxts.*`) | `unxts.api`, `unxts.hypothesis`, `unxts.interop.gala`, `unxts.interop.matplotlib`, `unxts.interop.xarray`, `unxts.linalg`, `unxts.parametric` | current, since v2.0.0 |
| **legacy shims** (hyphenated, `unxt-*`) | `unxt-api`, `unxt-hypothesis` | back-compat re-exports of `unxts.api`/`unxts.hypothesis`; new code should depend on the canonical package |

New functionality goes in a canonical `unxts.*` package, never in a shim. Release tags are hyphenated even for dotted packages (`unxts.api` → tag `unxts-api-vX.Y.Z`) — see [RELEASING.md](RELEASING.md).

| Package | Provides |
| --- | --- |
| `unxt` (root) | `Quantity`/`Q`, `Angle`, `StaticQuantity`, units, dims, unit systems, plum dispatch API |
| `unxts.api` | abstract dispatch interfaces (`uconvert`, `ustrip`, `unit`, `dimension`, ...), minimal deps |
| `unxts.hypothesis` | Hypothesis strategies for property-based testing of quantities |
| `unxts.parametric` | `ParametricQuantity`/`PQ` — dimension baked into the type, runtime-checked |
| `unxts.linalg` | `QuantityMatrix`/`QM`, `UnitsMatrix` — heterogeneous-unit linear algebra |
| `unxts.interop.gala` | `gala.units.UnitSystem` ↔ unxt `UnitSystem`, via `plum.conversion_method` |
| `unxts.interop.matplotlib` | `matplotlib.units.ConversionInterface` for plotting quantities |
| `unxts.interop.xarray` | xarray accessors/conversion for quantities |

## Architecture

`AbstractQuantity` (a `quax.ArrayValue`, so it's a JAX PyTree via Equinox) is the base of the whole hierarchy:

- **`Quantity`/`Q`** (default, root `unxt`) — non-parametric: one class, one pytree node type, for every physical dimension. No dimension checking at construction.
- **`ParametricQuantity`/`PQ`** (`unxts.parametric`, opt-in) — dimension encoded in the type (`PQ["length"]`), a distinct pytree type per dimension, runtime-checked at construction.
- **`BareQuantity`** — **deprecated** alias of `Quantity`; don't reintroduce it in new code (see [docs/reference/glossary.md](docs/reference/glossary.md), [docs/how-to/migrate-to-v2.md](docs/how-to/migrate-to-v2.md)).
- **`StaticQuantity`** — value held as a hashable static field (for `jax.jit(static_argnames=...)`); equality is unit-label-based by design (`same_unit_label`), not physical equivalence.
- **`Angle`** — wrapping-aware `Quantity` subtype.

Dims (`unxt.dims`) parse expressions via a small AST-based grammar in `src/unxt/_src/dimensions.py` — `* / ** ()` are supported, unary `+`/`-` deliberately raise ("dimensions are invariant under negation," not a missing feature). Units (`unxt.units`) wrap `astropy.units`; `AbstractUnit = apyu.UnitBase | apyu.FunctionUnitBase` (`StructuredUnit` is deliberately excluded). Unit systems live under `src/unxt/_src/unitsystems/`.

Naming rule (see [docs/explanation/api-conventions.md](docs/explanation/api-conventions.md)): `Abstract...` prefix marks a non-instantiable base; no abstract class inherits from a concrete one, no concrete class inherits from another concrete one.

## The `_mk` unchecked constructor

`AbstractQuantity._mk` (and its per-type overrides, e.g. `QuantityMatrix._mk`) is a private, unexported fast-path constructor: it writes `value`/`unit` fields directly, skipping the plum-dispatched converters and `__check_init__` (~1µs vs ~50µs for the checked path — see #816/#829/#839/#840). It is **only sound when the caller has already normalised the value and unit**; that proof belongs at the call site (`revalue` is the vetted way to state it). `StaticQuantity` overrides `_mk` back to the checked constructor because its converter is load-bearing, not redundant — don't "simplify" that override away.

Grep `_mk\(` before touching construction-hot code — it's used across `unxts.linalg`'s primitive registrations, `unxts.parametric`'s conversion path, and `unxts.interop.xarray`'s conversion helper. See [skills/unxt/SKILL.md](skills/unxt/SKILL.md) for the consumer-facing warning.

## Testing

- **Doctests are load-bearing tests.** Every public docstring `Examples` block, plus every `python`-tagged Markdown block in `README.md`/`docs/**`, is executed by [Sybil](https://sybil.readthedocs.io/) (`-p no:doctest`, Sybil replaces stdlib doctest). Output must match exactly, including `dtype=float32`.
- `--import-mode=importlib` is required — the monorepo has same-named test files (`test_public_api.py`) across packages with no `__init__.py`.
- `filterwarnings = ["error", ...]` — a warning you didn't expect is a test failure, not noise; check the curated ignore list in `pyproject.toml` before adding a new one.
- `xfail_strict = true`; `UNXT_ENABLE_RUNTIME_TYPECHECKING=beartype.beartype` is set in the test env (off by default at runtime).
- Optional-dependency tests auto-skip via `optional_dependencies.OptionalDependencyEnum`; `conftest.py` manages `collect_ignore_glob`.
- `pytest-benchmark`/`pytest-codspeed` benchmarks live in `tests/benchmark`; CI only runs them on PRs carrying the `run-benchmarks` label.
- Each package is linted with `pylint` **in isolation** (`nox -s pylint -- <package>`) — running it over the whole tree produces false `duplicate-code` positives between a shim and its canonical package.

## Pitfalls

- **`ruff-check` pre-commit hook autofixes.** It runs with `--fix --show-fixes`, so it can _modify files_, unlike a plain `uv run ruff check` which only reports. Run `uv run nox -s precommit` (or the specific hook) before assuming `ruff check` alone is the gate.
- **pyright/ty/mypy are scoped to `tests/typing/` only**, pinned versions (`pyright==1.1.411`, `ty==0.0.64`, `mypy==2.3.0`), run as local pre-commit hooks via `uv run --frozen --group typecheck --extra all <tool>`. They guard `Quantity(1, "m")`-style constructor typing, not the whole tree; mypy in particular sees the `unit` parameter as `Any`.
- `F811`/`F821`/`F722` are ruff-ignored project-wide — they're false positives from plum-dispatch redefinition and jaxtyping shape annotations, not oversights.
- `UP040` (prefer `type` alias) is ignored: `beartype.door` doesn't support PEP 695 `type` aliases with plum, so `TypeAlias` stays.
- Import aliases are enforced by ruff (`flake8-import-conventions`): `unxt as u`, `equinox as eqx`, `hypothesis.strategies as st`, `unxt_api as uapi`, `unxt_hypothesis as ust`. Follow the same convention for new canonical imports even where ruff doesn't enforce it yet (e.g. `unxts.parametric as up`, per README usage).
- **A `jax.jit` wrapper rebuilt inside a loop or method is a compile-cache miss every call, not a cheap re-trace.** `jax.jit` keys its cache on the Python identity of the function it wraps, not on argument equality — a fresh `@jax.jit def outer(...)` built per call costs a full retrace-and-compile every time (measured: several-hundred-x). Build the outer-wrapper closure once, at module or `__init__` scope. See `docs/how-to/optimize-performance.md`.
- Never write temp/scratch files outside the repo.

## Commit style

Conventional commits + gitmoji, enforced by `commitizen` (`cz-conventional-gitmoji`) as a pre-commit hook: `<emoji> <type>(<scope>): <description> (#PR)`. Scope is typically the package/module. Real examples from history:

```
⚡️ perf(parametric): memoize the inferred dimension, build conversions with `_mk` (#840)
🧹 chore(linalg): drop the strict_zip wrapper and a dead import (#828)
♻️ refactor(unitsystems): one registry, and plain class attrs for the ordered views (#821)
⬆️ dep-bump(astropy): drop the astropy<7.1 compatibility shim (#830)
➖ dep-rm: drop two unused runtime dependencies (#819)
✨ feat(linalg): add UnitsMatrix construction and diagonal helpers (#812)
🐛 fix(unitsystems): stop SI/CGS/dimensionless singletons corrupting globals (#704) (#718)
💥 boom(unitsystems)!: make `repr` round-trippable, `str` readable
👷 ci(release): drive package releases from GitHub App tag pushes (#797)
```

No `CHANGELOG.md` — intentional; GitHub Releases (generated at tag-push time) are the changelog, per [RELEASING.md](RELEASING.md).

## Dependencies & release

Dependency floors follow SPEC 0 roughly (see `pyproject.toml` for exact pins: `jax>=0.7.2`, `plum-dispatch>=2.7.0`, `quax>=0.4.2`, `quax-blocks>=0.5.0`, `quaxed>=0.10.5`, `astropy>=7.1`, Python `>=3.12`). Multi-package, tag-driven releases: a `vX.Y.0` coordinator tag releases everything together; any package can also get an independent `<pkg>-vX.Y.Z` bug-fix tag. Full detail, including the GitHub App token setup required for tag-triggered CD, is in [RELEASING.md](RELEASING.md).

## Further reading

- [skills/unxt/SKILL.md](skills/unxt/SKILL.md) — using `unxt` correctly (consumer-facing)
- [.github/skills/code-review/SKILL.md](.github/skills/code-review/SKILL.md) — reviewing PRs in this repo
- [docs/explanation/api-conventions.md](docs/explanation/api-conventions.md), [docs/reference/glossary.md](docs/reference/glossary.md), [docs/how-to/migrate-to-v2.md](docs/how-to/migrate-to-v2.md)
- [CONTRIBUTING.md](CONTRIBUTING.md), [RELEASING.md](RELEASING.md)
- Upstream skills this repo builds on: [quax](https://github.com/nstarman/quax/blob/main/skills/quax/SKILL.md), [quaxed](https://github.com/GalacticDynamics/quaxed/blob/main/skills/quaxed/SKILL.md), [quax-blocks](https://github.com/GalacticDynamics/quax-blocks/blob/main/skills/quax-blocks/SKILL.md)
