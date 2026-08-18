---
name: code-review
description: >
  Use when reviewing a pull request or diff in the unxt repository. Covers the unxt-specific defects that generic review misses — a Quantity built with the unchecked `_mk` from unnormalised input, a plum-dispatched conversion whose return annotation quietly disables caching, a new parametric/canonical package that duplicates a legacy `unxt-*` shim instead of extending `unxts.*`, unit-system code that reintroduces mutable shared state, and a doctest whose printed output no longer matches what JAX actually produces.
---

# unxt code review

- `unxt`'s objects are immutable JAX PyTrees (`quax.ArrayValue` + Equinox) constructed almost entirely through `plum`-dispatched converters — most real defects come from _bypassing_ that construction path (directly, via `_mk`, or indirectly, via a dispatch rule that disagrees with the checked path it's supposed to match).
- Every public docstring `Examples` block and every `python` block in `README.md`/`docs/**` is a real, exact-match doctest run by Sybil — a plausible-looking example that wasn't actually run against current JAX/unxt is a shipped bug waiting to be noticed by a user, not a maintainer.

## Scope of this review

- Don't restate what CI already catches: ruff (`nox -s lint`), pyright/ty/mypy (pre-commit, scoped to `tests/typing/`), pytest+Sybil, CodSpeed. If a check is purely mechanical and already enforced, skip it here.
- No generic security checklist — no user input, no network, no serialization of untrusted data in this library.
- Don't second-guess JAX's/astropy's own numeric correctness for cases already covered by their test suites; focus on unxt's own dispatch/construction/unit-safety layer on top.

## What changed → what to check

| Change touches | Read |
| --- | --- |
| A constructor, `revalue`, or any new/changed use of `_mk` | [`_mk` usage](#_mk-usage) |
| A `plum.dispatch`/`convert`/`unit()`/`dimension()` method, or a `@parametric` type | [Dispatch and conversion](#dispatch-and-conversion) |
| `src/unxt/_src/unitsystems/**` or unit-system construction | [Unit systems](#unit-systems) |
| Anything under `packages/unxt-api/`, `packages/unxt-hypothesis/`, or new files in `packages/unxt-*` | [Canonical vs. legacy packages](#canonical-vs-legacy-packages) |
| A docstring `Examples` block, or `python` blocks in `README.md`/`docs/**` | [Doctests](#doctests) |
| `except`/fallback/assert paths, or anything changing what an error message says | [Silent failure](#silent-failure) |
| `tests/**` | [Tests](#tests) |

## `_mk` usage

`_mk` writes `value`/`unit` fields directly, skipping both the plum-dispatched converters and `__check_init__`. Any diff introducing or extending a call to `_mk` is a correctness question by default, not a style nit:

- Does the PR show — in the diff itself or the surrounding code — that `value` and `unit` are provably normalised at the call site (right array/dtype, `unit` already an `AbstractUnit` instance, not a string)? If that proof isn't visible, ask for it.
- Is this actually a hot path (construction-heavy loop, `jax.jit`-traced primitive registration)? `_mk` exists for measured perf wins (#816, #829, #839, #840) — using it "for consistency" or speculatively is not a justification.
- If the type being built overrides `_mk` back to the checked constructor (`StaticQuantity` does this), a new subtype with a load-bearing converter should do the same — flag a new type that inherits the unchecked `_mk` without considering whether its converter matters.
- `revalue` is the vetted way to state the normalisation invariant reusably; a one-off inline `_mk` call scattered through business logic (vs. isolated in a primitive-registration or conversion module) is a smell.

## Dispatch and conversion

- Does a new/changed `plum.dispatch` method actually agree with the behavior of the checked constructor or converter it's meant to complement? A registered method that silently diverges from the "normal" path is worse than no method — it produces a `Quantity` that looks fine but violates an invariant elsewhere expects.
- Concrete return-type annotations on dispatched functions are load-bearing for plum's resolution cache, not decorative — a missing or overly-generic annotation can silently cost real dispatch-resolution time in hot paths. If the diff touches a frequently-called dispatched function, check the annotation is as concrete as the actual return.
- `@parametric` classes (e.g. `ParametricQuantity[dim]`) are cached per type-parameter — check that a new parametric type's parameter is hashable and that construction doesn't defeat that caching (e.g. inferring the parameter fresh every call instead of memoizing, cf. #834/#840).
- For a new conversion (`Quantity.from_`, `plum.conversion_method`, an interop package's converter), check both directions are covered if both are meaningful, and that the dimension/type-mismatch path raises rather than silently truncating.

## Unit systems

- Built-in unit systems (`si`, `cgs`, `galactic`, dimensionless) are shared singletons — #704/#718 fixed exactly this class of bug (mutation of a singleton corrupting global state). Any diff that adds mutable state to a unit-system instance, or that mutates a unit system in place instead of returning a new one, should be rejected.
- Unit-system identity/equality should stay order-independent (#778) — a diff that makes system identity depend on construction-argument order is regressing a fixed bug, not adding a feature.
- `equivalent()` on unit systems must consult actual units, not just dimensions (#783) — a "same dimensions, different scale" pair should not compare equivalent.

## Canonical vs. legacy packages

`unxt-api`/`unxt-hypothesis` (hyphenated) are back-compat shims that re-export `unxts.api`/`unxts.hypothesis` (dotted, canonical). New functionality must land in the canonical `unxts.*` package:

- A diff that adds real logic (not a re-export) inside `packages/unxt-api/` or `packages/unxt-hypothesis/` is misplaced — it belongs in the corresponding `unxts.*` package.
- A diff that duplicates a check/helper between a shim and its canonical package (rather than the shim simply re-exporting) will trip `pylint`'s `duplicate-code` the moment someone lints across packages instead of per-package — point this out even though per-package `nox -s pylint` runs won't catch it locally.
- New workspace packages should follow the canonical `unxts.<name>` naming and the hatch-vcs versioning template already used by the existing packages (see AGENTS.md's Workspace Packages section), not the legacy hyphenated style.

## Doctests

- Every changed or added `Examples` block (docstring) or `python`-tagged block (`README.md`, `docs/**`) must show output that was actually produced by running it — not hand-typed. Sybil matches exactly, including `dtype=float32` vs a hand-typed `dtype=float`.
- Prefer `u.Q(...)` over `u.Quantity(...)` (house style) and real physically-meaningful values over toy `x = 1` examples, per `CONTRIBUTING.md`/`docs/conventions.md`.
- A new `filterwarnings` ignore entry needs a comment saying which library/version emits it and why it's expected — an unscoped or uncommented ignore is a smell (warnings are errors project-wide by design).

## Silent failure

- A `try`/`except` that narrows an error into a vaguer message, or that swallows a `plum` ambiguity/no-method error into a generic fallback, should be justified in the PR — unxt's dispatch errors are usually more informative than a caught-and-rewrapped version.
- Check that a new dimension/unit-mismatch code path actually raises rather than coercing silently — e.g. a new arithmetic operator overload that falls back to treating an incompatible unit as dimensionless instead of raising.

## Tests

- New dispatch methods or `_mk` usage need a test that would fail if the normalisation invariant were violated (not just a happy-path call) — cf. `test_mk_matches_static_value_from_` as the model.
- Property-based tests (`unxts.hypothesis` strategies) are the right tool for quantity arithmetic laws (associativity, unit-conversion round-trips) — a PR adding new arithmetic behavior without a property test covering it is under-tested even if the example-based tests pass.
- Optional-dependency-gated tests (interop packages) must actually skip cleanly when the dependency is absent — check `collect_ignore_glob`/`OptionalDependencyEnum` usage rather than a bare `pytest.importorskip` scattered inline.
- A performance-sensitive change (anything using `_mk`, memoization, or touching `unxts.linalg`'s primitive registrations) should add or update a `tests/benchmark` case — reviewers can't otherwise verify the claimed win, and CI won't run it without the `run-benchmarks` label.

## Repo conventions

- `uv run nox -s ...`, never bare `pytest`/`ruff`/`python`.
- Conventional commits + gitmoji (commitizen); scope is the package/module.
- The pre-commit `ruff-check` hook runs with `--fix --show-fixes` and can modify files — a "ruff check passed" claim based on a bare `uv run ruff check` isn't the same gate as CI's pre-commit run.
- pylint runs per-package (`nox -s pylint -- <package>`) to avoid cross-package `duplicate-code` false positives; don't assume a clean per-package pylint run means the whole tree is clean of duplication.

## Further reading

- [../../../AGENTS.md](../../../AGENTS.md) — commands, architecture, pitfalls
- [../../../skills/unxt/SKILL.md](../../../skills/unxt/SKILL.md) — consumer-facing usage skill (same `_mk` warning, from the other side)
- [../../../CONTRIBUTING.md](../../../CONTRIBUTING.md)
- Upstream: [quax code-review skill](https://github.com/nstarman/quax/tree/main/.github/skills/code-review), [quaxed code-review skill](https://github.com/GalacticDynamics/quaxed/blob/main/.github/skills/code-review/SKILL.md)
