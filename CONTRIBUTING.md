# Contributing to `unxt`

## Reporting Issues

When opening an issue to report a problem, please try to provide:

- a minimal code example that reproduces the issue
- and details of the operating system and the dependency versions you are using.

## Contributing Code and Documentation

So you are interested in contributing to the `unxt`? Excellent! We love contributions! `unxt` is open source, built on open source, and we'd love to have you hang out in our community.

## How to Contribute, Best Practices

Most contributions to `unxt` are done via [pull requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) from GitHub users' forks of the [`unxt` repository](https://github.com/GalacticDynamics/unxt). If you are new to this style of development, check out Astropy's [development workflow](https://docs.astropy.org/en/latest/development/quickstart.html).

Once you open a pull request (which should be opened against the `main` branch, not against any of the other branches), please make sure to include the following:

- **Code**: the code you are adding.

- **Tests**: these are usually tests to ensure code that previously failed now works (regression tests), or tests that cover as much as possible of the new functionality to make sure it does not break in the future and also returns consistent results on all platforms (since we run these tests on many platforms/configurations).

- **Documentation**: if you are adding new functionality, be sure to include a description in the main documentation (in `docs/`). The docs follow [Diátaxis](https://diataxis.fr) — pick `tutorials/`, `how-to/`, `reference/` or `explanation/` for your page; see [Contributing](https://unxt.readthedocs.io/en/latest/about/contributing.html) for which is which.

- **Performance improvements**: if you are making changes that impact `unxt` performance, consider adding a performance benchmark in `tests`. A maintainer will also be able to run comparative benchmarks to catch performance changes. The PR needs to have the `run-benchmarks` label to run the workflow.

## Exploring the Codebase

`unxt` is wired up for [graphify](https://github.com/Graphify-Labs/graphify), which turns the source into a queryable knowledge graph: every symbol, the links between them, the architectural hubs, and which subsystem each symbol belongs to. Extraction is local tree-sitter AST parsing: no LLM, and nothing leaves your machine. Every edge is tagged `EXTRACTED` (explicit in the source) or `INFERRED` (resolved by graphify), so you can tell what was read directly from what was guessed. Querying the graph is usually cheaper and more accurate than grepping. AI assistants are told to query it before their first grep (see [AGENTS.md](https://github.com/GalacticDynamics/unxt/blob/main/AGENTS.md)), and you can too.

The graph itself — `graphify-out/graph.json` — is **committed**, so every clone has it with no build step. The rest of `graphify-out/` (`graph.html`, the clickable graph; `GRAPH_REPORT.md`, with hubs, communities and suggested questions; some caches) is generated locally and gitignored.

### One-time setup

Install the pinned release and register the merge driver, so two branches that both changed the graph union-merge instead of conflicting. Git will not take a merge-driver command from a tracked file, so this cannot be committed for you — it is once per clone:

```bash
uv tool install "graphifyy==0.9.119"   # the pin in .pre-commit-config.yaml
PY="$(sed -n '1s/^#!//p' "$(command -v graphify)")"
git config merge.graphify.name "graphify graph.json union merge"
git config merge.graphify.driver "\"$PY\" -m graphify merge-driver %O %A %B"
```

Pointing at the interpreter rather than the `graphify` script means merges also work in GUI git clients, where `PATH` is not your shell's. If you would rather not install anything, `uv run --no-project --with graphifyy==0.9.119 graphify` works for queries (`alias gq=` that); the merge driver does want the install.

### Querying it

```bash
graphify query "how are units converted"      # a scoped subgraph for a question
graphify explain "AbstractQuantity"           # one symbol and its neighbours
graphify path "Quantity" "AbstractUnitSystem" # how two things connect
graphify god-nodes                            # the architectural hubs
graphify update . --force                     # refresh after code changes, no LLM
open graphify-out/graph.html                  # the clickable graph (any rebuild writes it)
```

### Day to day

- The `graphify-update` pre-commit hook rebuilds the graph on every commit — a full rebuild, about 13 seconds with a warm cache. Like a formatter, it rebuilds and **stops the commit**: `git add graphify-out/graph.json` and commit again.
- Commit without rebuilding: `SKIP=graphify-update git commit ...`. The weekly workflow below will notice and open a pull request.
- Rebuild by hand the way the hook does: `PYTHONHASHSEED=0 graphify update . --force`. Run it in a normal clone: in a `git worktree` the scan picks up the worktree's `.git` _file_ and the graph comes out with different community names and a few differently-classified `__init__.py` edges, so a worktree-built graph churns against everyone else's. If you develop in worktrees, `SKIP=graphify-update` and let the weekly workflow rebuild it. Both flags matter — without `PYTHONHASHSEED=0` the communities renumber, and an incremental rebuild (no `--force`) leaves nodes behind for code that changed, so the graph slowly drifts from what a fresh clone would build.
- If the graph ever conflicts in a merge, rebuild rather than edit it: `rm graphify-out/graph.json && PYTHONHASHSEED=0 graphify update . --force && git add graphify-out/graph.json`. Never resolve it by picking a side; the file is generated, so the answer is always to rebuild it from the merged code.
- A changed `graph.json` in a pull request is generated output. It is one line, marked `linguist-generated`, collapsed by GitHub and kept out of `git diff`, and it is not reviewed.
- The graph is not byte-reproducible across operating systems: for the same source and pin, Linux extracts three fewer `imports_from` edges than macOS (node ids and clustering come out identical — it is edge extraction that differs, not the hashing or the clustering). So the rebuild is deliberately _not_ part of `nox -s lint` or CI — it would fail forever for everyone on the other platform. Your commit hook keeps your graph current; the weekly workflow is the drift check.
- A weekly `.github/workflows/graphify.yml` workflow rebuilds the graph and opens a pull request if it drifted, so a skipped hook does not leave it stale for long. It runs on Linux, so a run after a macOS contributor's commit may open a pull request that only normalises those three edges.

## Checklist for Contributed Code

Before being merged, a pull request for a new feature will be reviewed to see if it meets the following requirements. If you are unsure about how to meet all of these requirements, please submit the PR and ask for help and/or guidance. A `unxt` maintainer will collaborate with you to make sure that the pull request meets the requirements for inclusion in the package:

**Relevance:**

- Is the submission relevant to `unxt`?
- Does the code perform as expected?
- If applicable, are references included to the origin source for the algorithm?
- Has the code been tested against previously existing implementations?

**Code Quality:**

- Are the coding guidelines followed?
- Is the code compatible with the supported versions of Python?
- Are there dependencies other than the run-time dependencies listed in pyproject.toml?

**Testing:**

- Are the testing guidelines followed?
- Are the inputs to the functions sufficiently tested?
- Are there tests for any exceptions raised?
- Are there tests for the expected performance?
- Are the sources for the tests documented?
- Does `uv run nox -s test` run without failures?

**Documentation:**

- Is there a docstring in the function describing:
  - What the code does?
  - The format of the inputs of the function?
  - The format of the outputs of the function?
  - References to the original algorithms?
  - Any exceptions which are raised?
  - An example of running the code?
- Is there any information needed to be added to the docs to describe the function?
- Does the documentation build without errors or warnings?
