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

`unxt` is wired up for [graphify](https://github.com/Graphify-Labs/graphify), which turns the source into a queryable knowledge graph. Extraction is local tree-sitter AST parsing: no LLM, and nothing leaves your machine. Querying the graph is usually cheaper and more accurate than grepping.

```bash
nox -s graph                 # build graphify-out/
nox -s graph -- --no-label   # ... without the LLM pass that names communities
```

That writes `graphify-out/graph.html` (clickable graph), `graphify-out/GRAPH_REPORT.md` (hubs, communities, suggested questions) and `graphify-out/graph.json` (the graph itself). The whole directory is generated and gitignored.

Query it through `uv run`, the same as any other tool in this repo (`--no-project` keeps it out of the project env; `uv tool install graphifyy` instead if you use it often enough to want it on `PATH`):

```bash
alias gq='uv run --no-project --with graphifyy graphify'

gq god-nodes                                  # the architectural hubs
gq explain "AbstractQuantity"                 # one symbol and its neighbours
gq path "Quantity" "AbstractUnitSystem"       # how two things connect
gq query "how are units converted"            # a scoped subgraph for a question
gq update .                                   # refresh after code changes, no LLM
```

Every edge is tagged `EXTRACTED` (explicit in the source) or `INFERRED` (resolved by graphify), so you can tell what was read directly from what was guessed.

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
