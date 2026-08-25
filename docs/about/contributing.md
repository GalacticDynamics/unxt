# Contributing

```{include} ../../CONTRIBUTING.md
:start-line: 1
```

## Building the documentation

Documentation is built with `nox`:

```bash
nox -s docs
```

To rebuild and serve with live reloading:

```bash
nox -s docs -- --serve
```

To check external links:

```bash
nox -s docs -- -b linkcheck
```

## Which kind of page am I writing?

The documentation follows [Diátaxis](https://diataxis.fr), which sorts every page into one of four kinds. Two questions decide it:

| The content…      | …serving the reader's… | …belongs in    |
| ----------------- | ---------------------- | -------------- |
| informs action    | acquisition of skill   | `tutorials/`   |
| informs action    | application of skill   | `how-to/`      |
| informs cognition | application of skill   | `reference/`   |
| informs cognition | acquisition of skill   | `explanation/` |

Practically:

- **`tutorials/`** — a lesson a newcomer can complete start to finish, with no choices and no prerequisites beyond installing `unxt`. Tutorials must never fail.
- **`how-to/`** — one guide, one real task, title starting with "How to". Assume competence; link to reference rather than listing every option.
- **`reference/`** — description only. Structure mirrors the code. No instructions, no rationale. Each fact lives in exactly one table row; everywhere else links to it.
- **`explanation/`** — answers a _why_. Discusses alternatives and trade-offs. No numbered procedures, no exhaustive tables.
- **`about/`** — material that is not practitioner documentation, like this page.

If a page resists classification it is usually two pages fused together. Split it rather than inventing a fifth section.

Every Python block in a `.md` file is executed by [Sybil](https://sybil.readthedocs.io/) as part of the test suite, so a block must run and produce exactly the output shown. When moving a block between pages, carry its imports with it.
