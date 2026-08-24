# About

Material about the `unxt` project itself, rather than about using it.

```{toctree}
:maxdepth: 1
:hidden:

contributing
```

[Contributing](contributing) covers reporting issues, contributing code and documentation, and building the docs.

## Citation

[![JOSS](https://joss.theoj.org/papers/10.21105/joss.07771/status.svg)](https://doi.org/10.21105/joss.07771) [![DOI](https://zenodo.org/badge/734877295.svg)](https://zenodo.org/doi/10.5281/zenodo.10850455)

If `unxt` was useful in your work and you want to support the development and maintenance of lower-level libraries for the scientific community, please consider citing it.

## Ecosystem

`unxt` builds on:

- [Equinox](https://docs.kidger.site/equinox/) — one-stop JAX library, for everything not already in core JAX.
- [Quax](https://github.com/patrick-kidger/quax) — JAX + multiple dispatch + custom array-ish objects.
- [Quaxed](https://quaxed.readthedocs.io/en/latest/) — pre-`quaxify`ed JAX.
- [plum](https://pypi.org/project/plum-dispatch/) — multiple dispatch in Python.
- [unxts.api](https://pypi.org/project/unxts.api/) — the abstract dispatch API for `unxt`.

And is built on by:

- [unxts.hypothesis](https://pypi.org/project/unxts.hypothesis/) — `hypothesis` strategies for `unxt`.
- [coordinax](https://github.com/GalacticDynamics/coordinax) — coordinates in JAX.
- [galax](https://github.com/GalacticDynamics/galax) — galactic dynamics in JAX.
