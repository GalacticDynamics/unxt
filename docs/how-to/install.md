# How to install unxt

`unxt` is on PyPI and supports the Python versions listed on its [PyPI page](https://pypi.org/project/unxt/).

::::{tab-set}

:::{tab-item} pip

```bash
pip install unxt
```

:::

:::{tab-item} uv

```bash
uv add unxt
```

:::

:::{tab-item} from GitHub

```bash
pip install git+https://github.com/GalacticDynamics/unxt.git
```

:::

:::{tab-item} from a clone

```bash
git clone https://github.com/GalacticDynamics/unxt.git
cd unxt
pip install -e .  # editable mode
```

:::

::::

Installing `unxt` brings in JAX, [`quax`](https://github.com/patrick-kidger/quax), [`quaxed`](https://quaxed.readthedocs.io/en/latest/) and [`astropy`](https://www.astropy.org/), which is the unit backend.

## Install an add-on package

The `unxts.*` packages are separate distributions, installed only if you want them:

| Package | Adds |
| --- | --- |
| `unxts.parametric` | `ParametricQuantity` — dimension in the type, runtime dimension checking |
| `unxts.linalg` | Unitful linear algebra |
| `unxts.hypothesis` | `hypothesis` strategies for property-based testing |
| `unxts.interop.gala` | Interoperability with `gala` |
| `unxts.interop.matplotlib` | Plotting quantities with `matplotlib` |
| `unxts.interop.xarray` | Interoperability with `xarray` |

```bash
pip install unxts.parametric
```

Each has its own documentation under **Packages** in the sidebar.

## Check it worked

```{code-block} python
>>> import unxt as u
>>> u.Q(1.0, "m")
Quantity(Array(1., dtype=float32...), unit='m')
```

If that prints a quantity, you are ready for {doc}`../tutorials/first-quantity`.

## See also

- {doc}`../tutorials/first-quantity` — the first lesson.
- {doc}`migrate-to-v2` — upgrading from `unxt` v1.
- {doc}`../about/contributing` — installing for development.
