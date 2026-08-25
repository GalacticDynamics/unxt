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

Installing `unxt` brings in JAX, [`quax`](https://github.com/patrick-kidger/quax), [`quaxed`](https://github.com/GalacticDynamics/quaxed) and [`astropy`](https://www.astropy.org/), which is the unit backend.

## Install an add-on package

The `unxts.*` packages are separate distributions. Install each through the matching **extra** rather than by name, so that it and `unxt` are resolved together as a compatible set:

| Extra | Package | Adds |
| --- | --- | --- |
| `unxt[parametric]` | `unxts.parametric` | `ParametricQuantity` — dimension in the type, runtime dimension checking |
| `unxt[linalg]` | `unxts.linalg` | `QuantityMatrix` — per-element units for Jacobians and metrics |
| `unxt[interop-gala]` | `unxts.interop.gala` | Unit-system conversion with `gala` |
| `unxt[interop-mpl]` | `unxts.interop.matplotlib` | Plotting quantities with `matplotlib` |
| `unxt[interop-xarray]` | `unxts.interop.xarray` | A `.unxt` accessor for `DataArray` and `Dataset` |

```bash
pip install "unxt[parametric]"
```

Two packages have no extra of their own — install them by name:

| Package | Adds |
| --- | --- |
| `unxts.hypothesis` | `hypothesis` strategies for property-based testing |
| `unxts.api` | The abstract dispatch API. Already a dependency of `unxt`; install it directly only if you are writing against the API _without_ `unxt` |

```bash
pip install unxts.hypothesis
```

### Everything at once

`unxt[workspace]` installs every package above, `unxts.hypothesis` and `unxts.api` included. `unxt[all]` is `workspace` plus `backend-astropy`, which pins the `astropy` unit backend explicitly:

```bash
pip install "unxt[all]"
```

Each package has its own documentation under **Packages** in the sidebar.

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
