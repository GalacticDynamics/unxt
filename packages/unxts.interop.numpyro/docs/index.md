# `unxts.interop.numpyro`

```{toctree}
:maxdepth: 1
:hidden:

tutorial-fit-mean-and-width
guide
api
```

`unxts.interop.numpyro` lets [NumPyro](https://num.pyro.ai/) models take `unxt` quantities. I wrap a model in the `unitful` handler, give its distributions `Quantity` parameters, and fit it with the usual NumPyro inference. The handler strips the units into a unit system I choose before NumPyro sees them, so NumPyro computes on plain numbers. The sampled values come back to the model as quantities, and `attach_units` puts the units back on the posterior samples.

## Install

The recommended install adds `unxts.interop.numpyro` alongside `unxt` via the `interop-numpyro` [extra](https://peps.python.org/pep-0508/#extras), so it, `unxt` and `numpyro` are resolved together as a compatible set:

::::{tab-set}

:::{tab-item} uv

```bash
uv add "unxt[interop-numpyro]"
```

:::

:::{tab-item} pip

```bash
pip install "unxt[interop-numpyro]"
```

:::

::::

Or install the package directly:

::::{tab-set}

:::{tab-item} uv

```bash
uv add unxts.interop.numpyro
```

:::

:::{tab-item} pip

```bash
pip install unxts.interop.numpyro
```

:::

::::

On import the package patches two NumPyro bugs ([#2428](https://github.com/pyro-ppl/numpyro/issues/2428) and [#2429](https://github.com/pyro-ppl/numpyro/issues/2429)) on the versions that lack the fixes. The patched functions behave like the originals except for a pytree that holds a tracer, such as a `Quantity` under `jit`, and an array-like that has its own `reshape`, such as a `Quantity`.

## At a glance

```python
>>> import jax
>>> import numpyro
>>> import numpyro.distributions as dist
>>> import unxt as u
>>> from unxts.interop.numpyro import unitful

>>> def model():
...     return numpyro.sample("m", dist.Normal(u.Q(0.0, "pc"), u.Q(3.0, "pc")))

>>> m = numpyro.handlers.seed(unitful(model, u.unitsystems.galactic), 0)()
>>> m.unit
Unit("kpc")

```

I wrote the distribution in parsecs, and the sampled value comes back in kiloparsecs because I gave `unitful` the galactic unit system.

## Pages

**Tutorial**

- [Fit a mean and a width to data with units](tutorial-fit-mean-and-width) — start here: run NUTS on a `Quantity` data set and get the posterior back with units.

**How-to**

- [How to use unitful in a model](guide) — where the handler goes among other handlers, `scan` and `cond`, subsampling, `jnp` code, and what the limits are.

**Reference**

- [API](api) — `unitful`, `attach_units` and `UnitfulError`.

## See also

- [NumPyro documentation](https://num.pyro.ai/)
- [unxt documentation](https://unxt.readthedocs.io/)
