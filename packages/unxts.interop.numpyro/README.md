# unxts.interop.numpyro

NumPyro integration for [unxt](https://github.com/GalacticDynamics/unxt).

The `unitful` handler lets NumPyro models take `unxt` quantities. It converts them to plain numbers in a unit system you choose before NumPyro sees them. The model gets its sampled values back as quantities, and `attach_units` puts the units on the posterior samples.

## Install

```bash
pip install unxts.interop.numpyro
```

## Usage

```python
import jax
import numpyro
import numpyro.distributions as dist
import unxt as u
from numpyro.infer import MCMC, NUTS
from unxts.interop.numpyro import attach_units, unitful


def model(x=None):
    m = numpyro.sample("m", dist.Normal(u.Q(0.0, "kpc"), u.Q(3.0, "kpc")))
    s = numpyro.sample("s", dist.HalfNormal(u.Q(1.0, "kpc")))
    with numpyro.plate("n", 200):
        numpyro.sample("x", dist.Normal(m, s), obs=x)


x = u.Q(1.0 + 0.2 * jax.random.normal(jax.random.PRNGKey(1), (200,)), "kpc")
M = unitful(model, u.unitsystems.galactic)
mcmc = MCMC(NUTS(M), num_warmup=200, num_samples=200, progress_bar=False)
mcmc.run(jax.random.PRNGKey(0), x)
samples = attach_units(mcmc.get_samples(), M)  # samples["m"] is a Quantity in kpc
```
