# unxts.interop.numpyro

NumPyro integration for [unxt](https://github.com/GalacticDynamics/unxt).

The `unitful` handler lets NumPyro models take `unxt` quantities. It strips the units into a unit system you choose before NumPyro sees them, gives the sampled values back to the model as quantities, and `attach_units` puts the units on the posterior samples.

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


M = unitful(model, u.unitsystems.galactic)
mcmc = MCMC(NUTS(M), num_warmup=200, num_samples=200)
mcmc.run(jax.random.PRNGKey(0), x)  # x is a Quantity of shape (200,)
samples = attach_units(mcmc.get_samples(), M)  # samples["m"] is a Quantity in kpc
```
