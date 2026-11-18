# Guides

Each guide covers one task. The {doc}`tutorial <../tutorial-fit-mean-and-width>` shows a whole fit, and the {doc}`concepts <../concepts/index>` explain why `unitful` works as it does.

A fit with `unitful` has four steps:

1. **Write the model in units.** Give distributions `Quantity` parameters and pass `Quantity` data. If a family has no unit rule, name its dimension ({doc}`give-a-site-units`). For computations between sites, see {doc}`use-jnp-code`. For loops and branches, see {doc}`use-scan-and-cond`. For partial data, see {doc}`mask-and-subsample`.
2. **Wrap it and add handlers.** Wrap the model in `unitful` with a unit system. Then put each other handler on the right side of it ({doc}`place-handlers`).
3. **Run inference.** Use MCMC ({doc}`run-mcmc`) or SVI ({doc}`fit-with-svi`).
4. **Read the results.** Pass the samples and the handler to `attach_units` ({doc}`run-mcmc`, {doc}`fit-with-svi`).

{doc}`troubleshooting` maps each error and symptom to its fix.

```{toctree}
:maxdepth: 1

give-a-site-units
use-jnp-code
use-scan-and-cond
mask-and-subsample
place-handlers
run-mcmc
fit-with-svi
troubleshooting
```
