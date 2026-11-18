# Distribution unit rules

`unitful` finds the dimension of a sample site from the parameters of its distribution. This page lists which parameters carry the dimension, family by family, and how the other parameters are treated.

## Families with a rule

A parameter with power 1 has the site's dimension. A parameter with power −1 has its inverse.

| Family | Parameters that carry the dimension | Power |
| --- | --- | --- |
| `Normal`, `Cauchy`, `StudentT`, `Laplace`, `Logistic`, `Gumbel` | `loc`, `scale` | 1 |
| `HalfNormal`, `HalfCauchy` | `scale` | 1 |
| `Uniform` | `low`, `high` | 1 |
| `Exponential`, `Gamma` | `rate` | −1 |
| `Delta` | `v` | 1 |
| `VonMises` | `loc` | 1, always in radians |
| `Chi2` | none | — |

`TruncatedNormal` and the other truncated distributions add their bounds to the rule of the base distribution: `low` and `high` for a two-sided truncation, `low` for a left truncation, `high` for a right one. `TruncatedNormal(loc, scale, high=u.Q(2.0, "kpc"))`, with `low` left out, is a right truncation.

A `VonMises` site has to be an angle. It is stripped and wrapped in radians whatever the angle unit of `usys` is, because its density is defined on radians.

## Wrappers and mixtures

`unitful` follows these down to the distribution that has the parameters:

- `Independent`, `ExpandedDistribution` and `MaskedDistribution`, through `base_dist`.
- The truncated distributions, through `base_dist` plus their bounds.
- `MixtureSameFamily`, through `component_distribution`.
- A `TransformedDistribution` with a single `AffineTransform` over a base with bare parameters. The transform's `loc` and `scale` carry the dimension.

Any other `TransformedDistribution` has no rule.

## Parameters that carry the dimension

All the carrying parameters of a site have to agree on the dimension, to their powers. A site whose carrying parameters disagree raises `UnitfulError`.

A carrying parameter can be bare only if every carrying parameter is bare, so the site has no unit. The exceptions are a concrete zero or infinity, such as `0.0`, `np.zeros(3)` or `jnp.inf`. `Normal(0.0, u.Q(3.0, "kpc"))` is a length site. A `jnp.zeros(3)` built inside the model is a tracer under `jit` and NUTS, so it counts as bare there.

A site whose carrying parameters are dimensionless `Quantity` objects is a dimensionless site. Its latent comes back as a `Quantity` with unit `""`, or bare with `rewrap=False`. Bare parameters and a bare observed value there are read as plain numbers, so `dist.Normal(a + b * (r / u.Q(1.0, "kpc")), s)` with bare `a`, `b`, `s` and bare data is a dimensionless site.

When every carrying parameter of a site that is a `Quantity` is a `u.Angle`, the latent comes back to the model as a `u.Angle`. `attach_units` still gives a plain `Quantity`.

## Shape parameters

Parameters outside the table, such as `df` of a `StudentT`, `concentration` of a `Gamma` or a `VonMises`, and mixing probabilities, carry no dimension. A dimensionless `Quantity` there is stripped. A `Quantity` with a dimension raises `UnitfulError`.

Some NumPyro constructors reject a `Quantity` before `unitful` sees it:

| Constructor | Rejects | What to pass |
| --- | --- | --- |
| `StudentT`, `Beta`, `Categorical`, `Dirichlet` | a `Quantity` shape parameter, even a dimensionless one | a bare number, as in `dist.StudentT(3.0, loc, scale)` |
| `Pareto` | any `Quantity` parameter | bare parameters in the unit of `usys`, plus {ref}`the units= keyword <numpyro-the-units-keyword>` |

## Families with no rule

A family that is not in the table above, such as `LogNormal`, `ImproperUniform`, `MixtureGeneral`, the censored distributions, or a `TransformedDistribution` with a `ComposeTransform`, has no rule:

- With only bare or dimensionless parameters, the site has no unit. A dimensionless `Quantity` is stripped, such as the logits of a `Bernoulli` built from `beta * x`.
- With a `Quantity` that has a dimension anywhere in it, the site raises `UnitfulError`.
- With {ref}`the units= keyword <numpyro-the-units-keyword>`, the site has the given dimension and its bare parameters are read in the unit of `usys`.

The `censored=` argument of a censored distribution is a boolean mask, and NumPyro's constructor rejects a `Quantity` there. [How to give units to a site whose distribution has no unit rule](how-to-give-a-site-units) has a censored example.

## Out of scope

Distributions of the `MultivariateNormal` family with `Quantity` parameters.
