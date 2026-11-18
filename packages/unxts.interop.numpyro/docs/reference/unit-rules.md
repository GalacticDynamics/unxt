# Distribution unit rules

`unitful` finds a site's dimension from the parameters of its distribution. A family's {term}`unit rule` names its {term}`carrying parameters <carrying parameter>` and their powers. A parameter with power 1 has the site's dimension. A parameter with power −1 has its inverse.

## Families with a unit rule

| Family | Carrying parameters | Power |
| --- | --- | --- |
| `Normal`, `Cauchy`, `StudentT`, `Laplace`, `Logistic`, `Gumbel` | `loc`, `scale` | 1 |
| `HalfNormal`, `HalfCauchy` | `scale` | 1 |
| `Uniform` | `low`, `high` | 1 |
| `Exponential`, `Gamma` | `rate` | −1 |
| `Delta` | `v` | 1 |
| `VonMises` | `loc` | 1 |
| `Chi2` | none | — |

A truncated distribution adds its bounds to the rule of its base distribution:

| Truncation | Added carrying parameters |
| ---------- | ------------------------- |
| two-sided  | `low`, `high`             |
| left       | `low`                     |
| right      | `high`                    |

`TruncatedNormal(loc, scale, high=u.Q(2.0, "kpc"))` has no `low`, so it is a right truncation.

A `VonMises` site must be an angle. Its {term}`site unit` is always radians.

## Wrappers and mixtures

`unitful` follows these to the distribution that holds the parameters:

| Wrapper | Followed through |
| --- | --- |
| `Independent`, `ExpandedDistribution`, `MaskedDistribution` | `base_dist` |
| truncated distributions | `base_dist`, plus the bounds |
| `MixtureSameFamily` | `component_distribution` |
| `TransformedDistribution` with one `AffineTransform` over a base with bare parameters | the transform's `loc` and `scale`, which carry the dimension |

Any other `TransformedDistribution` has no unit rule.

## Carrying parameters

The carrying parameters of a site must agree on the dimension, after their powers.

A carrying parameter can be a {term}`bare value` only if all of them are bare. The site then has no unit. A concrete zero or infinity counts as unit-free, so `Normal(0.0, u.Q(3.0, "kpc"))` is a length site. Concrete means `0.0`, `np.zeros(3)` or `jnp.inf`. A `jnp.zeros(3)` built inside the model is a tracer under `jit` and NUTS, so it counts as bare there.

A site whose carrying parameters are dimensionless quantities is a dimensionless site. Its latent value is a `Quantity` with unit `""`, or bare with `rewrap=False`. Its bare parameters and bare observed values are plain numbers. For example, `dist.Normal(a + b * (r / u.Q(1.0, "kpc")), s)` is a dimensionless site when `a`, `b`, `s` and the data are bare.

If every carrying parameter that is a `Quantity` is a `u.Angle`, the model gets the latent value as a `u.Angle`. `attach_units` still returns a plain `Quantity`.

## Shape parameters

A {term}`shape parameter` has no dimension. Examples are `df` of a `StudentT`, `concentration` of a `Gamma` or a `VonMises`, and mixing probabilities. `unitful` strips a dimensionless `Quantity` there, and rejects one with a dimension.

Some NumPyro constructors reject a `Quantity` before `unitful` sees it:

| Constructor | Rejects | Pass instead |
| --- | --- | --- |
| `StudentT`, `Beta`, `Categorical`, `Dirichlet` | a `Quantity` shape parameter, even a dimensionless one | a bare number, as in `dist.StudentT(3.0, loc, scale)` |
| `Pareto` | any `Quantity` parameter | bare parameters in the unit of `usys`, with {ref}`units= <numpyro-the-units-keyword>` |

## Families with no unit rule

Every family not listed above has no unit rule. Examples are `LogNormal`, `ImproperUniform`, `MixtureGeneral`, the censored distributions, and a `TransformedDistribution` with a `ComposeTransform`.

| Parameters | Result |
| --- | --- |
| bare or dimensionless only | The site has no unit. Dimensionless quantities are stripped, such as the logits of a `Bernoulli` built from `beta * x`. |
| any `Quantity` with a dimension | `UnitfulError`. |
| bare, with a {ref}`units= <numpyro-the-units-keyword>` entry | The site has the given dimension. Its parameters are read in the unit of `usys`. |

A censored distribution's `censored=` argument is a boolean mask. NumPyro's constructor rejects a `Quantity` there. {doc}`../guides/give-a-site-units` has a censored example.

## Out of scope

`unitful` does not support `MultivariateNormal`-family distributions with `Quantity` parameters.
