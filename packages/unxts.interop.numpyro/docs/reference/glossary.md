# Glossary

Each entry defines a term as these pages and the error messages use it. The core [`unxt` glossary](../../../reference/glossary) defines units, dimensions and unit systems.

```{glossary}
:sorted:

bare value
  A plain number or array, with no unit. NumPyro computes on bare values. `unitful` reads a bare value at a site with a dimension in the {term}`site unit`.

carrying parameter
  A distribution parameter that has the site's dimension, or a power of it. For `Normal`, the carrying parameters are `loc` and `scale`. The {term}`unit rule` of a family names them.

latent site
  A {term}`sample site` without an observed value. NumPyro samples its value. Also called a latent.

observed site
  A {term}`sample site` with an `obs=` value. NumPyro scores the value instead of sampling it.

replay
  Running a model with values taken from an earlier trace. SVI replays a guide's samples into the model.

sample site
  A call to `numpyro.sample` in a model. Each sample site has a name, a distribution and a value.

shape parameter
  A distribution parameter that is not a {term}`carrying parameter`, such as `df` of a `StudentT`. It has no dimension.

site unit
  The unit `unitful` uses for a site: the `usys` unit of the site's dimension, or radians for a `VonMises` site. `handler.site_units` lists them.

unit rule
  The mapping from a distribution family to its {term}`carrying parameters <carrying parameter>` and their powers. The rules are listed on the {doc}`unit-rules` page.
```
