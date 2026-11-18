# Unit systems

A `Quantity` becomes a number only once a unit is chosen. The same length is 1 in kiloparsecs and about 3.1 × 10¹⁹ in metres. `unitful` takes that choice from its unit system argument, `usys`, which names one unit for each dimension.

## The choice changes the log density

A probability density over a length has the units of one over that length. Expressing the same data in another unit rescales the density by the conversion factor. Its logarithm shifts by a constant.

The constant does not change the posterior, because it does not depend on the parameters. It does change every reported log density. A log-likelihood value therefore compares only with another computed under the same unit system. {doc}`../reference/behaviour` shows the shift between kiloparsecs and metres.

`usys` is a required argument for this reason. The model's code then states the unit of every log density it reports. The units of the incoming data do not decide it.

## Bare numbers are read in the unit system

Some numbers reach a site that has a dimension, but carry no unit themselves. A handler outside `unitful` can set such a site to a bare value. A family with no unit rule has bare parameters, and a `units=` entry gives its site a dimension. `unitful` reads these numbers in the unit of `usys`. A site with no dimension has no unit, so its numbers stay plain.

This is why a `units=` entry must name the dimension, or the `usys` unit itself. Under the `galactic` unit system, bare parameters are read in kiloparsecs. An entry of `"pc"` would claim parsecs for numbers that are really kiloparsecs, so `unitful` rejects it.

## One exception: `VonMises`

NumPyro defines the `VonMises` density on radians. `unitful` therefore strips and wraps a `VonMises` site in radians, whatever angle unit `usys` uses. The site's log density is then the one NumPyro defines.

## Related pages

- {doc}`sites` explains how `unitful` strips and wraps each site.
- {doc}`handler-order` explains where a bare value can come from.
- {doc}`../guides/give-a-site-units` shows how to use `units=`.
