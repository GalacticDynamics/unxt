# Errors

`unitful` and `attach_units` raise `UnitfulError`, a subclass of `ValueError`, when a site's units are missing, inconsistent or unsupported. Most messages start with the name of the site. This page lists every message, grouped by what it is about. Text in angle brackets, such as `<name>`, stands for a value filled in at run time.

## Distribution parameters

<!-- prettier-ignore-start -->
`Site '<name>': parameter '<p>' is bare next to Quantity parameters; give it units.`
: **When.** One {term}`carrying parameter` is a `Quantity` and another is a {term}`bare value`.
  **Cause.** The bare parameter has no unit, so `unitful` cannot strip it with the others.
  **Fix.** Give it units. A concrete zero or infinity, such as `0.0`, is allowed bare.

The message adds a sentence when the bare parameter is a JAX tracer, such as `jnp.zeros(3)` under `jit`. A tracer is never a concrete zero. Write `0.0` or `np.zeros(3)` instead. Inside a `scan` or `cond` body, a further sentence says that a value passed in bare stays bare.

`Site '<name>': <Family> has Quantity parameters, but the parameters that carry the site's dimension (<params>) are bare; give them units.`
: **When.** A {term}`shape parameter` has a dimension, and every carrying parameter is bare.
  **Fix.** Give the carrying parameters units, and make the shape parameter dimensionless.

`Site '<name>': <Family> has Quantity parameters with dimensions but no unit rule; give it bare parameters in the units of usys and pass units={'<name>': '<unit>'} to unitful.`
: **When.** A family with no {term}`unit rule` holds a `Quantity` with a dimension.
  **Fix.** Write its parameters as bare numbers in the unit of `usys`, and pass the suggested `units=` entry. For a discrete family, the message instead says that the parameters must be dimensionless.

`Site '<name>': parameters have different dimensions: <dimensions>.`
: **When.** The carrying parameters disagree, such as a `loc` in kpc and a `scale` in seconds.
  **Fix.** Give them the same dimension.

`Site '<name>': VonMises needs an angle, got <dimension>.`
: **When.** A `VonMises` `loc` is not an angle.
  **Fix.** Give `loc` an angle unit.

`Site '<name>': shape parameter '<p>' has dimension <dimension>; it must be dimensionless.`
: **When.** A shape parameter, such as `df` of a `StudentT`, has a dimension.
  **Fix.** Pass it as a bare number.

`Site '<name>': the unit system has no unit for <dimension>.`
: **When.** The site's dimension is missing from `usys`.
  **Fix.** Use a unit system with a unit for that dimension.
<!-- prettier-ignore-end -->

## Values

<!-- prettier-ignore-start -->
`Site '<name>': value is bare. A bare value would be read in <unit>, the unit of usys; wrap the data in its own unit, as in u.Q(value, <your unit>).`
: **When.** An observed value, or a value set by a handler inside `unitful`, is bare at a site with a dimension.
  **Cause.** A bare value there would be read silently in the {term}`site unit`.
  **Fix.** Pass the value as a `Quantity`. If the bare value is already in the unit of `usys`, set it from a handler outside `unitful` instead.

The message adds a sentence when `unitful` is used as a `with` context. Inference handlers then run inside it and set bare values. Wrap the model instead, as in `log_density(unitful(model, usys), ...)`.

`Site '<name>': value has dimension <dimension>, expected <expected>.`
: **When.** A value's dimension differs from the site's dimension.
  **Fix.** Pass a value of the site's dimension.

`Site '<name>': has unit <unit> in one place and <unit> in another. A site needs one unit in every call and in both branches of a cond.`
: **When.** One site name gets two different site units. This happens when the two branches of a `cond` sample a site in different units, or when a site's unit changes between calls.
  **Fix.** Give the site one dimension everywhere it appears.

`Site '<name>': the value is a container that holds Quantities. unitful strips a value only when it is a Quantity itself; give each Quantity a site of its own, or pass one Quantity array.`
: **When.** A `deterministic`, `param` or observed value is a dict, list or other container of quantities.
  **Fix.** Give each `Quantity` its own site, or pass one `Quantity` array.

`Site '<name>': a Quantity value was set at a site with no unit.`
: **When.** A handler outside `unitful` sets a `Quantity` with a dimension at a site whose distribution has bare parameters. For a `param` site, a callable init that returns a `Quantity` raises it too, and the message says so. An `obs=` value or an inner handler's value raises the "value has dimension" message instead.
  **Fix.** Give the distribution `Quantity` parameters, or set a bare value.

`Site '<name>': a Quantity value set outside unitful reached the site unstripped. Put the handler that set it inside: unitful(handlers.condition(model, ...), usys).`
: **When.** A handler outside `unitful` set a `Quantity` value that `unitful` could not strip.
  **Fix.** Move that handler inside `unitful`.

`Site '<name>': <what> has dimension <dimension>, expected dimensionless.`
: **When.** A `handlers.scale` factor or a `factor` site's `log_factor` has a dimension.
  **Fix.** Pass a bare or dimensionless value.
<!-- prettier-ignore-end -->

## The `units=` keyword

<!-- prettier-ignore-start -->
`Site '<name>': units= takes a dimension or a unit, such as 'length' or 'kpc', got <value>.`
: **When.** A `units=` value is neither a dimension nor a unit.

`Site '<name>': units= gives <unit>, but bare parameters are read in the unit of usys; pass units={'<name>': '<usys unit>'}.`
: **When.** A `units=` value is a unit that is not the `usys` unit of its dimension, such as `"pc"` under `galactic`.
  **Fix.** Pass the dimension, or the suggested `usys` unit.

`Site '<name>': parameter '<p>' is bare, and <Family> has a unit rule; give its parameters units instead of using units=.`
: **When.** `units=` names a site whose family has a unit rule, and its carrying parameters are bare.
  **Fix.** Give the carrying parameters units and remove the `units=` entry.

`Site '<name>': units= gives <dimension>, the distribution gives <dimension>.`
: **When.** `units=` disagrees with the dimension from the family's unit rule.
  **Fix.** Remove the `units=` entry.

`Site '<name>': units= needs bare parameters, but the distribution has Quantity parameters with dimensions and no unit rule.`
: **When.** `units=` names a site whose family has no unit rule, and the distribution holds a `Quantity` with a dimension.
  **Fix.** Write the parameters as bare numbers in the unit of `usys`.

`Site '<name>': units= applies to sample sites, not <kind> sites.`
: **When.** `units=` names a `deterministic`, `param` or `factor` site.
  **Fix.** Remove the entry. `units=` does not apply to those sites.

`units= names [<names>], which no call of the model has reached yet.`
: **When.** A model call returns, and no call so far has reached a site named in `units=`.
  **Fix.** Check the spelling. A site behind an `if` stops raising once a call reaches it.

`units= names [<names>], which are site names after a handler outside unitful renamed them. Key units= by the name inside unitful, or put the scope inside unitful.`
: **When.** A `units=` key includes a prefix that a `scope` outside `unitful` added.
  **Fix.** Drop the prefix, or move the `scope` inside `unitful`.
<!-- prettier-ignore-end -->

## Handlers and replay

<!-- prettier-ignore-start -->
`Site '<name>': this unitful already stripped the site. ...`
: **When.** The same site reaches the same `unitful` twice.
  **Cause.** Either the model calls itself through the same `unitful` instance, or a handler outside, such as `handlers.do`, sends the site again.
  **Fix.** Make a new `unitful` for the inner call, or put the handler inside, as in `unitful(handlers.do(model, ...), usys)`.

`Site '<name>': the replayed value (for example from an SVI guide) was made under unitful with usys <usys>, but the model runs under <usys>. ...`
: **When.** A guide's handler and the model's handler use different unit systems.
  **Fix.** Build the guide's handler as `unitful(guide, model_handler.usys)`.

`Site '<name>': the replayed value (for example from an SVI guide) is in <unit>, but the model reads this site in <unit>. ...`
: **When.** A guide site and its model site have different units.
  **Fix.** Give the guide's site the model site's dimension. If one of the two is a `VonMises` site, which is always in radians, use a `VonMises` in both.
<!-- prettier-ignore-end -->

## `param` constraints

<!-- prettier-ignore-start -->
`Site '<name>': the constraint has Quantity bounds, but the init value is bare; pass the init value as a Quantity.`
: **When.** A `param` constraint has `Quantity` bounds and the init value is bare.
  **Fix.** Pass the init value as a `Quantity`.

`Site '<name>': a constraint bound has dimension <dimension>, expected <expected>.`
: **When.** A constraint bound's dimension differs from the init value's.
  **Fix.** Give the bound the init value's dimension.
<!-- prettier-ignore-end -->

## `attach_units`

<!-- prettier-ignore-start -->
`The handler has no recorded sites: run the model first.`
: **When.** `attach_units` gets a handler that has not run a model.
  **Fix.** Pass the handler that ran the inference.
<!-- prettier-ignore-end -->

## Other exceptions

`unitful` raises `TypeError` for a call without a model, `unitful(None, usys)`, and for a `usys` that is neither a unit system nor a string. A string that names no registered unit system raises `ValueError`.
