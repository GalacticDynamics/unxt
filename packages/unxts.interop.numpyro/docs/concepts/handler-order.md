# Handler order

NumPyro passes each site's message through a stack of handlers. The message goes from the innermost handler, nearest the model, to the outermost. The results then come back from the outermost handler inwards. `unitful` strips the units on the way out and wraps the value on the way back. Handlers inside `unitful` therefore see quantities on the way out. Handlers outside it see {term}`bare values <bare value>`.

## What each handler needs

That one fact decides where each handler goes.

- `reparam` rebuilds a distribution from its parameters with plain arithmetic. It needs bare numbers, so it goes outside `unitful`.
- `condition` and `substitute` set a site's value. Inside `unitful`, `unitful` strips the value like any other, so it must be a `Quantity` when the site has a dimension. Outside, the site is already bare, so a bare number in the unit of `usys` also works.
- `do` sends a copy of the site through the stack a second time. Outside `unitful`, that copy has already been stripped. `unitful` refuses to strip it twice, so `do` goes inside.
- `lift` turns a `param` site into a sample site with a prior. Outside `unitful`, this happens after `unitful` has handled the site as a `param`. The prior's parameters are never stripped, so they must be bare.
- `trace` records each message on the way back. Inside `unitful`, it would record the stripped distribution next to the rewrapped value, which do not match. It goes outside.

A `Quantity` set by a handler outside `unitful` still reaches NumPyro as a bare value. `unitful` places an extra step at the outer end of the stack, which strips it.

## Guides and replay

SVI runs a guide, then {term}`replays <replay>` the guide's sampled values into the model. The replayed values are bare. The model's handler reads them in its own unit system. The guide must therefore produce them in the same unit system.

Suppose the guide's handler used parsecs and the model's used kiloparsecs. Every replayed value would then be wrong by a factor of 1000, and NumPyro would raise no error. `unitful` checks the unit of each replayed site and raises `UnitfulError` instead. A guide handler built as `unitful(guide, M.usys)` shares the model's unit system by construction.

## Two handlers on one model

A `unitful` inside another `unitful` strips the sites first. The outer handler then sees only bare values and records no units. With a different unit system, nothing converts between the two. One `unitful` per model avoids both problems.

## Related pages

- {doc}`../guides/place-handlers` shows where to put each handler.
- {doc}`../reference/behaviour` has the full handler table.
- {doc}`../guides/fit-with-svi` shows how to build a guide's handler.
