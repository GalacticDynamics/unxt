# Why an abstract dispatch API

`unxts.api` contains almost no code. It declares a dozen functions with {func}`plum.dispatch.abstract` and implements none of them; `unxt` supplies the bodies. Splitting a package in half like that is unusual enough to be worth justifying.

## The dependency argument

`unxt` depends on JAX, NumPy and astropy. That is the right cost for a library whose job is unitful JAX arrays, and the wrong cost for a library that merely wants to _interoperate_ with one.

Consider a package defining its own physical-quantity type. It would like `unit_of(my_thing)` to work, so that code written against `unxt`'s API accepts its type too. Without the split, that means depending on `unxt` — and therefore on JAX — to gain an interface it will never call into. With the split, it depends on `unxts.api` and `plum`, and its users decide whether `unxt` is also installed.

That is the concrete win, and it is why the boundary is drawn where it is: at the point where "what functions exist and what they mean" separates from "how they work on JAX arrays".

## Why dispatch rather than an ABC

The conventional way to express "here is an interface, implement it" is an abstract base class. Multiple dispatch is a better fit here for one reason: the types being extended are usually **not yours to subclass**.

`unxt` supports astropy quantities, and — through the `unxts.interop.*` packages — `gala` unit systems and `xarray` objects. None of those can be made to inherit from an `unxt` base class; they belong to other projects. With dispatch, support is added from the outside, by registering a function on a type you do not own. The same mechanism serves third parties adding support for _their_ types without anyone modifying `unxt`.

A second, smaller benefit: dispatch selects on _all_ arguments, not just the first. `uconvert_value(to_unit, from_unit, value)` genuinely varies with the combination of unit types, and a method on one receiver would have to re-dispatch internally on the others.

## What it costs

The separation is not free, and it is honest to name the costs:

- **Errors move to call time.** A missing implementation surfaces as `plum.NotFoundLookupError` when the call happens, not as an unimplemented abstract method at class definition. A static type checker will not catch it.
- **The contract lives in prose.** An abstract signature annotated `Any -> Any` documents very little; what `dimension_of` is _supposed_ to return for your type is stated in the [reference](api) rather than enforced.
- **Registration is a side effect.** Implementations arrive when a module is imported. That makes import order matter in a way it would not with explicit registration, and it is why `unxts.parametric` documents that you must import it for its primitive rules to exist.

These are real, and they are the reason the API package stays small: the wider the abstract surface, the more contract there is to keep honest in prose.

## See also

- [How to extend unxt with your own types](extending) — the mechanics.
- [API](api) — what each abstract function promises.
- [About unxt's API conventions](../../explanation/api-conventions) — why `unxt` pairs a functional and an object-oriented form for each of these.
