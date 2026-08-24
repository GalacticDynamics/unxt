# How to teach JAX a new primitive rule

`Quantity` works inside JAX because [`quax`](https://docs.kidger.site/quax/) lets you register a handler for a JAX **primitive** — the low-level operations (`lax.add_p`, `lax.mul_p`, `lax.dot_general_p`, …) that every `jax.numpy` function eventually lowers to. `unxt` ships several hundred such rules; that is the whole mechanism by which units survive a `quaxify`.

You only need to write one yourself in two situations:

- **You have your own array-like type** and want it to flow through JAX the way `Quantity` does.
- **You hit a primitive `unxt` does not cover.** In that case please [open an issue](https://github.com/GalacticDynamics/unxt/issues) as well — a gap in `unxt`'s coverage is a bug, not something you should have to patch around.

If instead you want `unxt`'s _functions_ (`unit_of`, `uconvert`, `ustrip`) to work on a type of your own, that is a different and much smaller job — see {doc}`Teach unxt about your own type <../packages/unxts.api/tutorial-your-own-type>`.

## The shape of a rule

A rule is a function decorated with `quax.register`, taking the primitive's operands and returning the result:

```python
import quax
from jax import lax

from unxt.quantity import AbstractQuantity


@quax.register(lax.abs_p)
def abs_p_q(x: AbstractQuantity, /) -> AbstractQuantity:
    ...
```

Dispatch is on the annotated operand types, so one primitive can have many rules — `lax.mul_p` alone needs handlers for quantity-times-quantity, quantity-times-array, and array-times-quantity, each computing a different unit for the result.

## Read the real reference

`quax` is where this is documented, and it is worth reading before writing a rule: the [quax documentation](https://docs.kidger.site/quax/) covers `quax.register`, the `ArrayValue` type your custom type would subclass, and how `quaxify` traces through a function to reach your rules.

## Read unxt's rules as examples

`unxt`'s own registrations are the largest worked example available, and they are grouped by package:

| Module | Covers |
| --- | --- |
| `unxt/_src/quantity/register_primitives.py` | the core `Quantity` rules |
| `unxts.parametric/_src/register_primitives.py` | rules that fire only for a parametric operand |
| `unxts.linalg/_src/_register_primitives.py` | `QuantityMatrix`, including `dot_general` and `gather` |

These are private modules. They are excellent to read and are **not** a stable interface — import from them and your code will break.

## See also

- [quax documentation](https://docs.kidger.site/quax/) — the actual reference for this extension point.
- {doc}`use-jax-functions` — using the rules that already exist.
- {doc}`../explanation/sharp-bits` — where primitive-level behaviour surprises people, such as `jnp.where` letting a raw array adopt a unit.
