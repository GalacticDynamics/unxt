# Discussion

Background on why `unxt` works the way it does. These pages are for reading away from the keyboard — they discuss trade-offs and alternatives rather than giving instructions.

If you want to get something done, see the {doc}`../how-to/index`; if you want to look something up, see the {doc}`../reference/index`.

```{toctree}
:maxdepth: 1

sharp-bits
why-quantity-is-non-parametric
equality-and-equivalence
api-conventions
```

- {doc}`sharp-bits` — the behaviours that surprise people, and why each one is inherent rather than a bug.
- {doc}`why-quantity-is-non-parametric` — why the default `Quantity` stopped encoding dimension in its type, and why the usual `jit`-cache explanation is wrong.
- {doc}`equality-and-equivalence` — why `==` and `unxt.equivalent` answer different questions, and what that has to do with `jax.jit`.
- {doc}`api-conventions` — why every operation has both a functional and an object-oriented form, and why the operator argument comes first.
