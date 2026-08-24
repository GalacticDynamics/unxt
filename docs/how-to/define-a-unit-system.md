# How to define a unit system

`unxt` ships realizations for SI, CGS, galactic, solar-system and the natural unit systems, and {func}`unxt.unitsystem` will build one on the fly from any set of units. You only need a hand-written subclass when you want a **named, statically defined** system — one you can annotate against, dispatch on, or ship from your own package.

If you just need a system for a set of units, call `unitsystem` instead:

```{code-block} python
>>> import unxt as u
>>> u.unitsystem("m", "s")
LengthTimeUnitSystem(length=Unit("m"), time=Unit("s"))
```

## Write the subclass

Declare one `Annotated` field per base dimension, pairing the unit type with the dimension it measures. The class must be a frozen dataclass and must be registered as a static JAX pytree node, exactly as the built-in systems are:

```{code-block} python
>>> from dataclasses import dataclass
>>> from typing import Annotated
>>> import jax.tree_util as jtu
>>> from astropy.units import UnitBase, get_physical_type
>>> from unxt.unitsystems import AbstractUnitSystem

>>> @jtu.register_static
... @dataclass(frozen=True, slots=True)
... class MyUSys(AbstractUnitSystem):
...     energy: Annotated[UnitBase, get_physical_type("energy")]
...     frequency: Annotated[UnitBase, get_physical_type("frequency")]

```

## Construct it

A hand-built subclass stores its fields as given, so pass unit _objects_. String parsing is a service of the `unitsystem` factory, not of the dataclass.

```{code-block} python
>>> usys = MyUSys(u.unit("erg"), u.unit("Hz"))
>>> usys["energy"]
Unit("erg")

```

Indexing works the same as for any built-in system, including derived dimensions composed from the base units.

## See also

- {doc}`../reference/unitsystems` — the built-in realizations and the full `unitsystem` input list.
- {doc}`../explanation/api-conventions` — why `unxt` pairs a functional and an object-oriented API.
