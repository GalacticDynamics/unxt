# Units

Units are measures of dimensions. Some common units are meters for length or
seconds for time or Joules for energy.

`unxt` has two primary functions for working with units: `unit` and `unit_of`.

```{code-block} python
>>> import unxt as u

>>> u.unit
<multiple-dispatch function unit (...)>

>>> u.unit_of
<multiple-dispatch function unit_of (...)>
```

The function `unit` is for creating a unit, while `unit_of` is for getting the
unit of an object.

First let's create some units:

```{code-block} python
>>> m = u.unit('m')  # from a str
>>> m
Unit("m")

>>> u.unit(m)  # from a unit object
Unit("m")

```

Now let's get the units from objects:

```{code-block} python
>>> print(u.unit_of("m"))  # str have no units
None

>>> u.unit_of(m)  # from a unit object
Unit("m")

>>> q = u.Q(5, 'm')  # from a Quantity
>>> u.unit_of(q)
Unit("m")

```

:::{seealso}

[API Documentation for Units](../api/units.md)

:::

<br><br><br>

# Unit Systems

A unit system is a standardized collection of units used together, such as the
International System of Units (SI), the Imperial system, and natural units like
Planck or atomic units. Each system defines base units (e.g., meters, kilograms)
and derived units (e.g., joules, newtons) for consistent expression of
quantities.

`unxt` provides a sub-module for working with unit systems: `unxt.unitsystems`.
unit systems are implemented as subclasses of the base class
`AbstractUnitSystem`, which can be used for statically defining unit systems.
However this rarely needs to be used directly. Instead, `unxt` provides some
built-in unit systems and their realizations, such as `galactic` and
`solarsystem`. Also, `unxt` has the functions `unitsystem` and `unitsystem_of`
for dynamically making the realization of a unit system or getting one from an
object.

:::{seealso}

[API Documentation for Unit Systems](../api/unitsystems.md)

:::

## `AbstractUnitSystem` class

:::{seealso}

This rarely needs to be used. See `unxt.unitsystem`.

:::

```{code-block} python
>>> from typing import Annotated
>>> from astropy.units import UnitBase, get_physical_type
>>> from unxt.unitsystems import AbstractUnitSystem


>>> class MyUSys(AbstractUnitSystem):
...     energy: Annotated[UnitBase, get_physical_type("energy")]
...     frequency: Annotated[UnitBase, get_physical_type("frequency")]
...     luminance: Annotated[UnitBase, get_physical_type("luminance")]

```

## Built-in Unit Systems

`unxt` provides several built-in unit systems.

```{code-block} python
>>> from unxt.unitsystems import si
>>> si
unitsystem(m, kg, s, mol, A, K, cd, rad)
```

```{code-block} python
>>> from unxt.unitsystems import cgs
>>> cgs
unitsystem(cm, g, s, dyn, erg, Ba, P, St, rad)
```

```{code-block} python
>>> from unxt.unitsystems import galactic
>>> galactic
unitsystem(kpc, Myr, solMass, rad)
```

```{code-block} python
>>> from unxt.unitsystems import solarsystem
>>> solarsystem
unitsystem(AU, yr, solMass, rad)
```

## Functions for Unit Systems

`unxt` has two primary functions for working with units: `unitsystem` and
`unitsystem_of`.

```{code-block} python
>>> from unxt.unitsystems import unitsystem, unitsystem_of
```

`unitsystem` can return named unit systems:

```{code-block} python
>>> unitsystem("si")
unitsystem(m, kg, s, mol, A, K, cd, rad)

>>> unitsystem("cgs")
unitsystem(cm, g, s, dyn, erg, Ba, P, St, rad)

>>> unitsystem("galactic")
unitsystem(kpc, Myr, solMass, rad)

>>> unitsystem("solarsystem")
unitsystem(AU, yr, solMass, rad)

```

Unit systems are statically defined, the "galactic" and "solarsystem" units are
instances of `LTMAUnitSystem` (length-time-mass-angle). If passed a set of units
with dimensions matching one of the pre-defined unit system classes `unitsystem`
will recognize this and return an instance of that unit system.

```{code-block} python

>>> from unxt.unitsystems import LTMAUnitSystem

>>> usys = unitsystem("kpc", "Myr", "solMass", "degree")
>>> usys
unitsystem(kpc, Myr, solMass, deg)

>>> isinstance(usys, LTMAUnitSystem)
True

>>> usys == unitsystem("galactic")
False

```

If the set of units does not correspond to any pre-defined unit system class,
`unitsystem` will dynamically define this class, cache it for reuse, and return
an instance for the set of units.

```{code-block} python

>>> from unxt.unitsystems import LTMAUnitSystem

>>> usys = unitsystem("kpc", "Myr", "solMass", "degree", "candela")
>>> usys
LengthTimeMassAngleLuminousIntensityUnitSystem(length=Unit("kpc"), time=Unit("Myr"), mass=Unit("solMass"), angle=Unit("deg"), luminous_intensity=Unit("cd"))

>>> isinstance(usys, LTMAUnitSystem)
False

```

`unxt.unitsystem` can create a dimensionless unit system if given `None`.

```{code-block} python

>>> unitsystem(None)
DimensionlessUnitSystem()

```

The dimensionless unit system is not the only special unit system. `unxt` also
supports creating dynamical unit systems, where $G = 1$ (or some other constant)
and one of the "length", "time", "mass" dimensions is adjusted to make the units
consistent.

```{code-block} python

>>> from unxt.unitsystems import DynamicalSimUSysFlag

>>> unitsystem(DynamicalSimUSysFlag, "m", "kg")
LengthMassTimeUnitSystem(length=Unit("m"), mass=Unit("kg"), time=Unit("122404 s"))

```

The construction of these constrained unit systems requires passing an
`unxt.unitsystems.AbstractUSysFlag`.

Also, `unitsystem` can replace a unit in a unit system or extend a unit system.

```{code-block} python

>>> usys = unitsystem("m", "kg", "s")

>>> unitsystem(usys) is usys
True

>>> unitsystem(usys, "deg")
LengthMassTimeAngleUnitSystem(length=Unit("m"), mass=Unit("kg"), time=Unit("s"), angle=Unit("deg"))

```

:::{note}

`unxt.unitsystem` supports a wide range of inputs, only some of which were
covered here. To see the full range of options, execute `unxt.unitsystem` in an
interactive Python session.

:::
