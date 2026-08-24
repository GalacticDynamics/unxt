# Unit systems

A {term}`unit system` is a collection of base units used together. `unxt` models them as frozen dataclasses subclassing `AbstractUnitSystem`, registered as static JAX pytree nodes. `unxt.unitsystems` provides the built-in realizations and two functions, both re-exported on the top-level `unxt` namespace.

| Function             | Purpose                              |
| -------------------- | ------------------------------------ |
| `unxt.unitsystem`    | Construct or retrieve a unit system. |
| `unxt.unitsystem_of` | Return the unit system of an object. |

To define your own subclass, see {doc}`../how-to/define-a-unit-system`.

## Built-in realizations

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

Each is also reachable by name through `unitsystem` (see below).

### Natural unit systems

[Natural unit systems](https://en.wikipedia.org/wiki/Natural_units) fix a chosen set of fundamental physical constants to the dimensionless value 1. `unxt` realizes this _numerically_: the base units are chosen so that the named constants evaluate to `1.0`, while the full dimensional structure is preserved.

| Realization | Constants set to 1 | Base dimensions | Free scale |
| --- | --- | --- | --- |
| `hep` | `hbar`, `c` | length, mass, time | `HEPUSysFlag(energy=...)` |
| `geometrized` | `c`, `G` | length, mass, time | `GeometrizedUSysFlag(length=...)` |
| `planck` | `hbar`, `c`, `G`, `k_B` | length, mass, time, temperature | none — fully determined |
| `atomic` | `m_e`, `hbar`, `e`, `4*pi*eps0` | length, mass, time, electrical charge | none |

```{code-block} python
>>> from unxt.unitsystems import hep, geometrized, planck, atomic

>>> hep  # high-energy physics: hbar = c = 1  (1 GeV scale)
LengthMassTimeUnitSystem(length=Unit("...e-16 m"), mass=Unit("...e-27 kg"), time=Unit("...e-25 s"))

>>> geometrized  # general relativity: c = G = 1  (1 m scale)
LengthMassTimeUnitSystem(length=Unit("m"), mass=Unit("...e+27 kg"), time=Unit("...e-09 s"))

>>> planck  # hbar = c = G = k_B = 1
LengthMassTimeTemperatureUnitSystem(length=Unit("...e-35 m"), mass=Unit("...e-08 kg"), time=Unit("...e-44 s"), temperature=Unit("...e+32 K"))

>>> atomic  # Hartree: m_e = hbar = e = 4*pi*eps0 = 1
LengthMassTimeElectricalChargeUnitSystem(length=Unit("...e-11 m"), mass=Unit("...e-31 kg"), time=Unit("...e-17 s"), electrical_charge=Unit("...e-19 A s"))
```

Natural unit systems are _numeric_ only: they do not add equivalencies between dimensions, so a `Quantity` in `MeV` remains an energy, not a mass. For worked examples on each system see {doc}`../how-to/work-in-natural-units`.

## `unitsystem`

```{code-block} python
>>> from unxt.unitsystems import unitsystem, unitsystem_of
```

**From a name.** Returns the corresponding built-in realization.

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

**From a set of units.** If the dimensions match a pre-defined unit system class, an instance of that class is returned. `"galactic"` and `"solarsystem"` are both instances of `LTMAUnitSystem` (length-time-mass-angle).

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

If the dimensions match no pre-defined class, a class is defined dynamically, cached for reuse, and instantiated.

```{code-block} python

>>> usys = unitsystem("kpc", "Myr", "solMass", "degree", "candela")
>>> usys
AngleLengthLuminousIntensityMassTimeUnitSystem(angle=Unit("deg"), length=Unit("kpc"), luminous_intensity=Unit("cd"), mass=Unit("solMass"), time=Unit("Myr"))

>>> isinstance(usys, LTMAUnitSystem)
False

```

**From `None`.** Returns the dimensionless unit system.

```{code-block} python

>>> unitsystem(None)
DimensionlessUnitSystem()

```

**From a flag.** Constrained unit systems are constructed by passing an `unxt.unitsystems.AbstractUSysFlag` as the first argument. `DynamicalSimUSysFlag` sets $G = 1$, solving for whichever of length/time/mass is left unspecified.

```{code-block} python

>>> from unxt.unitsystems import DynamicalSimUSysFlag

>>> unitsystem(DynamicalSimUSysFlag, "m", "kg")
LengthMassTimeUnitSystem(length=Unit("m"), mass=Unit("kg"), time=Unit("122404 s"))

```

**From an existing system.** Passing a unit system returns it unchanged; passing additional units extends or replaces.

```{code-block} python

>>> usys = unitsystem("m", "kg", "s")

>>> unitsystem(usys) is usys
True

>>> unitsystem(usys, "deg")
unitsystem(m, s, kg, deg)

```

:::{note}

`unitsystem` accepts a wider range of inputs than are shown here. Run `unxt.unitsystem` in an interactive session for the full list of registered signatures.

:::

## Indexing a unit system

Indexing with a dimension name returns the unit that system uses for it, composing derived units from the base units as needed.

```{code-block} python
>>> import unxt as u
>>> usys = u.unitsystem("si")

>>> usys["length"]
Unit("m")

>>> usys["velocity"]
Unit("m / s")

>>> usys["energy"]
Unit("m2 kg / s2")

```

## Comparison

`==` compares unit systems structurally. {func}`unxt.equivalent` reports whether two systems span the same dimensions — see {doc}`../explanation/equality-and-equivalence`.

## See also

- {doc}`units` — individual units.
- {doc}`../how-to/define-a-unit-system` — writing an `AbstractUnitSystem` subclass.
- {doc}`../how-to/work-in-natural-units` — worked examples in each natural system.
- {doc}`API documentation for unit systems <api/unitsystems>`
