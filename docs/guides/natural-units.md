# Working in Natural Units

[Natural unit systems](https://en.wikipedia.org/wiki/Natural_units) set a chosen set of fundamental physical constants to the dimensionless value 1. This removes those constants from equations and makes quantities of different "everyday" dimensions share a scale — a mass expressed as an energy, a length as a time, and so on.

`unxt` provides four natural unit systems built in. They are realized _numerically_: the base units are chosen so the named constants evaluate to `1.0`, while the full dimensional structure is preserved (so `unxt`'s dimension checking keeps working). This tutorial works through each one on a small physics problem.

```{code-block} python
>>> import numpy as np
>>> import unxt as u
>>> from astropy import constants as const, units as apu
>>> from unxt.unitsystems import unitsystem, hep, geometrized, atomic, planck
```

## High-energy physics: masses as energies

In particle physics one sets $\hbar = c = 1$ and measures everything in powers of an energy, conventionally the GeV. The built-in `hep` system uses exactly this scale, so its mass unit is $1\,\mathrm{GeV}/c^2$:

```{code-block} python
>>> bool(np.isclose((1 * hep["mass"]).to_value("kg"),
...                 (1 * apu.GeV / const.c**2).to_value("kg")))
True
```

A proton then weighs about `0.938` in these units — the familiar $m_p \approx 0.938\ \mathrm{GeV}/c^2$:

```{code-block} python
>>> round(float(const.m_p / (1 * hep["mass"])), 3)
0.938
```

The energy scale is configurable through `HEPUSysFlag`. A larger energy gives a smaller length and time (since $\hbar = c = 1$):

```{code-block} python
>>> from unxt.unitsystems import HEPUSysFlag
>>> unitsystem(HEPUSysFlag, energy="TeV")["time"] == hep["time"] / 1000
True
```

## Geometrized units: masses as lengths

In general relativity one sets $c = G = 1$, turning masses and times into lengths. The natural length scale for a gravitating body is its _gravitational radius_ $r_g = G M / c^2$. Building a geometrized system at the Sun's gravitational radius makes the mass unit exactly one solar mass:

```{code-block} python
>>> from unxt.unitsystems import GeometrizedUSysFlag
>>> r_g = const.G * const.M_sun / const.c**2   # the Sun's gravitational radius
>>> apu.Unit(r_g)
Unit("1476... m")

>>> geo_sun = unitsystem(GeometrizedUSysFlag, length=apu.Unit(r_g))
>>> bool(np.isclose((1 * geo_sun["mass"]).to_value("kg"),
...                 const.M_sun.to_value("kg")))
True
```

By construction, $c$ and $G$ are both 1 in any geometrized system:

```{code-block} python
>>> bool(np.isclose(const.c.decompose(geometrized).value, 1.0))
True
>>> bool(np.isclose(const.G.decompose(geometrized).value, 1.0))
True
```

## Atomic units: the scale of the atom

Atomic (Hartree) units set $m_e = \hbar = e = 4\pi\varepsilon_0 = 1$. The length unit is the Bohr radius — about half an ångström:

```{code-block} python
>>> round(float((1 * atomic["length"]).to_value("Angstrom")), 4)
0.5292
```

Unlike the other systems, atomic units carry an electric-charge base dimension (the elementary charge):

```{code-block} python
>>> [str(d) for d in atomic.base_dimensions]
['length', 'mass', 'time', 'electrical charge']
```

## Planck units

Planck units set $\hbar = c = G = k_B = 1$ and are _fully determined_ — there is no free scale. The base units are the Planck length, mass, time, and temperature:

```{code-block} python
>>> [str(d) for d in planck.base_dimensions]
['length', 'mass', 'time', 'temperature']

>>> for name in ("c", "hbar", "G", "k_B"):
...     assert np.isclose(getattr(const, name).decompose(planck).value, 1.0)
```

## A note on semantics

`unxt`'s natural unit systems are _numeric_: they choose base units so the named constants equal 1. They do **not** (yet) add equivalences that let you convert directly between, say, a mass and an energy — a `Quantity` in `MeV` is still an energy, not a mass. Decomposing into a natural unit system (as above) is the supported way to obtain natural-unit values.
