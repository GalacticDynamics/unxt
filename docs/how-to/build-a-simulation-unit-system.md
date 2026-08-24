# How to build a unit system where G = 1

Gravitational dynamics codes almost always work in units where the gravitational constant is exactly 1. It removes $G$ from every force evaluation in the inner loop, and it keeps positions, masses and times at order unity instead of spanning forty orders of magnitude in SI.

`unxt` builds such a system for you: fix any **two** of length, mass and time, and it solves for the third so that $G = 1$.

```{code-block} python
>>> import numpy as np
>>> import unxt as u
>>> from astropy import constants as const
>>> from unxt.unitsystems import DynamicalSimUSysFlag

```

## Fix length and mass, solve for time

Pass `DynamicalSimUSysFlag` as the first argument, then the two units you want to keep. For galactic dynamics, kiloparsecs and solar masses:

```{code-block} python
>>> usys = u.unitsystem(DynamicalSimUSysFlag, "kpc", "solMass")
>>> usys
LengthMassTimeUnitSystem(length=Unit("kpc"), mass=Unit("solMass"), time=Unit("...kpc(3/2) s kg(1/2) / (solMass(1/2) m(3/2))"))

```

The time unit is whatever it had to be. It is not a round number and it is not meant to be read — it is the time unit in which $G$ comes out as 1:

```{code-block} python
>>> bool(np.isclose(const.G.decompose(usys).value, 1.0))
True

```

To see what that time unit actually is in seconds, decompose it:

```{code-block} python
>>> usys["time"].decompose()
Unit("...e+19 s")

```

or convert one of them to something familiar:

```{code-block} python
>>> u.Q(1.0, usys["time"]).uconvert("Myr")
Quantity(Array(471483., dtype=float32...), unit='Myr')

```

## Fix length and time instead

If your integrator's timestep is the thing you care about, fix time and let the mass unit fall out:

```{code-block} python
>>> usys2 = u.unitsystem(DynamicalSimUSysFlag, "kpc", "Myr")
>>> usys2["mass"].decompose()
Unit("...e+41 kg")

>>> bool(np.isclose(const.G.decompose(usys2).value, 1.0))
True

```

Same guarantee, different free choice. Which two you fix is a question of which numbers you want to be readable.

## Compare with a plain unit system

The built-in `galactic` system uses the same kind of units but makes no promise about $G$:

```{code-block} python
>>> float(const.G.decompose(u.unitsystem("galactic")).value)
4.4985021514695...e-12

```

That factor is exactly what a `DynamicalSimUSysFlag` system saves you from carrying through every force calculation.

## Move values in and out

Convert into the system with `ustrip`, passing the whole system so it picks the unit matching each quantity's dimension:

```{code-block} python
>>> float(u.Q(220.0, "km/s").ustrip(usys))
106082.1...

```

A circular velocity of 220 km/s is about 1.06e5 in these units. Going the other way, build the quantity with the system's own unit and convert out:

```{code-block} python
>>> u.Q(1.0, usys["time"]).uconvert("Myr")
Quantity(Array(471483., dtype=float32...), unit='Myr')

```

The derived units are composites of the solved base unit, and read as a plain scale factor on SI:

```{code-block} python
>>> usys["velocity"]
Unit("2.07387 m / s")

>>> usys["acceleration"]
Unit("1.39383e-19 m / s2")

```

## See also

- {doc}`../reference/unitsystems` — the other flags, including the natural-unit ones, and every input `unitsystem` accepts.
- {doc}`work-in-natural-units` — systems where $\hbar$, $c$ or $k_B$ are 1.
- {doc}`../tutorials/mars-lander` — building and working in a plain custom system.
