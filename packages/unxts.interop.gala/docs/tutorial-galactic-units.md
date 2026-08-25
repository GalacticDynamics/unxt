# Compute in galactic units

In this tutorial we will borrow a unit system from [`gala`][gala-link], bring it into `unxt`, and use it to work out how long the Sun takes to go once around the Galaxy — with every number carrying its units and the working units chosen by the system rather than typed in.

You need `unxt`, `unxts.interop.gala` and `gala` installed, and nothing else.

## Set up

```{code-block} python
>>> import unxt as u
>>> import gala.units as gu

```

Importing `unxt` registers the conversions for you when `gala` is present, so there is nothing else to import.

## Look at gala's unit system

`gala` ships a `galactic` system — the units galactic dynamicists actually work in:

```{code-block} python
>>> gu.galactic
<UnitSystem (kpc, Myr, solMass, rad)>

```

Kiloparsecs, megayears, solar masses, radians.

## Bring it into unxt

{func}`unxt.unitsystem` accepts it directly:

```{code-block} python
>>> usys = u.unitsystem(gu.galactic)
>>> usys
unitsystem(kpc, Myr, solMass, rad)

```

Same four units, now an `unxt` unit system.

## Ask it for units by dimension

Here is what that buys us. Rather than remembering which unit the system uses for what, we ask it:

```{code-block} python
>>> usys["length"]
Unit("kpc")

>>> usys["time"]
Unit("Myr")

>>> usys["mass"]
Unit("solMass")

```

And it will compose units it was never given, for dimensions built out of its base ones:

```{code-block} python
>>> usys["velocity"]
Unit("kpc / Myr")

```

Nobody put `kpc / Myr` in the system. It came from length over time.

## Build quantities in those units

Now let's set up the problem. The Sun sits about 8 kpc from the Galactic centre — and we can say "in whatever this system calls a length" instead of naming the unit:

```{code-block} python
>>> r = u.Q(8.0, usys["length"])
>>> r
Quantity(Array(8., dtype=float32, weak_type=True), unit='kpc')

```

Its orbital speed is about 220 km/s. That is _not_ a unit in this system, and it does not have to be — let's write it as it is normally quoted, then move it into the system's units:

```{code-block} python
>>> v = u.Q(220.0, "km/s")
>>> v.uconvert(usys["velocity"])
Quantity(Array(0.22499669, dtype=float32, weak_type=True), unit='kpc / Myr')

```

220 km/s is about 0.225 kpc per megayear. Notice we asked the _system_ for the target unit rather than writing `"kpc/Myr"` ourselves.

## Do the physics

Time to go round is distance over speed. We can just divide — the units sort themselves out:

```{code-block} python
>>> t = (r / v).uconvert("Myr")
>>> t
Quantity(Array(35.55608, dtype=float32, weak_type=True), unit='Myr')

```

About 36 megayears for a radian of orbit. We divided kiloparsecs by kilometres per second and got megayears, without a single conversion factor written by hand.

Strip it back to a bare number in the system's own units when you need to hand it to something that does not speak units:

```{code-block} python
>>> t.ustrip(usys)
Array(35.55608, dtype=float32, weak_type=True)

```

Passing the whole `usys` picks the right unit for the quantity's dimension — we did not have to say "megayears" again.

## What we built

You took a unit system defined in `gala`, used it in `unxt` as the source of truth for which units to work in, and computed an orbital timescale from a distance and a speed quoted in unrelated units. The system chose the working units; the arithmetic chose the result's.

## Where to go next

- [How to convert between gala and unxt unit systems](guide) — both directions, and what a round trip does and does not preserve.
- [API](api) — the exposed conversion functions.

[gala-link]: https://gala.adrian.pw/en/stable/
