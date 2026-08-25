# Land on Mars in your own units

In 1999 the Mars Climate Orbiter was lost. The investigation board found the cause: one team's software produced impulse in pound-force seconds, and the software receiving it expected newton-seconds. Both numbers were correct. Neither carried its unit across the interface.

In this tutorial we will build a unit system for a Mars mission, write a small descent simulation in it, land a spacecraft — and then reproduce that failure and watch our lander hit the ground at 139 metres per second. The simulation is our own invention, not a reconstruction of the mission; the unit mistake is the real one.

You need `unxt` installed and nothing else.

## Set up

```{code-block} python
>>> import unxt as u
>>> from astropy.units import imperial

```

## Build the mission unit system

Every mission picks working units and sticks to them. Ours will be kilometres, seconds, kilograms and radians:

```{code-block} python
>>> usys = u.unitsystem("km", "s", "kg", "rad")
>>> usys
unitsystem(km, s, kg, rad)

```

Those four are the _base_ units. Now ask the system for units it was never given:

```{code-block} python
>>> usys["velocity"]
Unit("km / s")

>>> usys["acceleration"]
Unit("km / s2")

>>> usys["force"]
Unit("kg km / s2")

```

Nobody wrote `kg km / s2` down. The system composed it from the base units the moment we asked for a force. Ask it for the quantity at the heart of the Mars Climate Orbiter failure, and it composes that too:

```{code-block} python
>>> usys["impulse"]
Unit("kg km / s")

```

## Set up the lander

We are two kilometres up, falling at 100 metres per second, with a half-tonne spacecraft. Notice we ask the system for the units rather than typing them:

```{code-block} python
>>> h0 = u.Q(2.0, usys["length"])
>>> h0
Quantity(Array(2., dtype=float32, weak_type=True), unit='km')

>>> mass = u.Q(500.0, usys["mass"])
>>> v0 = u.Q(-100.0, "m/s")

```

`v0` is in metres per second — not one of our base units, and that is fine. Mixing is allowed; the arithmetic will reconcile it. Mars gravity, likewise, in whatever units it was handed to us in:

```{code-block} python
>>> g_mars = u.Q(3.72076, "m/s2")
>>> dt = u.Q(0.5, usys["time"])

```

## Write the simulator

A braking burn: fire the engine whenever we are descending, integrate, and stop when we touch the ground.

```{code-block} python
>>> def descend(thrust):
...     h, v, steps = h0, v0, 0
...     while float(h.ustrip("m")) > 0.0 and steps < 3000:
...         if float(v.ustrip("m/s")) < 0.0:
...             a = thrust / mass - g_mars
...         else:
...             a = -g_mars
...         v = v + a * dt
...         h = h + v * dt
...         steps += 1
...     return v, steps

```

Every line of that is unit-aware. `thrust / mass` is a force over a mass; we never told it that makes an acceleration, and we never converted `km` against `m/s` by hand.

## Fly it

Our engine contractor quotes the thrust in pound-force, because that is what their test stand reads:

```{code-block} python
>>> thrust = u.Q(700.0, imperial.lbf)
>>> thrust
Quantity(Array(700., dtype=float32, weak_type=True), unit='lbf')

```

Hand it straight to the simulator:

```{code-block} python
>>> v_td, steps = descend(thrust)
>>> round(float(v_td.ustrip("m/s")), 2)
-0.58

```

Touchdown at 0.58 metres per second, after

```{code-block} python
>>> steps
288

```

steps. That is a landing. We never converted the contractor's pound-force into anything — `unxt` did it inside the arithmetic.

## Now make the Mars Climate Orbiter mistake

Suppose the thrust figure reaches our simulator the way it reached the orbiter's navigation software: as a bare number, stripped of its unit somewhere upstream, and assumed to be in newtons.

```{code-block} python
>>> v_bug, steps_bug = descend(u.Q(700.0, "N"))
>>> round(float(v_bug.ustrip("m/s")), 2)
-139.45

```

The lander hits the ground at 139 metres per second. And it gets there fast:

```{code-block} python
>>> steps_bug
34

```

Thirty-four steps instead of 288 — it barely slowed at all.

Here is the entire reason, in one line:

```{code-block} python
>>> thrust.uconvert("N")
Quantity(Array(3113.7551, dtype=float32, weak_type=True), unit='N')

```

700 pound-force is 3114 newtons. Reading the number as newtons threw away a factor of 4.45 of engine — and the number `700.0` looked perfectly reasonable the whole way down.

## Report in whichever units you like

The result is a physical fact, not a number in a particular unit. Ask for it in the mission system:

```{code-block} python
>>> v_td.uconvert(usys["velocity"])
Quantity(Array(-0.0005835, dtype=float32, weak_type=True), unit='km / s')

```

Passing the whole system, rather than one unit, lets it pick the right unit for whatever dimension you hand it:

```{code-block} python
>>> round(float(v_td.ustrip(usys)), 6)
-0.000583

```

And a second unit system gives the same landing in different numbers:

```{code-block} python
>>> si = u.unitsystem("m", "s", "kg", "rad")
>>> round(float(v_td.ustrip(si)), 2)
-0.58

```

Same touchdown. Two systems, two numbers, one physical answer — which is the whole reason to let the system carry the units instead of your head.

## What we built

You have a mission unit system that composes derived units on demand, a descent simulation written entirely in quantities, and a landing flown from a thrust figure quoted in units the simulation never mentions. You also have the failure: one bare number crossing one interface, and a spacecraft at 139 metres per second.

The lesson is not to be more careful. It is that a number travelling without its unit is the bug, and the fix is to never let it.

## Where to go next

- {doc}`../reference/unitsystems` — the built-in systems, and every input `unitsystem` accepts.
- {doc}`../how-to/define-a-unit-system` — when you want a named, statically defined system rather than one built on the fly.
- {doc}`../how-to/work-in-natural-units` — unit systems where the constants are 1.
- {doc}`../explanation/sharp-bits` — including why a dimensionful quantity refuses to become a bare array.
