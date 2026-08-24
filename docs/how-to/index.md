# How-to guides

Directions for getting a specific job done. These pages assume you already know what you want and are looking for the shortest route to it.

If you are new to `unxt`, work through {doc}`../tutorials/first-quantity` first — these guides will not teach you the basics.

```{toctree}
:maxdepth: 1

install
convert-units
use-jax-functions
control-display
check-types-at-runtime
register-a-quax-rule
define-a-unit-system
build-a-simulation-unit-system
work-in-natural-units
interoperate-with-astropy
optimize-performance
migrate-to-v2
```

## Getting set up

- {doc}`install` — install `unxt` and its optional packages.
- {doc}`migrate-to-v2` — upgrade from `unxt` v1.

## Working with quantities

- {doc}`convert-units` — `uconvert`, `ustrip` and `uconvert_value`.
- {doc}`use-jax-functions` — `quaxify`, `quaxed`, `jit`, autodiff and functional updates.
- {doc}`control-display` — change how quantities render, for a call, a block, a process or a project.

## Units and unit systems

- {doc}`define-a-unit-system` — write your own `AbstractUnitSystem` subclass.
- {doc}`build-a-simulation-unit-system` — a system where $G = 1$, for dynamics codes.
- {doc}`work-in-natural-units` — worked examples in HEP, geometrized, atomic and Planck units.

## Correctness and speed

- {doc}`check-types-at-runtime` — enforce dtype and shape annotations at runtime.
- {doc}`optimize-performance` — keep wrapper overhead off your hot path.
- {doc}`register-a-quax-rule` — teach JAX a primitive rule of your own.
- {doc}`interoperate-with-astropy` — move values between `astropy` and `unxt` safely.
