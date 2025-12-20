# 📊 Matplotlib

If `matplotlib` is installed, `unxt` will automatically detect this and will
register parsers with `matplotlib` to enable plotting `Quantity` objects.

To ensure that a compatible version of `matplotlib` is installed, you can
install `unxt` with the `interop-mpl` extra:

::::{tab-set}

:::{tab-item} uv

```bash
uv add "unxt[interop-mpl]"
```

:::

:::{tab-item} pip

```bash
pip install unxt[interop-mpl]
```

::::

Once installed, you can plot `Quantity` objects directly with `matplotlib`:

::::{tab-set}

:::{tab-item} jax.numpy

```{code-block} python

import matplotlib.pyplot as plt
import jax.numpy as jnp
import unxt as u

x = u.Q(jnp.linspace(0, 360, 100), "deg")
y = u.Q(jnp.sin(x.ustrip("rad")), "")

plt.plot(x, y)
```

:::

:::{tab-item} quaxed.numpy

```{code-block} python

import quaxed.numpy as jnp

x = u.Q(jnp.linspace(0, 360, 100), "deg")
y = jnp.sin(x)

plt.plot(x, y)
```

:::
