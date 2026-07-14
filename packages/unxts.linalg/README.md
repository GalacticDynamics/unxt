# unxts.linalg

Heterogeneous-unit matrices and vectors for [unxt](https://github.com/GalacticDynamics/unxt).

This is the canonical package (`unxts.linalg`). It provides `QuantityMatrix` (alias `QM`): a quantity container whose elements may each carry a different unit, together with a static `UnitsMatrix` unit structure and unit-aware linear-algebra primitives (`det`, `inv`, and Quax-registered add/sub/matmul/transpose/diag).

## Install

```bash
pip install unxts.linalg
```

## Usage

```python
import jax.numpy as jnp
import unxts.linalg as ul

qv = ul.QuantityMatrix(jnp.array([1.0, 2.0, 3.0]), unit=("m", "s", "kg"))
```
