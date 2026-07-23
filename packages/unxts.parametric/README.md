# unxts.parametric

Dimension-parametrized quantities for [unxt](https://github.com/GalacticDynamics/unxt).

This is the canonical package (`unxts.parametric`). It provides quantity types parametrized by physical dimension, enabling static and runtime enforcement of dimensional constraints.

## Install

```bash
pip install unxts.parametric
```

## Usage

`ParametricQuantity` (alias `PQ`) encodes its physical dimension in its _type_, so — unlike the default `unxt.Quantity` — it checks the dimension at construction:

```python
import unxts.parametric as up

up.PQ([1, 2, 3], "m")  # dimension inferred from the unit
up.PQ["length"](1, "m")  # dimension checked against the unit

# A unit that doesn't match the declared dimension raises:
try:
    up.PQ["length"](1, "s")
except ValueError as e:
    print(e)  # Physical type mismatch.
```
