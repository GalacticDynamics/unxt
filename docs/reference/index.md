# Reference

Descriptions of what `unxt` contains: classes, functions, options and their defaults. Consult these while you work; they do not explain or instruct.

For instructions see the {doc}`../how-to/index`; for the reasoning behind the design see {doc}`../explanation/index`.

```{toctree}
:maxdepth: 1
:hidden:

quantity
units
unitsystems
dimensions
configuration
dataclassish
glossary
api/index
```

## Core types

- {doc}`quantity` — `Quantity`, `Angle`, `StaticQuantity`, `StaticValue`.
- {doc}`dimensions` — `dimension` and `dimension_of`, and the expression syntax.
- {doc}`units` — `unit` and `unit_of`.
- {doc}`unitsystems` — built-in realizations, natural unit systems, `unitsystem`.

## Settings and interop

- {doc}`configuration` — every display option, its type and its default, including the `pyproject.toml` keys.
- {doc}`dataclassish` — what each `dataclassish` function returns for each `unxt` type.

## Vocabulary and generated API

- {doc}`glossary` — the terms used throughout these docs.
- {doc}`api/index` — the generated API documentation, module by module.

## Not yet written

There is no error reference keyed by message text. `unxt` raises mostly through `astropy`'s `UnitConversionError`, so error strings currently appear only inline in the guides that provoke them.
