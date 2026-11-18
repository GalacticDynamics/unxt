# Concepts

These pages explain how `unitful` connects a model written in units to NumPyro, which computes on plain numbers. They contain no code to run.

- {doc}`sites` explains how `unitful` strips and wraps the units at each sample site, and where the units go after a fit.
- {doc}`unit-systems` explains why `unitful` needs a unit system, and how it changes the log density.
- {doc}`handler-order` explains why some handlers go inside `unitful` and others outside.

```{toctree}
:hidden:

sites
unit-systems
handler-order
```
