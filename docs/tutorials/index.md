# Tutorials

Lessons that take you through building something with `unxt`, one step at a time. Start here if you have not used `unxt` before: these pages assume no prior knowledge and every step is checked to work.

If you already know what you want to accomplish, the {doc}`../how-to/index` will get you there faster.

```{toctree}
:maxdepth: 1
:hidden:

first-quantity
mars-lander
dimensional-analysis
ecosystem-pipeline
design-a-unitful-api
```

Work through them in order.

1. {doc}`first-quantity` — build a unit-aware projectile calculator, then compile, differentiate and vectorise it. **Start here.**
2. {doc}`mars-lander` — build your own unit system, fly a descent simulation in it, and reproduce the unit mistake that destroyed the Mars Climate Orbiter.
3. {doc}`dimensional-analysis` — catch errors in a formula, and derive a result, without computing anything.
4. {doc}`ecosystem-pipeline` — carry units through a whole analysis: load a dataset with `xarray`, compute a derived quantity, and plot it with `matplotlib`.
5. {doc}`design-a-unitful-api` — write a function other people call, that cannot be called wrong by accident.

Every `unxts.*` package has its own tutorial too, listed under **Packages** in the sidebar.
