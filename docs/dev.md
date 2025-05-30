# Developer Guide

## Building the Documentation

To build the documentation, use `nox`:

```bash
nox -s docs
```

To build with `sphinx-autobuild` (and host a local web server for live reloading):

```bash
nox -s docs -- -serve
```
