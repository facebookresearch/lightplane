# Lightplane documentation

## Build instructions

1) [Install `lightplane`](../README.md#installation)

2) Install dependencies:
```bash
pip install sphinx
pip install myst-parser
pip install sphinx-rtd-theme
```

3) Build documentation
```bash
cd ${LIGHTPLANE_ROOT}/docs/
make html
```

4) The documentation can be found under `${LIGHTPLANE_ROOT}/docs/_build/`.