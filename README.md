# AutoFS vNext (Reference Implementation)

This is a clean, modernized feature selection platform implementing the milestone plan:
- Standardized preprocessing (numeric + categorical)
- Phase-based method execution
- Canonical scoring outputs
- Aggregation (RRF/Borda)
- Optional stability selection and redundancy clustering

## Quickstart

```bash
python -m autofs_vnext.cli list-methods
python -m autofs_vnext.cli run --config examples/config_classification.json
```

Outputs are written to: `output.dir/output.run_name/`

## Notes

- Some "market methods" require optional dependencies. In this reference implementation they are registered as placeholders
  (e.g., Boruta, MIC, genetic selection). Add them via plugin packages without changing core contracts.

- The legacy-style technique names from your older config examples are supported as aliases. fileciteturn2file4L393-L444

## Local Installation (recommended before containerization)

Place `requirements.txt` at the **repository root** (same folder as `pyproject.toml` and `README.md`). This repo includes:

- `requirements.txt` (core runtime deps)
- `requirements-core.txt` (same as above, explicit name)
- `requirements-plugins.txt` (optional deps for plugin methods)

### Quickstart (macOS/Linux)

```bash
./bootstrap.sh
autofs-vnext list-methods
autofs-vnext run --config examples/config_classification.json
```

### Manual virtualenv setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
autofs-vnext list-methods
```

### Optional plugins

```bash
pip install -r requirements-plugins.txt
```

## Containerization (Docker)

```bash
docker build -t autofs-vnext:local .
docker run --rm -it autofs-vnext:local list-methods
```
