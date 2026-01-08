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
