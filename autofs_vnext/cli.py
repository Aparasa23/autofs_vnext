from __future__ import annotations

import argparse
from autofs_vnext.core.schemas import load_config
from autofs_vnext.core.runner import run_autofs
from autofs_vnext.core.registry import REGISTRY, autodiscover

def _preflight():
    # Core dependency checks (fail fast with actionable guidance)
    required = [
        ("numpy", "pip install -r requirements.txt"),
        ("pandas", "pip install -r requirements.txt"),
        ("sklearn", "pip install -r requirements.txt"),
        ("scipy", "pip install -r requirements.txt"),
        ("joblib", "pip install -r requirements.txt"),
        ("yaml", "pip install -r requirements.txt"),
        ("tqdm", "pip install -r requirements.txt"),
    ]
    missing = []
    import importlib
    for mod, hint in required:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append((mod, hint))
    if missing:
        lines = ["Missing required dependencies:"]
        for mod, hint in missing:
            lines.append(f"  - {mod}  (hint: {hint})")
        lines.append("")
        raise SystemExit("\n".join(lines))

def _method_availability(meta):
    # returns (available: bool, missing: list[str])
    import importlib
    missing = []
    for req in getattr(meta, "requires", []) or []:
        try:
            importlib.import_module(req)
        except Exception:
            missing.append(req)
    return (len(missing) == 0, missing)

def main():
    _preflight()

    parser = argparse.ArgumentParser(prog="autofs_vnext", description="AutoFS vNext - feature selection platform")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run feature selection")
    run_p.add_argument("--config", required=True, help="Path to JSON config")

    list_p = sub.add_parser("list-methods", help="List available methods")

    args = parser.parse_args()

    if args.cmd == "list-methods":
        autodiscover()
        for m in sorted(REGISTRY.list(), key=lambda x: x.name):
            ok, miss = _method_availability(m)
            status = "available" if ok else ("missing:" + ",".join(miss))
            tasks_str = ",".join(sorted(m.tasks))
            print(f"{m.name}\t[{m.family}]\t{tasks_str}\t{m.compute}\t{status}\t{m.description}")
        return

    if args.cmd == "run":
        cfg = load_config(args.config)
        out = run_autofs(cfg)
        print(str(out))
        return

if __name__ == "__main__":
    main()
