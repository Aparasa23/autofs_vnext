from __future__ import annotations

import argparse
from autofs_vnext.core.schemas import load_config
from autofs_vnext.core.runner import run_autofs
from autofs_vnext.core.registry import REGISTRY, autodiscover

def main():
    parser = argparse.ArgumentParser(prog="autofs_vnext", description="AutoFS vNext - feature selection platform")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run feature selection")
    run_p.add_argument("--config", required=True, help="Path to JSON config")

    list_p = sub.add_parser("list-methods", help="List available methods")

    args = parser.parse_args()

    if args.cmd == "list-methods":
        autodiscover()
        for m in sorted(REGISTRY.list(), key=lambda x: x.name):
            print(f"{m.name}\t[{m.family}]\t{','.join(sorted(m.tasks))}\t{m.compute}\t{m.description}")
        return

    if args.cmd == "run":
        cfg = load_config(args.config)
        out = run_autofs(cfg)
        print(str(out))
        return

if __name__ == "__main__":
    main()
