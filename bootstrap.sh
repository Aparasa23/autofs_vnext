#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip

pip install -r requirements.txt
pip install -e .

echo ""
echo "Installed core dependencies."
echo "Try:"
echo "  autofs-vnext list-methods"
echo "  autofs-vnext run --config examples/config_classification.json"
echo ""
echo "Optional plugin deps:"
echo "  pip install -r requirements-plugins.txt"
