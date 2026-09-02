#!/usr/bin/env bash
#
# Static + dynamic verification before a push: compile every module, import
# every package, then run the full pytest suite. Uses only the local venv.
#
set -euo pipefail

cd "$(dirname "$0")/.."
PY=./.venv/bin/python

echo "== py_compile all package modules =="
find scsf tests -name '*.py' -print0 | xargs -0 "$PY" -m py_compile

echo "== import every package =="
"$PY" - <<'PY'
import importlib, pkgutil
mods = []
for p in pkgutil.walk_packages(__import__("scsf").__path__,
                               prefix="scsf."):
    importlib.import_module(p.name)
    mods.append(p.name)
print(f"imported {len(mods)} modules: {', '.join(mods)}")
PY

echo "== pytest =="
PYTHONPATH=. "$PY" -m pytest tests/ -q

echo "verify OK"