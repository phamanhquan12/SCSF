"""Merge per-fold OOF shards.

``python -m scsf.merge_oof a.json b.json out=merged.json expected_ids_file=...``
"""

from __future__ import annotations

import os
import sys

from .engine.artifacts import load_json, merge_oof_payloads, save_json, validate_oof_payload
from .engine.config import overrides_from_cli


def main(argv=None) -> dict:
    argv = list(sys.argv[1:] if argv is None else argv)
    positional = [a for a in argv if "=" not in a]
    ov = overrides_from_cli([a for a in argv if "=" in a])
    if len(positional) < 2:
        raise SystemExit("usage: python -m scsf.merge_oof shard_a.json shard_b.json out=merged.json")
    out = str(ov.pop("out"))
    parts = [load_json(p) for p in positional]
    merged = merge_oof_payloads(parts)
    ids = [int(r["id"]) for r in merged["rows"]]
    validate_oof_payload(merged, ids)
    digest = save_json(out, merged)
    print(f"merge_oof: n={merged['n']} out={out} sha256={digest[:12]}")
    return merged


if __name__ == "__main__":
    main()
