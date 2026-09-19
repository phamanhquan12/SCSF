"""Identity/integrity helpers for NEXT5 teacher targets and neighbor memory."""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np


OOF_SCHEMA = "scsf.oof_v1"
MEMORY_SCHEMA = "scsf.memory_v1"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: str, payload: dict) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)
    return _sha256_file(path)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def validate_oof_payload(payload: dict, expected_ids, *, teacher_hash=None,
                         source_sha=None) -> None:
    if payload.get("schema") != OOF_SCHEMA:
        raise ValueError(f"OOF schema mismatch: {payload.get('schema')!r}")
    rows = payload.get("rows") or []
    ids = [int(r["id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("OOF cache contains duplicate sample IDs")
    expected = set(int(i) for i in expected_ids)
    got = set(ids)
    missing = expected - got
    extra = got - expected
    if missing or extra:
        raise ValueError(
            f"OOF ID mismatch: missing={len(missing)} extra={len(extra)}"
        )
    if teacher_hash is not None and payload.get("teacher_config_hash") != teacher_hash:
        raise ValueError("OOF teacher_config_hash does not match the teacher run")
    if source_sha is not None and payload.get("source_sha") not in (None, "", source_sha):
        if payload.get("source_sha") != source_sha:
            raise ValueError("OOF source_sha does not match the execution checkout")


def merge_oof_payloads(parts: list, allow_smoke_overlap: bool = False) -> dict:
    """Merge per-fold OOF shards; hard-fail on overlap or schema mismatch."""
    if not parts:
        raise ValueError("no OOF shards to merge")
    seen = {}
    rows = []
    smoke = all(bool(p.get("smoke_only")) for p in parts)
    for p in parts:
        if p.get("schema") != OOF_SCHEMA:
            raise ValueError("OOF shard schema mismatch")
        for r in p.get("rows") or []:
            i = int(r["id"])
            if i in seen:
                if smoke or allow_smoke_overlap:
                    continue
                raise ValueError(f"OOF fold overlap on id={i}")
            seen[i] = True
            rows.append(r)
    rows.sort(key=lambda r: int(r["id"]))
    return {
        "schema": OOF_SCHEMA,
        "merged": True,
        "smoke_only": smoke,
        "n": len(rows),
        "parts": [
            {
                "fold": p.get("fold"),
                "teacher_run": p.get("teacher_run"),
                "teacher_config_hash": p.get("teacher_config_hash"),
                "checkpoint": p.get("checkpoint"),
                "fold_hash": p.get("fold_hash"),
                "source_sha": p.get("source_sha"),
                "smoke_only": p.get("smoke_only", False),
            }
            for p in parts
        ],
        "rows": rows,
    }


def oof_error_table(payload: dict, n_official: int):
    """Return (error float32 [N], valid bool [N], logits float32 [N,C] or None)."""
    err = np.full((n_official,), np.nan, dtype=np.float32)
    valid = np.zeros((n_official,), dtype=bool)
    logits = None
    max_c = 0
    for r in payload["rows"]:
        i = int(r["id"])
        err[i] = 1.0 - float(r["correct"])
        valid[i] = True
        if r.get("logits") is not None:
            max_c = max(max_c, len(r["logits"]))
    if max_c:
        logits = np.full((n_official, max_c), np.nan, dtype=np.float32)
        for r in payload["rows"]:
            if r.get("logits") is not None:
                logits[int(r["id"])] = np.asarray(r["logits"], dtype=np.float32)
    return err, valid, logits


def validate_memory_file(path: str, expected_ids, *, reference_hash=None) -> dict:
    data = np.load(path, allow_pickle=True)
    if str(data.get("schema", MEMORY_SCHEMA)) not in (MEMORY_SCHEMA, "scsf.memory_v1"):
        # np.savez stores strings as arrays
        schema = data["schema"].item() if "schema" in data else None
        if schema != MEMORY_SCHEMA:
            raise ValueError(f"memory schema mismatch: {schema!r}")
    ids = np.asarray(data["ids"]).astype(np.int64)
    if len(ids) != len(set(ids.tolist())):
        raise ValueError("memory contains duplicate IDs")
    expected = set(int(i) for i in expected_ids)
    got = set(int(i) for i in ids.tolist())
    if got != expected:
        raise ValueError(
            f"memory ID mismatch: missing={len(expected-got)} extra={len(got-expected)}"
        )
    if reference_hash is not None:
        stored = data["reference_config_hash"].item() if "reference_config_hash" in data else None
        if stored != reference_hash:
            raise ValueError("memory reference_config_hash does not match the CE anchor")
    return {k: data[k] for k in data.files}


__all__ = [
    "MEMORY_SCHEMA",
    "OOF_SCHEMA",
    "load_json",
    "merge_oof_payloads",
    "oof_error_table",
    "save_json",
    "validate_memory_file",
    "validate_oof_payload",
]
