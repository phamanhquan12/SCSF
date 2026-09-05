#!/bin/bash
# SCSF SAGE-v2 preservation manifest generator (server-side).
# Hashes seed-13 artifacts + frozen v1 baselines + confirmation run dirs.
set -u
cd /root
OUT=scsf_v2_preservation_manifest.txt
rm -f "$OUT"
{
  echo "# SCSF SAGE-v2 preservation manifest (SHA256)  generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# host: $(uname -a | cut -c1-120)"
  echo "# v1 frozen tree: /root/scsf (commit 0fda4f8, NOT altered)"
  echo "# v2 mirror:      /root/scsf_v2 (commit 9649eea)"
  echo
  echo "== seed-13 v2 runs (exploratory, commit 9649eea) =="
  for d in /root/scsf_v2_results/cifar10-vgg16_bn-sage_ds_v2-rccl_sc_reference-s13 \
           /root/scsf_v2_results/cifar100-vgg16_bn-sage_ds_v2-rccl_sc_reference-s13; do
    while read -r f; do sha256sum "$f"; done < <(find "$d" -type f \( -name "*.jsonl" -o -name "*.json" -o -name "*.pt" -o -name "*.csv" \) | sort)
  done
  echo
  echo "== frozen v1 seed-13 baselines (read-only) =="
  while read -r f; do sha256sum "$f"; done < <(find /root/scsf/results/cifar10-vgg16_bn-sage_ds-rccl_sc_reference-s13 \
                                                    /root/scsf/results/cifar100-vgg16_bn-sage_ds-rccl_sc_reference-s13 \
                                                    -type f \( -name "*.jsonl" -o -name "*.json" -o -name "*.pt" -o -name "*.csv" \) | sort)
  echo
  echo "== confirmation run dirs (in progress) =="
  while read -r f; do sha256sum "$f"; done < <(find /root/scsf_v2_confirm -type f \( -name "*.jsonl" -o -name "*.json" -o -name "*.pt" -o -name "*.csv" -o -name "progress*" \) | sort)
} > "$OUT"
echo "lines: $(wc -l < "$OUT")"
echo "utility telemetries hashed: $(grep -c 'sage_ds_v2_utility.jsonl' "$OUT")"
head -4 "$OUT"