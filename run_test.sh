#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${DATA_ROOT:-/home/xxj/Data}"
gpu_ids="${GPU_IDS:-0}"
num_workers="${NUM_WORKERS:-4}"

if [[ $# -ne 2 ]]; then
  printf 'Usage: %s {levircd|levircdplus|cdd|whucd|sysucd} CHECKPOINT\n' "$0" >&2
  exit 2
fi

case "$1" in
  levircd) dataset="LEVIR"; dataset_root="$data_root/LEVIR_CD"; image_a="A"; image_b="B"; label_dir="label" ;;
  levircdplus) dataset="LEVIR+"; dataset_root="$data_root/LEVIR_CD_PLUS/cut_data"; image_a="A"; image_b="B"; label_dir="label" ;;
  cdd) dataset="CDD"; dataset_root="$data_root/CDD"; image_a="A"; image_b="B"; label_dir="OUT" ;;
  whucd) dataset="WHU"; dataset_root="$data_root/WHUCD"; image_a="A"; image_b="B"; label_dir="label" ;;
  sysucd) dataset="SYSU"; dataset_root="$data_root/SYSU_CD"; image_a="time1"; image_b="time2"; label_dir="label" ;;
  *) printf 'Unknown dataset key: %s\n' "$1" >&2; exit 2 ;;
esac

checkpoint="$2"
if [[ ! -f "$checkpoint" ]]; then
  printf 'Checkpoint not found: %s\n' "$checkpoint" >&2
  exit 1
fi
for folder in "$image_a" "$image_b" "$label_dir"; do
  if [[ ! -d "$dataset_root/test/$folder" ]]; then
    printf 'Test data directory missing: %s\n' "$dataset_root/test/$folder" >&2
    exit 1
  fi
done

cd "$repo_root"
export CUDA_VISIBLE_DEVICES="$gpu_ids"
python test.py \
  --dataset "$dataset" \
  --data_path "$dataset_root" \
  --image_dir_a "$image_a" \
  --image_dir_b "$image_b" \
  --label_dir "$label_dir" \
  --model_path "$checkpoint" \
  --num_workers "$num_workers"
