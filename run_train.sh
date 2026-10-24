#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${DATA_ROOT:-/home/xxj/Data}"
gpu_ids="${GPU_IDS:-0}"
devices="${DEVICES:-1}"
num_workers="${NUM_WORKERS:-8}"
pretrained="${PRETRAINED:-$repo_root/ckpt/gcc-b_in1k.pth}"

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s {levircd|levircdplus|cdd|whucd|sysucd}\n' "$0" >&2
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

for split in train val; do
  for folder in "$image_a" "$image_b" "$label_dir"; do
    if [[ ! -d "$dataset_root/$split/$folder" ]]; then
      printf 'Dataset directory missing: %s\nRun prepare_server_data.sh first or set DATA_ROOT.\n' "$dataset_root/$split/$folder" >&2
      exit 1
    fi
  done
done

if [[ ! -f "$pretrained" ]]; then
  printf 'ChangeGCC-B pretrained checkpoint not found: %s\n' "$pretrained" >&2
  printf 'The repository README does not provide a download URL for gcc-b_in1k.pth. Set PRETRAINED after obtaining the official checkpoint.\n' >&2
  exit 1
fi

cd "$repo_root"
export CUDA_VISIBLE_DEVICES="$gpu_ids"
python train.py \
  --dataset "$dataset" \
  --data_dir "$dataset_root" \
  --image_dir_a "$image_a" \
  --image_dir_b "$image_b" \
  --label_dir "$label_dir" \
  --pretrain_dir "$pretrained" \
  --output_dir "$repo_root/ChangeGCC_work_dirs/$1" \
  --devices "$devices" \
  --num_workers "$num_workers"
