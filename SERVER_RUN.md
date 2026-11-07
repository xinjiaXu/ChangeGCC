# ChangeGCC server setup

The adapters read `/home/xxj/Data` directly. They do not copy images or create dataset replicas. Model outputs are kept under `ChangeGCC_work_dirs/<DatasetName>/ChangeGCC-b/` (for example, CDD uses `ChangeGCC_work_dirs/CDD/ChangeGCC-b/`).

## Environment

From `/home/xxj/code/DINOCLIP_comparative_test/ChangeGCC`:

```bash
bash setup_server_env.sh
conda activate changegcc-rs
```

The repository README recommends Python 3.10 and gives a PyTorch 2.10/CUDA 13.0 example. This setup uses Python 3.10 and the PyTorch 2.7.1 CUDA 12.8 build already verified to detect the server's RTX 5090. All packages are installed in the isolated `changegcc-rs` environment. NumPy and headless OpenCV are pinned to versions compatible with the repository's image pipeline.

## Data layout

Optionally verify all five dataset paths:

```bash
bash prepare_server_data.sh /home/xxj/Data
```

The command only checks folders. The ChangeGCC loader now supports the existing split layout and reads filenames from each split's label directory, so the original `list/*.txt` files are not required for these datasets. The original flat `A/B/label/list` layout remains supported.

For `levircd`, the launcher tiles inputs into aligned 256x256 patches while reading. This matches the upstream LEVIR preprocessing while leaving raw 1024x1024 files untouched. If using an alternate LEVIR directory, set `LEVIRCD_ROOT` to its split-root path. Other dataset launchers do not enable tiling.

| Key | Dataset root | T1 | T2 | Mask |
| --- | --- | --- | --- | --- |
| `levircd` | `/home/xxj/Data/LEVIR_CD` | `A` | `B` | `label` |
| `levircdplus` | `/home/xxj/Data/LEVIR_CD_PLUS/cut_data` | `A` | `B` | `label` |
| `cdd` | `/home/xxj/Data/CDD` | `A` | `B` | `OUT` |
| `whucd` | `/home/xxj/Data/WHUCD` | `A` | `B` | `label` |
| `sysucd` | `/home/xxj/Data/SYSU_CD` | `time1` | `time2` | `label` |

## Pretrained weights

Training uses `ckpt/gcc-b_in1k.pth` by default. The checked-in README names this file but does not provide a download URL, and the repository contains no checkpoint. If the authors' intended GCC-B ImageNet-1K checkpoint becomes available, place it at that path or set `PRETRAINED=/path/to/gcc-b_in1k.pth`. To explicitly train from random initialization without the checkpoint, set `PRETRAINED=none`; the architecture and training hyperparameters remain unchanged, but the result is not the pretrained configuration reported by the paper.

The README also shows a benchmark checkpoint named `ckpt/changegcc-b_levir.pth`, but gives no download URL. For testing, pass an available checkpoint explicitly.

## Train

```bash
bash run_train.sh levircd
bash run_train.sh levircdplus
bash run_train.sh cdd
bash run_train.sh whucd
bash run_train.sh sysucd
```

The wrapper preserves code defaults: base model, batch size 32, 200 epochs, Adam, learning rate `1e-5`, cosine schedule, 20 warmup epochs, and weight decay `1e-4`. The WHU validation loader uses batch size 4 in the upstream code. Runtime settings default to one visible GPU and 8 workers; select GPU 1 with `GPU_IDS=1`, or both GPUs with `GPU_IDS=0,1 DEVICES=2`. Set `BATCH_SIZE=8` or `BATCH_SIZE=4` if GPU memory is insufficient; the default remains 32.

Checkpoints, `res.out`, and per-epoch XLSX logs are written below `ChangeGCC_work_dirs/<DatasetName>/ChangeGCC-b/`. The training code overwrites `last.pth` every epoch and saves numbered snapshots from epoch 30 onward. It keeps the two newest numbered snapshots plus `last.pth`, at most three weight files; it does not select or save a best-validation checkpoint.

## Test

```bash
bash run_test.sh levircd ChangeGCC_work_dirs/LEVIR/ChangeGCC-b/last.pth
```

Replace the dataset key and checkpoint path as needed. Test inference uses batch size 1 and reports the repository's binary segmentation metrics.
