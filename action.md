# ChangeGCC Server Actions

Server checkout: `/home/xxj/code/DINOCLIP_comparative_test/ChangeGCC`  
Dataset root: `/home/xxj/Data`

## Pull reproduce branch

First time:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/ChangeGCC
git fetch origin
git switch --track -c reproduce origin/reproduce
```

Later updates:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/ChangeGCC
git switch reproduce
git pull --ff-only
```

## Environment and pretrained weight

```bash
bash setup_server_env.sh
conda activate changegcc-rs
bash prepare_server_data.sh /home/xxj/Data
```

## Prepare raw 1024x1024 LEVIR-CD

The upstream ChangeGCC README requires LEVIR images and masks to be cropped into aligned 256x256 patches before training or evaluation. If `/home/xxj/Data/LEVIR_CD` contains the original 1024x1024 images, create a separate cropped copy; the script never modifies its source directory:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/ChangeGCC
conda activate changegcc-rs
python prepare_levir_patches.py
```

By default, patches are written to `/home/xxj/Data/ChangeGCC_data/LEVIR_CD`, preserving `train`, `val`, and `test` splits. Override `--source` or `--output` if your paths differ. After verifying the patch sizes and counts, use `DATA_ROOT=/home/xxj/Data/ChangeGCC_data` for LEVIR-CD commands so the existing launcher selects the cropped copy.

Training uses `ckpt/gcc-b_in1k.pth` by default. The repository README names it but does not give a download URL. If you obtain the authors' GCC-B ImageNet-1K checkpoint, place it at that path or set `PRETRAINED=/path/to/gcc-b_in1k.pth`. To train from random initialization without it, prefix the command with `PRETRAINED=none`. This is a scratch-trained ChangeGCC-B result, not the pretrained configuration reported by the paper.

## Model complexity

After activating `changegcc-rs`, count the base model with one paired 256x256 input. FLOPs and parameter counts do not require the GCC-B checkpoint:

```bash
python -m pip install -r requirements_profile.txt
CUDA_VISIBLE_DEVICES=0 python profile_model.py --model-size base --device cuda:0 \
  --output ChangeGCC_work_dirs/profile_256.json
```

The JSON lists total/trainable parameters, fvcore-counted operations, and unsupported operators. The FLOP total is incomplete if unsupported operators are listed.

## Train

```bash
bash run_train.sh levircd
bash run_train.sh levircdplus
bash run_train.sh cdd
bash run_train.sh whucd
bash run_train.sh sysucd
```

For example, train CDD from scratch on GPU 1:

```bash
PRETRAINED=none GPU_IDS=1 NUM_WORKERS=4 bash run_train.sh cdd
```

Train LEVIR-CD from scratch using the 256x256 patches above:

```bash
DATA_ROOT=/home/xxj/Data/ChangeGCC_data PRETRAINED=none GPU_IDS=1 DEVICES=1 \
  NUM_WORKERS=4 BATCH_SIZE=2 bash run_train.sh levircd
```

If GPU memory is insufficient, lower only the batch size, for example `BATCH_SIZE=8` or `BATCH_SIZE=4`. The default remains 32. Training output paths are `ChangeGCC_work_dirs/<DatasetName>/ChangeGCC-b/`; for CDD this is `ChangeGCC_work_dirs/CDD/ChangeGCC-b/`.

Defaults: base model, batch 32, 200 epochs, Adam, lr `1e-5`, cosine scheduler, 20 warmup epochs, weight decay `1e-4`. For GPU 1 use `GPU_IDS=1`; for both use `GPU_IDS=0,1 DEVICES=2`.

## Test

Training overwrites `last.pth` every epoch and saves numbered snapshots from epoch 30 onward. It keeps only the two newest numbered snapshots alongside `last.pth`, for a maximum of three weight files in the run directory. Pass `last.pth` to the test launcher:

```bash
bash run_test.sh levircd ChangeGCC_work_dirs/LEVIR/ChangeGCC-b/last.pth
bash run_test.sh levircdplus ChangeGCC_work_dirs/LEVIR+/ChangeGCC-b/last.pth
bash run_test.sh cdd ChangeGCC_work_dirs/CDD/ChangeGCC-b/last.pth
bash run_test.sh whucd ChangeGCC_work_dirs/WHU/ChangeGCC-b/last.pth
bash run_test.sh sysucd ChangeGCC_work_dirs/SYSU/ChangeGCC-b/last.pth
```

For the cropped LEVIR-CD copy, set the same `DATA_ROOT`:

```bash
DATA_ROOT=/home/xxj/Data/ChangeGCC_data GPU_IDS=1 NUM_WORKERS=4 \
  bash run_test.sh levircd ChangeGCC_work_dirs/LEVIR/ChangeGCC-b/last.pth
```

Evaluation uses batch size 1 and prints the model's binary change metrics. Outputs and XLSX logs are under `ChangeGCC_work_dirs/<DatasetName>/ChangeGCC-b/`.
