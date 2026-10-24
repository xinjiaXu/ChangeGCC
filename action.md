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

Training requires `ckpt/gcc-b_in1k.pth`. The repository README names this checkpoint but does not give a download URL. Obtain the authors' GCC-B ImageNet-1K weight and place it at that path, or set `PRETRAINED=/path/to/gcc-b_in1k.pth`. The launcher exits if it is missing.

## Train

```bash
bash run_train.sh levircd
bash run_train.sh levircdplus
bash run_train.sh cdd
bash run_train.sh whucd
bash run_train.sh sysucd
```

Defaults: base model, batch 32, 200 epochs, Adam, lr `1e-5`, cosine scheduler, 20 warmup epochs, weight decay `1e-4`. For GPU 1 use `GPU_IDS=1`; for both use `GPU_IDS=0,1 DEVICES=2`.

## Test

Training saves `last.pth` each epoch. Pass the corresponding checkpoint to the test launcher:

```bash
bash run_test.sh levircd ChangeGCC_work_dirs/levircd/LEVIR/ChangeGCC-b/last.pth
bash run_test.sh levircdplus ChangeGCC_work_dirs/levircdplus/LEVIR+/ChangeGCC-b/last.pth
bash run_test.sh cdd ChangeGCC_work_dirs/cdd/CDD/ChangeGCC-b/last.pth
bash run_test.sh whucd ChangeGCC_work_dirs/whucd/WHU/ChangeGCC-b/last.pth
bash run_test.sh sysucd ChangeGCC_work_dirs/sysucd/SYSU/ChangeGCC-b/last.pth
```

Evaluation uses batch size 1 and prints the model's binary change metrics. Outputs and XLSX logs are under `ChangeGCC_work_dirs/<dataset>/`.
