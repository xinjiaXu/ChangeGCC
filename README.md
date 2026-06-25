<h1 align="center">ChangeGCC (ECCV'26)</h1>

<h3 align="center">Beyond Attention: Convolutional Global Context for Remote Sensing Change Detection</h3>

## 🔭Overview

![ChangeGCC](ChangeGCC.png)

## 🗝️Quick start on LEVIR dataset

1. Data preparation:

   - Download LEVIR dataset from [Here](https://justchenhao.github.io/LEVIR/) (Official) or [AI Studio](https://aistudio.baidu.com/datasetdetail/53795) (Spare).

   - Crop the images & masks into $256\times 256$ size.

   - Arrange them into the following format:

     ```
     data
     |- LEVIR
        |- ABlabel
           |- A
              |- train_1_0_256_0_256.png
              |- ...
              |- val_1_0_256_0_256.png
              |- ...
              |- test_1_0_256_0_256.png
              |- ...
           |- B
              |- (same as above)
           |- label
              |- (same as above)
           |- list
              |- train.txt
              |- val.txt
              |- test.txt
     ```
     
     Here, each line of .txt file is a file name, like:
     
     ```
     train_1_0_256_0_256.png
     train_1_0_256_256_512.png
     train_1_0_256_512_768.png
     train_1_0_256_768_1024.png
     ...
     ```

2. Installation:

   - Create and activate a new conda environment:

     ```bash
     conda create -n changegcc python=3.10
     conda activate changegcc
     ```

   - Install PyTorch according to CUDA or ROCm version (taking CUDA 13.0 as an example here): 

     ```bash
     pip3 install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
     ```
   
   - Install other dependencies:
   
     ```bash
     pip install -r requirements.txt
     ```
   
4. Train:

   ```bash
   python train.py --model_size 'base' --pretrain_dir 'ckpt/gcc-b_in1k.pth'
   ```

5. Benchmark:

   ```bash
   python test.py --model_size 'base' --model_path 'ckpt/changegcc-b_levir.pth'
   ```

## 🤝Acknowledgments
This project is based on [AFCF3D-Net](https://github.com/wm-Githuber/AFCF3D-Net) and [ChangeTitans](https://github.com/NUST-Machine-Intelligence-Laboratory/ChangeTitans). Thanks for their excellent works!!
