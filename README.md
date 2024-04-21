# NAID

Official Code of [NIR-Assisted Image Denoising: A Selective Fusion Approach and A Real-World Benchmark Dataset](https://arxiv.org/abs/2404.08514)

## Preparation and Dataset
* Prerequisites
  - Python 3.x and PyTorch 1.12.
  - OpenCV, NumPy, Pillow, tqdm, lpips, einops, scikit-image and tensorboardX.
* Dataset
  - Real-NAID dataset can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1BCdFnxOCweIZiZv5t2ildQ) (password: ik9u)
## Quick Start
* Testing
  - Download pre-trained models from [Baidu Netdisk](https://pan.baidu.com/s/1ZMi6zpGTL9ByCZwfhyq1BA) (password: jcdb) and put the pre-trained checkpoint with the corresponding folder under `./ckpt/` folder
  - Download Real-NAID dataset and modify `dataroot` in `test.sh`
  - Modify `name`, `model` in `test.sh` and then run
    ```
    sh test.sh
    ```
* Training
  - Download Real-NAID dataset and modify `dataroot` in `test.sh`
  - Modify `name`, `model` and then run
    ```
    sh train.sh
      ```
* Note
  - You can specify which GPU to use by `--gpu_ids`, e.g., `--gpu_ids 0`, `--gpu_ids 0,1`, `--gpu_ids -1` (for CPU mode). In the default setting, all GPUs are used.
  - You can refer to [options]() for more arguments.
    
