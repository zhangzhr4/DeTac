# DeTac

**DeTac: Dual-branch Decoupling of Texture and Force Signals in Vision-Based Tactile Sensors**

Zhirui Zhang, Yunkai Tang, Jiahang Li, Fan Fei, Jinnuo Zhang, Kairan Ding, Jieji Ren, Yandong Guo, and Boxin Shi

Conference on Robot Learning (CoRL), 2026 — **Spotlight**

DeTac recovers a clean shading image and a marker image from a single mixed tactile observation. This repository provides the training and inference code for our method.

## Installation

Using a recent Conda installation, create an environment and install the dependencies:

```bash
conda create -n detac python=3.11 pip -y
conda activate detac
python -m pip install -r requirements.txt --index-url https://pypi.org/simple
```

A CUDA-capable NVIDIA GPU is required for the training command below. Run all commands from the repository root.

## Data

Download `detac.zip` from [Google Drive](https://drive.google.com/drive/folders/1b3sS9jP2fdjtvAYCslPzPfazlREPQaSC?usp=sharing). Extract it and place the three image folders directly under `data/detac/` (move them out of any extra parent folder in the archive):

```text
data/detac/
├── marker/        # Mixed tactile images
├── nomarker/      # Paired clean shading images
└── mask/          # Marker masks
```

Generate the train, validation, and test splits before training:

```bash
python prepare_data.py --data-root data/detac
```

This generates `manifests/` with the default train, validation, and test splits. Run this once after extraction, then follow the training command below. If you store the dataset elsewhere, use that path for `--data-root`.

### Preparing your own data

Collecting new paired data requires the custom sensor and mechanical alignment setup described in the paper. First capture marker-free shading images, then fabricate markers on the same sensor and repeat the same contact poses to capture aligned mixed images. Generate marker masks from the mixed images using manually annotated marker-center prompts and SAM, as described in the paper. After alignment and preprocessing, use the same three-folder layout as the released dataset. Matching images and masks must share a filename stem, such as `Obj_01__0000` (object ID followed by frame number).

```text
data/custom/
├── marker/Obj_01__0000.jpg
├── nomarker/Obj_01__0000.jpg
└── mask/Obj_01__0000.png
```

Generate splits with the same preparation command:

```bash
python prepare_data.py --data-root data/custom
```

## Training

```bash
python train.py \
  --data-root data/detac \
  --output-dir outputs/detac
```

Training runs for 300 epochs. Checkpoints and loss history are saved in `outputs/detac/`. To resume an interrupted run:

```bash
python train.py \
  --data-root data/detac \
  --output-dir outputs/detac \
  --resume
```

## Prediction

Predict the test split using the trained checkpoint:

```bash
python infer.py \
  --checkpoint outputs/detac/best.pt \
  --data-root data/detac --split test \
  --output-dir outputs/test_predictions
```

For a single image or a directory of mixed images, replace `--data-root data/detac --split test` with `--input /path/to/image_or_directory`.

To predict the validation split, use `--split val --output-dir outputs/val_predictions` instead.

Results are saved under `shading/`, `markers/`, and `binary_masks/`. The `panels/` directory shows the input, restored shading, and predicted markers side by side.

For dataset splits with ground truth, inference also reports grayscale PSNR for shading restoration and IoU for marker masks. Per-image scores and their averages are saved in `summary.json`. Using `--input` only saves predictions, since no ground truth is provided.

## Citation

```bibtex
@misc{zhang2026detacdualbranchdecouplingtexture,
      title={DeTac: Dual-branch Decoupling of Texture and Force Signals in Vision-Based Tactile Sensors},
      author={Zhirui Zhang and Yunkai Tang and Jiahang Li and Fan Fei and Jinnuo Zhang and Kairan Ding and Jieji Ren and Yandong Guo and Boxin Shi},
      year={2026},
      eprint={2610.16463},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2610.16463},
}
```
