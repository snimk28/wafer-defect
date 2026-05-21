# Wafer Defect Detection with Deep Learning

Classify semiconductor wafer defect patterns using CNN and ResNet18 on the WM-811K dataset.

## Defect Classes (8 types)

| Class | Description |
|---|---|
| Center | Defects clustered at wafer center |
| Donut | Ring-shaped defect band |
| Edge-Loc | Localized defects at wafer edge |
| Edge-Ring | Full edge ring defects |
| Loc | Localized cluster anywhere |
| Near-full | Almost entire wafer defective |
| Random | Randomly scattered defects |
| Scratch | Scratch/line defects |

## Quick Start

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Run demo with synthetic data (no download needed)
```
python demo_synthetic.py
```

### 3. Train on real WM-811K data

Download `LSWMD.pkl` from:
https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map

Place it at `data/raw/LSWMD.pkl`, then:
```
python main.py --mode train
python main.py --mode train --model resnet18
python main.py --mode train --model efficientnet_b0
```

### 4. Evaluate / Demo
```
python main.py --mode eval
python main.py --mode demo
```

## Project Structure

```
wafer defect/
├── src/
│   ├── dataset.py      # data loading, augmentation, visualization
│   ├── model.py        # CNN, ResNet18, EfficientNet-B0
│   ├── train.py        # training loop, early stopping
│   ├── evaluate.py     # metrics, confusion matrix, prediction viz
│   └── utils.py        # config, seed, device helpers
├── configs/
│   └── config.yaml     # all hyperparameters
├── notebooks/
│   └── wafer_defect_exploration.ipynb
├── outputs/
│   ├── checkpoints/    # saved model weights
│   ├── figures/        # plots
│   └── logs/           # training history JSON
├── data/
│   └── raw/            # put LSWMD.pkl here
├── main.py
├── demo_synthetic.py
└── requirements.txt
```

## Configuration (`configs/config.yaml`)

Key settings to change:

| Setting | Options | Default |
|---|---|---|
| `model.name` | `cnn`, `resnet18`, `efficientnet_b0` | `resnet18` |
| `model.pretrained` | `true`, `false` | `true` |
| `data.image_size` | `32`, `64`, `128` | `64` |
| `training.epochs` | any | `50` |
| `training.batch_size` | any | `64` |
| `training.use_weighted_sampler` | `true`, `false` | `true` |
| `training.scheduler` | `cosine`, `step`, `none` | `cosine` |

## Model Comparison (on WM-811K, 8 classes)

| Model | Params | Expected Accuracy |
|---|---|---|
| WaferCNN (custom) | ~1.2M | ~92% |
| ResNet18 (pretrained) | ~11M | ~96% |
| EfficientNet-B0 (pretrained) | ~5M | ~95% |

## Outputs

After training:
- `outputs/checkpoints/best_model.pt` — best checkpoint
- `outputs/figures/confusion_matrix.png` — per-class performance
- `outputs/figures/training_curves.png` — loss & accuracy curves
- `outputs/figures/class_distribution.png` — dataset balance
- `outputs/figures/wafer_samples.png` — sample wafer maps
- `outputs/logs/history.json` — epoch-by-epoch metrics
