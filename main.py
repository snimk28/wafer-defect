"""
Wafer Defect Detection - Main Entry Point

Usage:
    python main.py --mode train
    python main.py --mode train --model resnet18
    python main.py --mode eval  --checkpoint outputs/checkpoints/best_model.pt
    python main.py --mode demo  --checkpoint outputs/checkpoints/best_model.pt
"""
import argparse
import sys
from pathlib import Path

import torch

from src.utils import load_config, set_seed, get_device, count_parameters, load_checkpoint
from src.dataset import (
    load_wm811k, build_dataloaders, WaferDataset,
    DEFECT_CLASSES, plot_class_distribution, plot_wafer_samples
)
from src.model import build_model, build_criterion
from src.train import train
from src.evaluate import evaluate, visualize_predictions


def parse_args():
    p = argparse.ArgumentParser(description="Wafer Defect Detection")
    p.add_argument("--mode", choices=["train", "eval", "demo"], default="train")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--model", default=None,
                   help="Override model name: cnn | resnet18 | efficientnet_b0")
    p.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint for eval/demo")
    p.add_argument("--data", default=None, help="Override data path to LSWMD.pkl")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.model:
        cfg["model"]["name"] = args.model
    if args.data:
        cfg["data"]["raw_path"] = args.data

    set_seed(cfg["seed"])
    device = get_device(cfg)
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    data_path = cfg["data"]["raw_path"]
    if not Path(data_path).exists():
        print(f"\n[ERROR] Dataset not found at '{data_path}'")
        print("\nTo download the WM-811K dataset:")
        print("  1. Go to https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map")
        print("  2. Download 'LSWMD.pkl'")
        print("  3. Place it at:  data/raw/LSWMD.pkl")
        print("\nOr run the demo with synthetic data:")
        print("  python demo_synthetic.py")
        sys.exit(1)

    print(f"\nLoading dataset from {data_path} ...")
    df = load_wm811k(data_path, use_none_class=cfg["data"]["use_none_class"])

    use_none = cfg["data"]["use_none_class"]
    class_names = DEFECT_CLASSES if use_none else DEFECT_CLASSES[:-1]
    cfg["model"]["num_classes"] = len(class_names)
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Total samples: {len(df)}")
    print(df["label_name"].value_counts().to_string())

    # ── Build data loaders ────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, train_df, val_df, test_df = \
        build_dataloaders(df, cfg)

    print(f"\nSplit sizes → train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    print(f"\nModel: {cfg['model']['name']}  |  Parameters: {count_parameters(model):,}")

    if args.mode == "train":
        # plot EDA figures before training
        fig_dir = cfg["output"]["figure_dir"]
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        plot_class_distribution(df, save_path=f"{fig_dir}/class_distribution.png")
        plot_wafer_samples(df, save_path=f"{fig_dir}/wafer_samples.png")

        criterion = build_criterion(cfg)
        history, best_ckpt = train(model, train_loader, val_loader, criterion, cfg, device)

        print(f"\nLoading best checkpoint for final test evaluation ...")
        load_checkpoint(model, best_ckpt, device)
        evaluate(model, test_loader, class_names, device,
                 figure_dir=cfg["output"]["figure_dir"])

    elif args.mode == "eval":
        ckpt = args.checkpoint or f"{cfg['output']['checkpoint_dir']}/best_model.pt"
        load_checkpoint(model, ckpt, device)
        evaluate(model, test_loader, class_names, device,
                 figure_dir=cfg["output"]["figure_dir"])

    elif args.mode == "demo":
        ckpt = args.checkpoint or f"{cfg['output']['checkpoint_dir']}/best_model.pt"
        load_checkpoint(model, ckpt, device)
        test_ds = WaferDataset(test_df, cfg["data"]["image_size"], augment=False)
        visualize_predictions(model, test_ds, class_names, device,
                              n_samples=16,
                              save_path=f"{cfg['output']['figure_dir']}/predictions.png")


if __name__ == "__main__":
    main()
