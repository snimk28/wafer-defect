"""
Quick demo using synthetically generated wafer maps.
Run this to verify the whole pipeline works before downloading the real dataset.

    python demo_synthetic.py
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.utils import load_config, set_seed, get_device, count_parameters, load_checkpoint
from src.dataset import DEFECT_CLASSES, WaferDataset, build_dataloaders
from src.model import build_model, build_criterion
from src.train import train
from src.evaluate import evaluate, visualize_predictions


# ── Synthetic wafer map generators ───────────────────────────────────────────

def _base(size=26):
    """Circular wafer template: 1=normal, 0=outside."""
    m = np.zeros((size, size), dtype=np.uint8)
    cx, cy, r = size // 2, size // 2, size // 2 - 1
    for i in range(size):
        for j in range(size):
            if (i - cx) ** 2 + (j - cy) ** 2 <= r ** 2:
                m[i, j] = 1
    return m


def _gen(pattern: str, size=26, rng=None) -> np.ndarray:
    """Generate a synthetic wafer map for a given defect pattern."""
    if rng is None:
        rng = np.random.default_rng()
    m = _base(size).copy()
    cx, cy, r = size // 2, size // 2, size // 2 - 1

    if pattern == "Center":
        cr = int(r * 0.35)
        for i in range(size):
            for j in range(size):
                if m[i, j] == 1 and (i - cx) ** 2 + (j - cy) ** 2 <= cr ** 2:
                    if rng.random() < 0.7:
                        m[i, j] = 2

    elif pattern == "Donut":
        inner = int(r * 0.35)
        outer = int(r * 0.75)
        for i in range(size):
            for j in range(size):
                d2 = (i - cx) ** 2 + (j - cy) ** 2
                if m[i, j] == 1 and inner ** 2 <= d2 <= outer ** 2:
                    if rng.random() < 0.6:
                        m[i, j] = 2

    elif pattern == "Edge-Ring":
        inner = int(r * 0.75)
        for i in range(size):
            for j in range(size):
                d2 = (i - cx) ** 2 + (j - cy) ** 2
                if m[i, j] == 1 and d2 >= inner ** 2:
                    if rng.random() < 0.7:
                        m[i, j] = 2

    elif pattern == "Edge-Loc":
        angle = rng.uniform(0, 2 * np.pi)
        for i in range(size):
            for j in range(size):
                if m[i, j] != 1:
                    continue
                d2 = (i - cx) ** 2 + (j - cy) ** 2
                a = np.arctan2(i - cx, j - cy)
                if d2 >= (r * 0.7) ** 2 and abs(a - angle) < 0.6:
                    if rng.random() < 0.7:
                        m[i, j] = 2

    elif pattern == "Loc":
        lx = rng.integers(2, size - 3)
        ly = rng.integers(2, size - 3)
        lr = rng.integers(2, 5)
        for i in range(size):
            for j in range(size):
                if m[i, j] == 1 and (i - lx) ** 2 + (j - ly) ** 2 <= lr ** 2:
                    if rng.random() < 0.75:
                        m[i, j] = 2

    elif pattern == "Scratch":
        x0, y0 = rng.integers(2, size - 2), rng.integers(2, size - 2)
        angle = rng.uniform(0, np.pi)
        length = rng.integers(6, size - 2)
        for t in range(length):
            xi = int(x0 + t * np.sin(angle))
            yi = int(y0 + t * np.cos(angle))
            if 0 <= xi < size and 0 <= yi < size and m[xi, yi] == 1:
                m[xi, yi] = 2
                for di in [-1, 0, 1]:
                    ni = xi + di
                    if 0 <= ni < size and m[ni, yi] == 1 and rng.random() < 0.4:
                        m[ni, yi] = 2

    elif pattern == "Random":
        for i in range(size):
            for j in range(size):
                if m[i, j] == 1 and rng.random() < 0.08:
                    m[i, j] = 2

    elif pattern == "Near-full":
        for i in range(size):
            for j in range(size):
                if m[i, j] == 1 and rng.random() < 0.85:
                    m[i, j] = 2

    return m


def make_synthetic_df(n_per_class: int = 300, size: int = 26, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    classes = DEFECT_CLASSES[:-1]  # exclude "none"
    rows = []
    for cls_i, cls in enumerate(classes):
        for _ in range(n_per_class):
            wmap = _gen(cls, size=size, rng=rng)
            rows.append({
                "waferMap": wmap,
                "label_name": cls,
                "label": cls_i,
            })
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config("configs/config.yaml")
    cfg["training"]["epochs"] = 20
    cfg["training"]["batch_size"] = 64
    cfg["model"]["name"] = "cnn"
    cfg["model"]["pretrained"] = False
    cfg["data"]["augment"] = True
    cfg["training"]["use_weighted_sampler"] = True
    cfg["model"]["num_classes"] = 8

    set_seed(cfg["seed"])
    device = get_device(cfg)
    print(f"디바이스: {device}")
    print("=" * 60)

    print("\n[1/5] 합성 웨이퍼 맵 생성 중... (클래스 8개 x 300개 = 2400개)")
    df = make_synthetic_df(n_per_class=300, size=26, seed=cfg["seed"])
    class_names = DEFECT_CLASSES[:-1]
    print("클래스별 샘플 수:")
    print(df["label_name"].value_counts().to_string())

    # EDA 그래프 (창으로 뜸 - 닫으면 다음 단계 진행)
    print("\n[2/5] 데이터 분포 시각화 (그래프 창을 닫으면 계속 진행됩니다)")
    fig_dir = cfg["output"]["figure_dir"]
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    from src.dataset import plot_class_distribution, plot_wafer_samples
    plot_class_distribution(df, title="클래스 분포",
                            save_path=f"{fig_dir}/synthetic_class_dist.png")
    plot_wafer_samples(df, n_per_class=3,
                       save_path=f"{fig_dir}/synthetic_wafer_samples.png")

    print("\n[3/5] 데이터 분할 및 이미지 준비 중...")

    train_loader, val_loader, test_loader, train_df, val_df, test_df = \
        build_dataloaders(df, cfg)
    print(f"  train: {len(train_df)}개 | val: {len(val_df)}개 | test: {len(test_df)}개")

    model = build_model(cfg).to(device)
    print(f"\n[4/5] 모델: {cfg['model']['name'].upper()}  |  파라미터 수: {count_parameters(model):,}")

    criterion = build_criterion(cfg)
    history, best_ckpt = train(model, train_loader, val_loader, criterion, cfg, device)

    print("\n[5/5] 최고 모델 불러와서 테스트 평가 중...")
    load_checkpoint(model, best_ckpt, device)
    metrics = evaluate(model, test_loader, class_names, device,
                       figure_dir=fig_dir)

    test_ds = WaferDataset(test_df, cfg["data"]["image_size"], augment=False)
    visualize_predictions(model, test_ds, class_names, device,
                          n_samples=16,
                          save_path=f"{fig_dir}/synthetic_predictions.png")

    print("\n" + "=" * 60)
    print("완료! 결과 파일 저장 위치: outputs/figures/")
    print("  - confusion_matrix.png    : 클래스별 정확도")
    print("  - training_curves.png     : 학습 곡선")
    print("  - synthetic_predictions.png : 예측 결과 샘플")


if __name__ == "__main__":
    main()
