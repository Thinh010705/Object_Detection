import argparse
import os
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models.detector import TinyGridDetector, detection_loss
from utils.dataset import DEFAULT_CLASSES, DetectionDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small from-scratch grid object detector.")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--val_image_dir", required=True)
    parser.add_argument("--checkpoint_dir", default="./models/")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=416)
    parser.add_argument("--grid_size", type=int, default=13)
    parser.add_argument("--model_width", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_targets(targets, device):
    return {
        "objectness": targets["objectness"].to(device),
        "classes": targets["classes"].to(device),
        "boxes": targets["boxes"].to(device),
    }


def run_epoch(model, loader, optimizer, device, num_classes, train=True):
    model.train(train)
    totals = {"loss": 0.0, "obj": 0.0, "cls": 0.0, "box": 0.0}
    count = 0

    for images, targets in loader:
        images = images.to(device)
        loss_targets = move_targets(targets, device)

        with torch.set_grad_enabled(train):
            pred = model(images)
            loss, parts = detection_loss(pred, loss_targets, num_classes)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = images.shape[0]
        count += batch_size
        for key in totals:
            totals[key] += parts[key] * batch_size

    return {key: value / max(1, count) for key, value in totals.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train_ds = DetectionDataset(
        args.train_data,
        args.image_dir,
        image_size=args.image_size,
        grid_size=args.grid_size,
        augment=True,
        classes=DEFAULT_CLASSES,
    )
    val_ds = DetectionDataset(
        args.val_data,
        args.val_image_dir,
        image_size=args.image_size,
        grid_size=args.grid_size,
        augment=False,
        classes=train_ds.classes,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.device.startswith("cuda"),
    )

    device = torch.device(args.device)
    model = TinyGridDetector(num_classes=len(train_ds.classes), grid_size=args.grid_size, width=args.model_width).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_val = float("inf")
    best_path = os.path.join(args.checkpoint_dir, "best.pth")
    last_path = os.path.join(args.checkpoint_dir, "last.pth")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, len(train_ds.classes), train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, len(train_ds.classes), train=False)
        scheduler.step()

        checkpoint = {
            "model": model.state_dict(),
            "classes": train_ds.classes,
            "image_size": args.image_size,
            "grid_size": args.grid_size,
            "model_width": args.model_width,
            "epoch": epoch,
            "val_loss": val_metrics["loss"],
        }
        torch.save(checkpoint, last_path)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(checkpoint, best_path)

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"obj={val_metrics['obj']:.4f} cls={val_metrics['cls']:.4f} box={val_metrics['box']:.4f}"
        )

    print(f"saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
