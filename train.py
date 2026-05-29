import argparse
import importlib.util
import json
import os
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models.detector import build_detector, decode_predictions, detection_loss
from utils.dataset import DEFAULT_CLASSES, DetectionDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train a custom grid object detector.")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--val_image_dir", required=True)
    parser.add_argument("--checkpoint_dir", default="./models/")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=416)
    parser.add_argument("--grid_size", type=int, default=13)
    parser.add_argument("--backbone", choices=["tiny", "resnet50"], default="tiny")
    parser.add_argument("--model_width", type=int, default=24)
    parser.add_argument("--no_pretrained_backbone", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--conf_threshold", type=float, default=0.05)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--max_detections", type=int, default=100)
    parser.add_argument("--score_every", type=int, default=1)
    parser.add_argument("--limit_train", type=int)
    parser.add_argument("--limit_val", type=int)
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


def load_evaluator():
    path = Path("public/tools/evaluate_predictions.py")
    spec = importlib.util.spec_from_file_location("evaluate_predictions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def evaluate_map(model, loader, device, classes, val_data, args):
    model.eval()
    predictions = []
    for images, targets in loader:
        images = images.to(device)
        outputs = model(images)
        for pred, image_id, orig_size in zip(outputs, targets["image_id"], targets["orig_size"]):
            image_width, image_height = [int(v) for v in orig_size.tolist()]
            detections = decode_predictions(
                pred,
                image_width=image_width,
                image_height=image_height,
                classes=classes,
                conf_threshold=args.conf_threshold,
                nms_threshold=args.nms_threshold,
                max_detections=args.max_detections,
            )
            boxes = []
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                boxes.append(
                    {
                        "class": det["class"],
                        "confidence": round(float(det["confidence"]), 6),
                        "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                    }
                )
            predictions.append({"image_id": image_id, "boxes": boxes})

    evaluator = load_evaluator()
    ground_truth = evaluator.load_json(Path(val_data))
    eval_classes, image_info = evaluator.validate_ground_truth(ground_truth)
    require_complete = len(predictions) == len(image_info)
    normalized = evaluator.normalize_predictions(
        predictions,
        classes=eval_classes,
        image_info=image_info,
        max_detections_per_image=args.max_detections,
        require_complete=require_complete,
    )
    return evaluator.evaluate(ground_truth, normalized, eval_classes, 0.5), predictions


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
    if args.limit_train is not None:
        train_ds.images = train_ds.images[: args.limit_train]
    if args.limit_val is not None:
        val_ds.images = val_ds.images[: args.limit_val]

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
    model = build_detector(
        args.backbone,
        num_classes=len(train_ds.classes),
        grid_size=args.grid_size,
        model_width=args.model_width,
        pretrained_backbone=not args.no_pretrained_backbone,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_map = -1.0
    best_loss = float("inf")
    best_path = os.path.join(args.checkpoint_dir, "best.pth")
    last_path = os.path.join(args.checkpoint_dir, "last.pth")
    best_score_path = os.path.join(args.checkpoint_dir, "best_val_score.json")
    best_pred_path = os.path.join(args.checkpoint_dir, "best_val_predictions.json")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, len(train_ds.classes), train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, len(train_ds.classes), train=False)
        scheduler.step()

        checkpoint = {
            "model": model.state_dict(),
            "classes": train_ds.classes,
            "image_size": args.image_size,
            "grid_size": args.grid_size,
            "backbone": args.backbone,
            "model_width": args.model_width,
            "pretrained_backbone": not args.no_pretrained_backbone,
            "freeze_backbone": args.freeze_backbone,
            "epoch": epoch,
            "val_loss": val_metrics["loss"],
            "conf_threshold": args.conf_threshold,
            "nms_threshold": args.nms_threshold,
            "max_detections": args.max_detections,
        }

        map_metrics = None
        if args.score_every > 0 and epoch % args.score_every == 0:
            map_metrics, predictions = evaluate_map(model, val_loader, device, train_ds.classes, args.val_data, args)
            checkpoint["mAP@0.5"] = map_metrics["mAP@0.5"]
            checkpoint["performance_points"] = map_metrics["performance_points"]

        torch.save(checkpoint, last_path)
        if map_metrics and map_metrics["mAP@0.5"] > best_map:
            best_map = map_metrics["mAP@0.5"]
            torch.save(checkpoint, best_path)
            with open(best_score_path, "w", encoding="utf-8") as f:
                json.dump(map_metrics, f, ensure_ascii=False, indent=2)
            with open(best_pred_path, "w", encoding="utf-8") as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)
        elif not map_metrics and val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save(checkpoint, best_path)

        score_text = ""
        if map_metrics:
            score_text = f" mAP@0.5={map_metrics['mAP@0.5']:.6f} points={map_metrics['performance_points']}"
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"obj={val_metrics['obj']:.4f} cls={val_metrics['cls']:.4f} box={val_metrics['box']:.4f}"
            f"{score_text}"
        )

    if best_map >= 0:
        print(f"saved best checkpoint to {best_path} with best mAP@0.5={best_map:.6f}")
    else:
        print(f"saved best checkpoint to {best_path} with best val_loss={best_loss:.6f}")


if __name__ == "__main__":
    main()
