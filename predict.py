import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image

from models.detector import build_detector, decode_predictions
from utils.dataset import DEFAULT_CLASSES, pil_to_normalized_tensor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run object detection inference.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="./models/best.pth")
    parser.add_argument("--conf_threshold", type=float, default=0.05)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--max_detections", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def list_images(image_dir):
    paths = []
    for entry in sorted(os.listdir(image_dir)):
        path = os.path.join(image_dir, entry)
        if os.path.isfile(path) and Path(entry).suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
    return paths


def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train first or pass --checkpoint.")
    ckpt = torch.load(checkpoint_path, map_location=device)
    classes = ckpt.get("classes", DEFAULT_CLASSES)
    image_size = int(ckpt.get("image_size", 416))
    grid_size = int(ckpt.get("grid_size", 13))
    model_width = int(ckpt.get("model_width", 24))
    backbone = ckpt.get("backbone", "tiny")
    model = build_detector(
        backbone,
        num_classes=len(classes),
        grid_size=grid_size,
        model_width=model_width,
        pretrained_backbone=False,
        freeze_backbone=bool(ckpt.get("freeze_backbone", False)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, classes, image_size


def round_box(box):
    x1, y1, x2, y2 = box
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    model, classes, image_size = load_model(args.checkpoint, device)

    results = []
    for image_path in list_images(args.image_dir):
        image_id = os.path.basename(image_path)
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        resized = image.resize((image_size, image_size), Image.BILINEAR)
        tensor = pil_to_normalized_tensor(resized).unsqueeze(0).to(device)
        pred = model(tensor)[0]
        detections = decode_predictions(
            pred,
            image_width=orig_w,
            image_height=orig_h,
            classes=classes,
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold,
            max_detections=args.max_detections,
        )
        boxes = []
        for det in detections:
            boxes.append(
                {
                    "class": det["class"],
                    "confidence": round(float(det["confidence"]), 6),
                    "bbox": round_box(det["bbox"]),
                }
            )
        results.append({"image_id": image_id, "boxes": boxes})

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
