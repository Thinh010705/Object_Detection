import json
import os
import random
from collections import defaultdict

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from .box_ops import xyxy_to_cxcywh


DEFAULT_CLASSES = ["person", "car", "dog", "cat", "chair"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_annotation_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    classes = data.get("classes", DEFAULT_CLASSES)
    grouped = defaultdict(list)
    for ann in data.get("annotations", []):
        grouped[ann["image_id"]].append(ann)
    return data.get("images", []), grouped, classes


def resolve_image_path(image_dir, file_name):
    candidates = [
        os.path.join(image_dir, os.path.basename(file_name)),
        os.path.join(image_dir, file_name),
        file_name,
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def pil_to_normalized_tensor(image):
    data = torch.as_tensor(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.view(image.size[1], image.size[0], 3).permute(2, 0, 1).float() / 255.0
    return (data - MEAN) / STD


def resize_image_and_boxes(image, boxes, image_size):
    orig_w, orig_h = image.size
    image = image.resize((image_size, image_size), Image.BILINEAR)
    if boxes.numel() > 0:
        scale = torch.tensor([image_size / orig_w, image_size / orig_h, image_size / orig_w, image_size / orig_h])
        boxes = boxes * scale
    return image, boxes


def horizontal_flip(image, boxes):
    w, _ = image.size
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if boxes.numel() > 0:
        x1 = boxes[:, 0].clone()
        x2 = boxes[:, 2].clone()
        boxes[:, 0] = w - x2
        boxes[:, 2] = w - x1
    return image, boxes


def random_color_jitter(image):
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = random.uniform(0.75, 1.25)
        image = enhancer_cls(image).enhance(factor)
    return image


def random_crop(image, boxes, labels, min_scale=0.75):
    if boxes.numel() == 0 or random.random() > 0.35:
        return image, boxes, labels

    w, h = image.size
    crop_w = random.randint(int(w * min_scale), w)
    crop_h = random.randint(int(h * min_scale), h)
    left = random.randint(0, w - crop_w)
    top = random.randint(0, h - crop_h)
    crop = torch.tensor([left, top, left + crop_w, top + crop_h], dtype=torch.float32)

    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    keep = (
        (centers[:, 0] >= crop[0])
        & (centers[:, 0] <= crop[2])
        & (centers[:, 1] >= crop[1])
        & (centers[:, 1] <= crop[3])
    )
    if keep.sum() == 0:
        return image, boxes, labels

    image = image.crop((left, top, left + crop_w, top + crop_h))
    boxes = boxes[keep].clone()
    labels = labels[keep]
    boxes[:, 0::2] = boxes[:, 0::2].clamp(left, left + crop_w) - left
    boxes[:, 1::2] = boxes[:, 1::2].clamp(top, top + crop_h) - top
    valid = (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
    return image, boxes[valid], labels[valid]


def encode_targets(boxes, labels, image_size, grid_size, num_classes):
    obj = torch.zeros((grid_size, grid_size), dtype=torch.float32)
    cls = torch.zeros((grid_size, grid_size), dtype=torch.long)
    box = torch.zeros((grid_size, grid_size, 4), dtype=torch.float32)
    area = torch.zeros((grid_size, grid_size), dtype=torch.float32)

    if boxes.numel() == 0:
        return {"objectness": obj, "classes": cls, "boxes": box}

    norm_xyxy = boxes / float(image_size)
    cxcywh = xyxy_to_cxcywh(norm_xyxy).clamp(0, 1)
    for b, label in zip(cxcywh, labels):
        cx, cy, bw, bh = b.tolist()
        gx = min(grid_size - 1, max(0, int(cx * grid_size)))
        gy = min(grid_size - 1, max(0, int(cy * grid_size)))
        this_area = bw * bh
        if obj[gy, gx] == 1 and this_area <= area[gy, gx]:
            continue
        obj[gy, gx] = 1
        cls[gy, gx] = int(label)
        box[gy, gx] = torch.tensor([cx * grid_size - gx, cy * grid_size - gy, bw, bh])
        area[gy, gx] = this_area
    return {"objectness": obj, "classes": cls, "boxes": box}


class DetectionDataset(Dataset):
    def __init__(self, annotation_file, image_dir, image_size=416, grid_size=13, augment=False, classes=None):
        self.images, self.annotations, file_classes = load_annotation_file(annotation_file)
        self.classes = classes or file_classes
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.image_dir = image_dir
        self.image_size = image_size
        self.grid_size = grid_size
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        image_id = info["id"]
        image = Image.open(resolve_image_path(self.image_dir, info["file_name"])).convert("RGB")
        boxes = []
        labels = []
        for ann in self.annotations.get(image_id, []):
            if ann["class"] not in self.class_to_idx:
                continue
            boxes.append(ann["bbox"])
            labels.append(self.class_to_idx[ann["class"]])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        if self.augment:
            image, boxes, labels = random_crop(image, boxes, labels)
            if random.random() < 0.5:
                image, boxes = horizontal_flip(image, boxes)
            if random.random() < 0.8:
                image = random_color_jitter(image)

        image, boxes = resize_image_and_boxes(image, boxes, self.image_size)
        tensor = pil_to_normalized_tensor(image)
        target = encode_targets(boxes, labels, self.image_size, self.grid_size, len(self.classes))
        target["image_id"] = image_id
        target["orig_size"] = torch.tensor([info["width"], info["height"]], dtype=torch.float32)
        return tensor, target


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = {
        "objectness": torch.stack([item[1]["objectness"] for item in batch]),
        "classes": torch.stack([item[1]["classes"] for item in batch]),
        "boxes": torch.stack([item[1]["boxes"] for item in batch]),
        "image_id": [item[1]["image_id"] for item in batch],
        "orig_size": torch.stack([item[1]["orig_size"] for item in batch]),
    }
    return images, targets
