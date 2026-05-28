# Object Detection Submission

This submission implements a small object detector from scratch in PyTorch. It does not use YOLO, Detectron2, MMDetection, Faster R-CNN, SSD, or torchvision detection models.

## Method

- Data reader for the provided `train.json` / `val.json` format.
- Resize to a square input size and ImageNet-style pixel normalization.
- Multi-object handling by assigning each object to the grid cell containing its center.
- Augmentations: horizontal flip, random crop, and color jitter.
- CNN backbone implemented with basic convolution, batch normalization, and SiLU layers.
- Anchor-free grid detection head predicting objectness, class logits, and bounding boxes.
- Loss: BCE objectness, Cross Entropy classification, Smooth L1 + GIoU box regression.
- Inference: confidence thresholding, per-class NMS implemented in `utils/box_ops.py`, and output boxes in original image coordinates.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset so that the annotation files are available under:

```text
public/annotations/train.json
public/annotations/val.json
public/train/images/
public/val/images/
```

## Train

Required command:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

The best model is saved to:

```text
./models/best.pth
```

Useful optional arguments:

```bash
python train.py ... --epochs 80 --batch_size 8 --image_size 416 --grid_size 13 --lr 2e-4
```

## Predict

Required command:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

By default, `predict.py` loads `./models/best.pth`. To use another checkpoint:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json \
  --checkpoint ./models/best.pth
```

The output JSON format is:

```json
[
  {
    "image_id": "img_7fd91a4c2e30.jpg",
    "boxes": [
      {
        "class": "person",
        "confidence": 0.91,
        "bbox": [48, 72, 210, 356]
      }
    ]
  }
]
```

Images without detections are still written with `"boxes": []`.

## Validate Format and Score

After producing predictions on the validation images:

```bash
python predict.py \
  --image_dir ./public/val/images \
  --output val_predictions.json

python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions val_predictions.json \
  --output val_score.json
```
