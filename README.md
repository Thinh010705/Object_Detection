# Object Detection Submission

This submission implements a custom object detector in PyTorch. It does not use YOLO, Detectron2, MMDetection, Faster R-CNN, SSD, or torchvision detection models.

## Method

- Data reader for the provided `train.json` / `val.json` format.
- Resize to a square input size and ImageNet-style pixel normalization.
- Multi-object handling by assigning each object to the grid cell containing its center.
- Augmentations: horizontal flip, random crop, and color jitter.
- CNN backbone implemented with basic convolution layers, or an optional ImageNet-pretrained ResNet50 feature extractor.
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

## Stronger Allowed Backbone

If ImageNet-pretrained feature extractors are allowed, train the same custom
grid detector with a ResNet50 backbone:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/ \
  --backbone resnet50 \
  --image_size 512 \
  --grid_size 16 \
  --model_width 64 \
  --epochs 80 \
  --batch_size 4 \
  --lr 1e-4 \
  --score_every 1
```

This still uses the project detection head, loss, decoder, and NMS. The best
checkpoint is selected by validation mAP@0.5 and saved with its score:

```text
./models/best.pth
./models/best_val_score.json
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
  --output predictions.json \
  --conf_threshold 0.05
```

By default, `predict.py` loads `./models/best.pth`. To use another checkpoint:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json \
  --checkpoint ./models/best.pth
```

The default confidence threshold is `0.05`, which improved validation mAP@0.5
for the current checkpoint compared with the original `0.25` threshold.

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
  --output val_predictions.json \
  --conf_threshold 0.05

python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions val_predictions.json \
  --output val_score.json
```
