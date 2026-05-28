import torch
from torch import nn
import torch.nn.functional as F

from utils.box_ops import cxcywh_to_xyxy, generalized_box_iou, nms


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyGridDetector(nn.Module):
    def __init__(self, num_classes=5, grid_size=13, width=32):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.backbone = nn.Sequential(
            ConvBNAct(3, width, 2),
            ConvBNAct(width, width, 1),
            ConvBNAct(width, width * 2, 2),
            ConvBNAct(width * 2, width * 2, 1),
            ConvBNAct(width * 2, width * 4, 2),
            ConvBNAct(width * 4, width * 4, 1),
            ConvBNAct(width * 4, width * 8, 2),
            ConvBNAct(width * 8, width * 8, 1),
            ConvBNAct(width * 8, width * 16, 2),
            ConvBNAct(width * 16, width * 16, 1),
        )
        self.head = nn.Sequential(
            ConvBNAct(width * 16, width * 16, 1),
            nn.Conv2d(width * 16, 1 + num_classes + 4, kernel_size=1),
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = F.adaptive_avg_pool2d(feat, (self.grid_size, self.grid_size))
        pred = self.head(feat)
        return pred.permute(0, 2, 3, 1).contiguous()


def detection_loss(pred, targets, num_classes, lambda_box=5.0, lambda_obj=1.0, lambda_noobj=0.35, lambda_cls=1.0):
    obj_target = targets["objectness"].to(pred.device)
    cls_target = targets["classes"].to(pred.device)
    box_target = targets["boxes"].to(pred.device)
    pos = obj_target > 0.5

    obj_logits = pred[..., 0]
    cls_logits = pred[..., 1 : 1 + num_classes]
    box_raw = pred[..., 1 + num_classes :]

    obj_weight = torch.where(pos, torch.full_like(obj_target, lambda_obj), torch.full_like(obj_target, lambda_noobj))
    obj_loss = F.binary_cross_entropy_with_logits(obj_logits, obj_target, weight=obj_weight, reduction="mean")

    if pos.any():
        cls_loss = F.cross_entropy(cls_logits[pos], cls_target[pos])
        box_pred = torch.sigmoid(box_raw[pos])
        reg_loss = F.smooth_l1_loss(box_pred, box_target[pos], reduction="mean")

        pred_xyxy = cxcywh_to_xyxy(_cell_box_to_global(box_pred, pos, pred.shape[1]))
        target_xyxy = cxcywh_to_xyxy(_cell_box_to_global(box_target[pos], pos, pred.shape[1]))
        giou_loss = (1.0 - generalized_box_iou(pred_xyxy, target_xyxy).diag()).mean()
        box_loss = reg_loss + giou_loss
    else:
        cls_loss = pred.sum() * 0.0
        box_loss = pred.sum() * 0.0

    total = lambda_obj * obj_loss + lambda_cls * cls_loss + lambda_box * box_loss
    return total, {"loss": total.item(), "obj": obj_loss.item(), "cls": cls_loss.item(), "box": box_loss.item()}


def _cell_box_to_global(boxes, pos_mask, grid_size):
    ys, xs = pos_mask.nonzero(as_tuple=True)[1:3]
    gx = xs.to(boxes.device).float()
    gy = ys.to(boxes.device).float()
    cx = (boxes[:, 0] + gx) / grid_size
    cy = (boxes[:, 1] + gy) / grid_size
    return torch.stack((cx, cy, boxes[:, 2], boxes[:, 3]), dim=-1).clamp(0, 1)


@torch.no_grad()
def decode_predictions(pred, image_width, image_height, classes, conf_threshold=0.25, nms_threshold=0.5, max_detections=100):
    grid_size = pred.shape[0]
    num_classes = len(classes)
    pred = pred.detach()
    obj = torch.sigmoid(pred[..., 0])
    cls_prob = torch.softmax(pred[..., 1 : 1 + num_classes], dim=-1)
    box = torch.sigmoid(pred[..., 1 + num_classes :])

    ys, xs = torch.meshgrid(
        torch.arange(grid_size, device=pred.device),
        torch.arange(grid_size, device=pred.device),
        indexing="ij",
    )
    cx = (box[..., 0] + xs) / grid_size
    cy = (box[..., 1] + ys) / grid_size
    wh = box[..., 2:4]
    xyxy = cxcywh_to_xyxy(torch.stack((cx, cy, wh[..., 0], wh[..., 1]), dim=-1))
    xyxy[..., 0::2] *= float(image_width)
    xyxy[..., 1::2] *= float(image_height)

    scores, labels = (obj[..., None] * cls_prob).max(dim=-1)
    keep = scores >= conf_threshold
    if keep.sum() == 0:
        return []

    boxes = xyxy[keep].reshape(-1, 4)
    scores = scores[keep].reshape(-1)
    labels = labels[keep].reshape(-1)
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, image_width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, image_height)

    valid = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
    boxes, scores, labels = boxes[valid], scores[valid], labels[valid]

    detections = []
    for cls_idx, cls_name in enumerate(classes):
        cls_keep = labels == cls_idx
        if cls_keep.sum() == 0:
            continue
        kept = nms(boxes[cls_keep], scores[cls_keep], nms_threshold)
        cls_boxes = boxes[cls_keep][kept]
        cls_scores = scores[cls_keep][kept]
        for b, s in zip(cls_boxes, cls_scores):
            detections.append(
                {
                    "class": cls_name,
                    "confidence": float(s.clamp(0, 1).cpu()),
                    "bbox": [float(v) for v in b.cpu().tolist()],
                }
            )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections[:max_detections]
