import torch


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = box_area(boxes1)[:, None] + box_area(boxes2) - inter
    return inter / union.clamp(min=1e-6)


def generalized_box_iou(boxes1, boxes2):
    iou = box_iou(boxes1, boxes2)
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return iou

    lt = torch.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    lt_inter = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb_inter = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_inter = (rb_inter - lt_inter).clamp(min=0)
    inter = wh_inter[:, :, 0] * wh_inter[:, :, 1]
    union = box_area(boxes1)[:, None] + box_area(boxes2) - inter
    return iou - (area - union) / area.clamp(min=1e-6)


def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)


def xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)


def nms(boxes, scores, iou_threshold=0.5):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]
    return torch.stack(keep)


def clip_boxes_to_image(boxes, width, height):
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
    return boxes
