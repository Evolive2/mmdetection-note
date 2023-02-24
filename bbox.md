# 模型预测bbox相关
## 0.bbox格式互转
mmdetection/mmdet/core/bbox/transforms.py/  

通常有三种格式来表示bounding box的位置：  
xyxy，即(x_1, y_1, x_2, y_2)，其中(x_1, y_1)是bounding box左上角的坐标，(x_2,y_2)是bounding box右下角的坐标；  
xywh，即(x, y, w, h)，其中(x, y)是bounding box左上角的坐标，w是矩形框的宽度，h是矩形框的高度；  
cxcywh，即(c_x_, c_y_, w, h)，其中(x, y)是bounding box中心点的坐标，w是矩形框的宽度，h是矩形框的高度。  

```
def bbox_cxcywh_to_xyxy(bbox):  
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.
        
    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)


def bbox_xyxy_to_cxcywh(bbox):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.
        
    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)
```
