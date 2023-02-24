# 配置文件中的loss
## 0. model配置中的bbox_head中的loss_cls、loss_bbox、loss_iou  
    bbox_head=dict(
        type='DeformableDETRHead',
        ...
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0))


配置序列：  
### 0.0 mmdetection/mmdet/models/losses/  训练损失(loss)  

A) focal_loss.py---[注册定义了FocalLoss的@LOSSES.register_module()]  
B) smooth_l1_loss.py---[注册定义了L1 loss、SmoothL1Loss等@LOSSES.register_module()]  
C) iou_loss.py---[注册定义了GIoULoss、IoULoss、BoundedIoULoss等@LOSSES.register_module()]  
IOU计算函数位于mmdetection/mmdet/core/bbox/iou_calculators/iou2d_calculator.py/中的def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6)
...  

## 1. model配置中的train_cfg中的assigner的cls_cost、reg_cost、iou_cost  
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),

配置序列：  
### 1.0 mmdetection/mmdet/core/bbox/match_costs/  匹配损失(match cost)  
A) match_cost.py---[注册定义了BBoxL1Cost、FocalLossCost、ClassificationCost、IoUCost等MATCH_COST.register_module()]  
B) builder.py---[定义了build_match_cost(cfg, MATCH_COST, default_args)函数构建cost]  

### 1.1 mmdetection/mmdet/core/bbox/assigners/  分配器(assigner)  
A) hungarian_assigner.py---[注册定义了HungarianAssigner的BBOX_ASSIGNERS.register_module()]  
B) approx_max_iou_assigner.py  
C) mask_hungarian_assigner.py  
...  
