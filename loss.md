# 配置文件中的loss
# 1)model配置中的loss_cls、loss_bbox、loss_iou  
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),

配置序列：  
a) mmdetection/mmdet/models/losses/  训练损失(loss)  

focal_loss.py---[注册定义了FocalLoss的@LOSSES.register_module()]  
smooth_l1_loss.py---[注册定义了L1 loss、SmoothL1Loss等@LOSSES.register_module()]  
iou_loss.py---[注册定义了GIoULoss、IoULoss、BoundedIoULoss等@LOSSES.register_module()]  
...  

# 2)model配置中的train_cfg中的assigner的cls_cost、reg_cost、iou_cost  
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),

配置序列：  
a) mmdetection/mmdet/core/bbox/match_costs/  匹配损失(match cost)  
match_cost.py---[注册定义了BBoxL1Cost、FocalLossCost、ClassificationCost、IoUCost等MATCH_COST.register_module()]  
builder.py---[定义了build_match_cost(cfg, MATCH_COST, default_args)函数构建cost]  

b) mmdetection/mmdet/core/bbox/assigners/  分配器(assigner)  
hungarian_assigner.py---[注册定义了HungarianAssigner的BBOX_ASSIGNERS.register_module()]  
approx_max_iou_assigner.py  
mask_hungarian_assigner.py  
...  
