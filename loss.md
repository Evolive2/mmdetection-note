# 配置文件中的loss
1)model配置中的loss_cls、loss_bbox、loss_iou  
2)model配置中的train_cfg中的assigner的cls_cost、reg_cost、iou_cost  
![image](https://user-images.githubusercontent.com/104058290/220815770-e0200890-3670-499c-95da-1c1914203fa2.png)  
配置序列：  
	a) mmdetection/mmdet/core/bbox/match_costs/  
match_cost.py  [注册定义了BBoxL1Cost、FocalLossCost、ClassificationCost、IoUCost、DiceCost、CrossEntropyLossCost等MATCH_COST.register_module()]  
builder.py  [定义了build_match_cost(cfg, MATCH_COST, default_args)函数构建cost]  
b) 
