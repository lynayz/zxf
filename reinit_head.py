import torch
import torch.nn as nn
from models.yolo import Model  # 你的flexible-yolov5模型导入

# ===== 配置 =====
backbone_neck_path = "/home/guoqingbei/data/exports/zengxiangfei/flexible-yolov5-2/runs/train/lrnet/weights/hrnet_best_backbone_neck.pt"
# 正确的配置文件（nc=1）
cfg_path = "/home/guoqingbei/data/exports/zengxiangfei/flexible-yolov5-2/configs/model_hrnet.yaml"  # 包含nc=1的配置
# 保存最终修正后的权重
final_weights_path = "/home/guoqingbei/data/exports/zengxiangfei/flexible-yolov5-2/runs/train/lrnet/weights/hrnet_best_nc1.pt"

# ===== 1. 加载正确配置的模型（nc=1）=====
# 初始化一个全新的模型（head层为nc=1）
model = Model(cfg_path).float().eval()

# ===== 2. 加载最优的backbone+neck权重 =====
ckpt = torch.load(backbone_neck_path, map_location="cpu")
backbone_neck_state_dict = ckpt["backbone_neck_state_dict"]

# ===== 3. 加载权重到新模型 =====
model_state_dict = model.state_dict()
# 匹配并加载backbone+neck权重
for k in backbone_neck_state_dict.keys():
    if k in model_state_dict:
        model_state_dict[k] = backbone_neck_state_dict[k]

# 重新初始化head的最后一层（nc=1）
for k, v in model_state_dict.items():
    if "head" in k and ("conv.weight" in k or "conv.bias" in k):
        # 初始化新的卷积层（nc=1）
        if "weight" in k:
            # 随机初始化（均值0，方差0.01）
            nn.init.normal_(v, mean=0.0, std=0.01)
        else:
            # 偏置初始化为0
            nn.init.constant_(v, 0.0)

# 加载更新后的权重
model.load_state_dict(model_state_dict)

# ===== 4. 保存最终修正后的权重 =====
torch.save({
    "model": model,
    "epoch": -1,  # 标记为修正后的权重
    "nc": 1,
    "config": cfg_path
}, final_weights_path)

print(f"✅ 已生成修正后的权重：{final_weights_path}")
print(f"   - 保留HRNet18最优backbone+neck")
print(f"   - head层已适配nc=1（正确类别数）")