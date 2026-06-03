import torch
import numpy as np
from od.models.backbone.hrnet.model import LightHourglassBlock  # 导入缺失的类
from od.models.backbone.hrnet import hrnet  # 导入HRNet整体模块
# ===== 配置 =====
# 你的最优权重（nc=80时训练的）
best_weights_path = "/home/guoqingbei/data/exports/zengxiangfei/flexible-yolov5-2/runs/train/lrnet/weights/best.pt"
# 保存剥离后的权重（仅backbone+neck）
save_backbone_neck_path = "/home/guoqingbei/data/exports/zengxiangfei/flexible-yolov5-2/runs/train/lrnet/weights/hrnet_best_backbone_neck.pt"

# ===== 加载权重 =====
ckpt = torch.load(best_weights_path, map_location="cpu")
model = ckpt["model"].float().eval()

# ===== 剥离head的最后一层（关键）=====
# 遍历模型，仅保留backbone和neck的权重
new_state_dict = {}
for k, v in model.state_dict().items():
    # 过滤掉head最后一层的权重（包含"head"且包含"conv"的层）
    if "head" in k and ("conv.weight" in k or "conv.bias" in k):
        continue
    new_state_dict[k] = v

# ===== 保存backbone+neck权重 =====
torch.save({
    "backbone_neck_state_dict": new_state_dict,
    "config": {
        "backbone": "hrnet18",
        "neck": "FPN-s+PAN-s",
        "nc_old": 80,
        "nc_new": 1
    }
}, save_backbone_neck_path)

print(f"✅ 已提取最优backbone+neck权重：{save_backbone_neck_path}")
print(f"   保留了HRNet18 + FPN(PAN)-s的所有权重，仅剔除了head的错误层")