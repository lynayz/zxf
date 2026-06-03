import torch
import torch.nn as nn
import os
import numpy as np
import random
from copy import deepcopy
# 适配YOLOv5模型导入（根据你的YOLOv5路径调整）
try:
    from models.yolo import Model  # YOLOv5源码目录下
except:
    # 若直接加载ckpt，无需导入Model，注释上面一行即可
    pass

# ===== 固定随机种子（保证量化结果可复现）=====
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# ===== 配置 =====
mvq_configs = {"MVQ-16": 16}  # 4bit=16个离散值
weights_path = "runs/train/yolo/weights/best.pt"  # 你的基准模型
save_root = "mvq_quant_models"
os.makedirs(save_root, exist_ok=True)
input_shape = (1, 3, 640, 640)

# ===== 加载基准模型 =====
ckpt = torch.load(weights_path, map_location="cpu")
model_base = ckpt["model"].float().eval()

# ===== MVQ-16量化函数（KMeans后量化）=====
def kmeans_torch(weight, k, iters=20, max_samples=500000):
    """适配k=16，迭代次数设为20保证收敛"""
    w = weight.view(-1).clone()
    # 采样避免内存溢出
    if w.numel() > max_samples:
        idx = torch.randperm(w.numel())[:max_samples]
        w_sample = w[idx]
    else:
        w_sample = w
    # 初始化聚类中心
    indices = torch.randperm(w_sample.numel())[:k]
    centers = w_sample[indices].clone()
    # KMeans迭代
    for _ in range(iters):
        dist = torch.abs(w_sample.unsqueeze(1) - centers.unsqueeze(0))
        labels = torch.argmin(dist, dim=1)
        # 更新聚类中心
        for i in range(k):
            mask = labels == i
            if mask.sum() > 0:
                centers[i] = w_sample[mask].mean()
    # 对全量权重赋值
    dist_full = torch.abs(w.unsqueeze(1) - centers.unsqueeze(0))
    labels_full = torch.argmin(dist_full, dim=1)
    return centers, labels_full

def quantize_model(model, k):
    """量化backbone的3x3 Conv层（和你的原逻辑一致）"""
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
            weight = module.weight.data
            shape = weight.shape
            centers, labels = kmeans_torch(weight, k=k)
            w_quant = centers[labels].view(shape)
            module.weight.data = w_quant.type_as(weight)
    return model

# ===== 生成MVQ-16模型+导出ONNX =====
for mvq_name, k in mvq_configs.items():
    # 复制模型避免覆盖
    model = deepcopy(model_base)
    # 执行MVQ-16量化
    model = quantize_model(model, k=k)
    # 保存量化模型
    save_pt = os.path.join(save_root, f"{mvq_name}.pt")
    torch.save({"model": model}, save_pt)
    # 导出ONNX（Jetson端TensorRT用）
    save_onnx = os.path.join(save_root, f"{mvq_name}.onnx")
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(
        model, dummy_input, save_onnx,
        opset_version=12,
        do_constant_folding=False,
        input_names=["images"],
        output_names=["output0", "output1", "output2"],
        verbose=False
    )
    print(f"✅ MVQ-16模型生成完成：")
    print(f"  - PT文件：{save_pt}")
    print(f"  - ONNX文件：{save_onnx}")