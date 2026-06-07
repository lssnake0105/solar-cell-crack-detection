# Model Notes

本项目的模型权重未完整上传至 GitHub。

## Model Description

`svd_seg/` 子模块包含一个用于 SVD 预处理图像裂纹区域分割的 UNet 训练与推理流程。权重文件用于生成裂纹掩膜与叠加可视化结果。

## File Size

本地检测到的主要权重文件包括 `svd_seg/runs/unet/best.pt` 与 `svd_seg/runs/unet/last.pt`，单个文件约 118 MB。训练归档压缩包约 228 MB/个。

## Access

模型权重和训练归档默认仅本地保存。若后续需要公开，应先确认数据授权、模型权重来源、文件体积限制和 Git LFS 策略。

## Reproduction

可使用 `src/svd_seg/train_unet.py` 在重新准备的数据集上训练模型，并使用 `src/svd_seg/infer_overlay.py` 生成叠加结果。若不公开权重，GitHub 仓库仍可展示方法、代码结构、配置和代表性结果图。
