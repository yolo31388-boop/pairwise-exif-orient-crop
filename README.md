# pairwise-exif-orient-crop

## 领域

游戏、图形与多媒体

## 主题

EXIF 方向与裁切坐标一致

## 说明

标签改写、像素空间裁切与写盘回调。

初始状态下 `imaging/exif_orient_crop.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。

## 修复实现说明

核心原则：**像素状态与 EXIF 方向标签在任何时刻必须自洽**。模块用一个 state
字典贯穿 `read → orient → crop → thumbnail → save → callback → consistency_check`，
并额外提供 `ImagePipeline` / `EditingSession` / `BatchExporter` 把各修复点串起来。

| 事故 | 修复 | 入口 |
| --- | --- | --- |
| 旋转像素后没改标签，再次保存被二次旋转 | 转置像素后显式把 Orientation(274) 改写为 `1` | `rewrite_orientation`、`orient` |
| 裁切框是显示器方向、像素仍是传感器方向，竖拍裁错区域 | 用 8 个方向的角点变换把裁切框映射到像素空间再裁（5/6/7/8 换轴） | `crop_in_pixel_space`、`crop` |
| 缩略图按原图比例，不跟随裁切 | 缩略图一律从裁切结果重生成并保持裁切框宽高比 | `thumb_after_crop` |
| 进度回调早于写盘 | 临时文件 + fsync + 原子 rename 完成后才回调；异步版 `await to_thread(写盘)` 后再回调 | `callback_after_write`、`export_batch`、`export_batch_async` |
| 多次编辑的中间副本标签/像素不同步 | 冻结一份原始像素+原始标签，所有中间副本共享；副本像素独立 | `keep_original_state`、`EditingSession` |
| 导出前缺少校验 | 校验标签—像素同步、换轴尺寸、裁切边界、缩略图来源/比例、写盘—回调顺序、原始副本哈希 | `consistency_check`、`validate_layout` |

另含 `metrics`（量化同步/裁切覆盖/比例偏差/一致性评分）、`audit`（有序审计轨迹）、
`fallback`（坏 EXIF 按方向 1 在原始像素上安全降级）、`determinism`（稳定摘要与坐标映射）。

坐标映射已用 Pillow 对全部 8 个方向 × 数千个随机裁切框验证：
“映射到传感器空间 → 裁切 → 再定向”与“先定向 → 显示空间裁切”逐像素一致。
`tests/test_exif_pipeline_fixes.py` 是对应六个修复点的回归测试。
