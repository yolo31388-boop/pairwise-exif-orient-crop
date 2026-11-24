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
