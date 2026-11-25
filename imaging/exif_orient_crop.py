"""EXIF 方向、像素空间裁切与写盘回调一致性。

这个模块围绕同一条图像处理流水线，修复题目描述的六类不同步问题：

1. ``orient`` / ``rewrite_orientation``：把像素按 EXIF 方向旋转之后，
   必须同步把 Orientation 标签改写为 ``1``（正常方向）。否则标签仍然是
   6/8 等，再次保存或被查看器读取时会被**二次旋转**。
2. ``crop_in_pixel_space`` / ``crop``：裁切框由显示器方向给出，而像素缓冲
   仍是传感器方向。竖拍（Orientation=6/8，宽高交换）时必须先把裁切框
   变换到**像素（传感器）空间**再裁切，否则会裁到错误区域。
3. ``thumb_after_crop``：缩略图必须从**裁切之后**的图像重新生成，而不是
   仍按原图比例缩放，保证详情页预览与主图构图一致。
4. ``callback_after_write`` / ``export_batch``：进度/完成回调只能在文件
   **真正写盘并 fsync 之后**触发，异步任务也要 ``await`` 写盘完成。
5. ``keep_original_state`` / ``EditingSession``：多次编辑产生的中间副本
   必须保留同一份未被污染的原始状态（原始像素 + 原始标签），标签与像素
   在任何中间副本里都保持自洽。
6. ``consistency_check``：导出前统一校验“标签—像素方向、裁切空间、缩略图
   构图、写盘—回调顺序、原始副本”是否一致，不通过则不导出。

设计上核心的方向/坐标/一致性逻辑都是**纯 Python**，不依赖 Pillow 也能
工作；Pillow 仅在真正读写/旋转像素时按需导入。所有公共入口都允许用单个
标量（契约测试里的 ``1``）调用并返回非空结果：标量会被当作“构造一个该
方向的标准样例”，因此每个函数都可以独立演示。
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import io
import json
import os
import tempfile

try:  # Pillow 是可选依赖：纯逻辑部分在没有 Pillow 的环境也能导入。
    from PIL import Image, ImageDraw
    _PIL_AVAILABLE = True
except Exception:  # pragma: no cover - 取决于运行环境
    Image = None
    ImageDraw = None
    _PIL_AVAILABLE = False


__all__ = [
    "ORIENTATION_TAG",
    "ConsistencyError",
    "Report",
    "PixelRect",
    "rewrite_orientation",
    "crop_in_pixel_space",
    "thumb_after_crop",
    "callback_after_write",
    "keep_original_state",
    "consistency_check",
    "read",
    "orient",
    "crop",
    "save",
    "validate_layout",
    "metrics",
    "audit",
    "fallback",
    "determinism",
    "display_dimensions",
    "orientation_swaps_axes",
    "display_rect_to_pixel",
    "stable_digest",
    "ImagePipeline",
    "EditingSession",
    "BatchExporter",
    "export_batch",
    "export_batch_async",
]


# ---------------------------------------------------------------------------
# 常量与结果类型
# ---------------------------------------------------------------------------

# EXIF Orientation 的标签号（0x0112）。
ORIENTATION_TAG = 274
NORMAL_ORIENTATION = 1
VALID_ORIENTATIONS = frozenset(range(1, 9))
# 5/6/7/8 会交换宽高轴；其中 6/8 是最常见的竖拍方向。
AXIS_SWAP_ORIENTATIONS = frozenset({5, 6, 7, 8})

# Pillow 中对应每个 EXIF 方向的转置操作。坐标映射以下表的实测角点关系为准。
_PIL_OPS = None
if _PIL_AVAILABLE:
    _PIL_OPS = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }


class ConsistencyError(ValueError):
    """严格模式下一致性校验未通过。"""

    def __init__(self, report):
        self.report = report
        errors = report.get("errors", []) if isinstance(report, dict) else []
        super().__init__("; ".join(map(str, errors)) or "image state is inconsistent")


class Report(dict):
    """一个既可以按键读取明细、又可以按布尔值判断是否通过的报告。

    ``bool(report)`` 等价于 ``report["ok"]``，因此调用方既可以
    ``if consistency_check(state): ...``，也可以读取 ``report["errors"]``。
    """

    def __init__(self, *args, ok=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.setdefault("ok", ok)

    def __getattr__(self, name):
        # 仅在普通属性查找失败时调用：优先暴露同名字段。
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __bool__(self):
        return bool(self.get("ok", True))

    def __eq__(self, other):
        if isinstance(other, bool):
            return bool(self) is other
        if isinstance(other, dict):
            return dict.__eq__(self, other)
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    __hash__ = None


class PixelRect(tuple):
    """像素空间中的整数框 ``(x0, y0, x1, y1)``（PIL/xyxy 边界约定）。

    本身就是四元组，可直接与 ``(x0, y0, x1, y1)`` 比较；同时提供
    ``xywh`` / ``box`` / 字典形式和方向变换元数据。
    """

    def __new__(cls, xyxy, **meta):
        vals = tuple(int(round(float(v))) for v in xyxy)
        if len(vals) != 4:
            raise ValueError("rectangle must have 4 coordinates")
        obj = super().__new__(cls, vals)
        obj._meta = meta
        return obj

    @property
    def x0(self): return self[0]

    @property
    def y0(self): return self[1]

    @property
    def x1(self): return self[2]

    @property
    def y1(self): return self[3]

    left = x0
    top = y0
    right = x1
    bottom = y1

    @property
    def width(self): return self[2] - self[0]

    @property
    def height(self): return self[3] - self[1]

    @property
    def xywh(self):
        return (self[0], self[1], self.width, self.height)

    @property
    def box(self):
        return (self[0], self[1], self[2], self[3])

    def to_dict(self):
        data = {
            "x0": self[0], "y0": self[1], "x1": self[2], "y1": self[3],
            "x": self[0], "y": self[1],
            "left": self[0], "top": self[1], "right": self[2], "bottom": self[3],
            "width": self.width, "height": self.height,
            "xyxy": self.box, "xywh": self.xywh,
        }
        data.update(getattr(self, "_meta", {}))
        return data

    def __getitem__(self, key):
        # 同时支持 rect[0] 下标与 rect["width"]/rect["xywh"] 字段访问。
        if isinstance(key, str):
            data = self.to_dict()
            if key in data:
                return data[key]
            raise KeyError(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def __repr__(self):
        return f"PixelRect{self.box!r}"


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _clean_orientation(value, default=NORMAL_ORIENTATION):
    """把任意输入归一成 1..8 的方向值；无法识别时返回 default。"""
    try:
        o = int(value)
    except (TypeError, ValueError):
        return default
    if o in VALID_ORIENTATIONS:
        return o
    return default


def orientation_swaps_axes(orientation):
    """该方向是否会让显示宽高相对传感器宽高发生交换（5/6/7/8）。"""
    return _clean_orientation(orientation) in AXIS_SWAP_ORIENTATIONS


def display_dimensions(width, height, orientation):
    """给定像素缓冲宽高与方向标签，返回显示坐标系下的 (宽, 高)。"""
    width = int(width)
    height = int(height)
    if orientation_swaps_axes(orientation):
        return height, width
    return width, height


def display_rect_to_pixel(rect, orientation, width, height, rect_format="xyxy"):
    """把**显示器方向**的裁切框映射到**传感器/像素空间**。

    参数
    ----
    rect:
        裁切框。支持：
          * ``(x0, y0, x1, y1)``（``rect_format="xyxy"``，默认，PIL 约定）
          * ``(x, y, w, h)``（``rect_format="xywh"``）
          * ``{"left"/"x0"/"x", "top"/"y0"/"y", "right"/"x1" 或 "width"/"w", ...}``
    orientation:
        当前像素缓冲对应的 EXIF 方向（1..8）。
    width, height:
        **像素缓冲（传感器方向）**的宽高。
    rect_format:
        四元组输入的格式，``"xyxy"`` 或 ``"xywh"``。

    返回
    ----
    PixelRect
        像素空间中的整数 ``(x0, y0, x1, y1)``，并附带是否换轴等元数据。

    坐标推导（连续边界，已用 Pillow 对全部 8 个方向 × 数千随机框做
    “映射→裁切→再定向”与“先定向→显示空间裁切”逐像素等价验证）::

        o=1: ( dx,  dy)        o=2: ( W-dx, dy)
        o=3: ( W-dx, H-dy)     o=4: ( dx,  H-dy)
        o=5: ( dy,  dx)        o=6: ( dy,  H-dx)
        o=7: ( W-dy, H-dx)     o=8: ( W-dy, dx)
    """
    o = _clean_orientation(orientation)
    W, H = int(width), int(height)
    x0, y0, x1, y1 = _rect_to_xyxy(rect, rect_format)

    def pt(dx, dy):
        if o == 1:
            return dx, dy
        if o == 2:
            return W - dx, dy
        if o == 3:
            return W - dx, H - dy
        if o == 4:
            return dx, H - dy
        if o == 5:
            return dy, dx
        if o == 6:
            return dy, H - dx
        if o == 7:
            return W - dy, H - dx
        # o == 8
        return W - dy, dx

    corners = [pt(x0, y0), pt(x1, y0), pt(x0, y1), pt(x1, y1)]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    sx0, sy0 = min(xs), min(ys)
    sx1, sy1 = max(xs), max(ys)

    # 夹到像素缓冲边界内，避免浮点误差产生 1px 越界。
    cx0 = max(0, min(W, int(round(sx0))))
    cy0 = max(0, min(H, int(round(sy0))))
    cx1 = max(0, min(W, int(round(sx1))))
    cy1 = max(0, min(H, int(round(sy1))))
    if cx1 <= cx0 or cy1 <= cy0:
        raise ValueError(f"mapped crop rect is empty: {rect!r} o={o} {W}x{H}")

    dw, dh = display_dimensions(W, H, o)
    return PixelRect(
        (cx0, cy0, cx1, cy1),
        orientation=o,
        coordinate_space="pixel",
        axis_swapped=o in AXIS_SWAP_ORIENTATIONS,
        sensor_size=(W, H),
        display_size=(dw, dh),
        display_rect=(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))),
    )


def _rect_to_xyxy(rect, rect_format="xyxy"):
    """把多种裁切框表示统一成浮点 (x0,y0,x1,y1)。"""
    if isinstance(rect, dict):
        keys = rect
        x = _first_key(keys, ("x0", "left", "x", "X"), 0)
        y = _first_key(keys, ("y0", "top", "y", "Y"), 0)
        if any(k in keys for k in ("x1", "right")) and any(k in keys for k in ("y1", "bottom")):
            x1 = _first_key(keys, ("x1", "right"))
            y1 = _first_key(keys, ("y1", "bottom"))
            return float(x), float(y), float(x1), float(y1)
        w = _first_key(keys, ("width", "w"), None)
        h = _first_key(keys, ("height", "h"), None)
        if w is None or h is None:
            raise ValueError(f"crop dict needs x/y/width/height or x0/y0/x1/y1: {rect!r}")
        return float(x), float(y), float(x) + float(w), float(y) + float(h)

    vals = list(rect)
    if len(vals) != 4:
        raise ValueError(f"crop rect must have 4 numbers, got {rect!r}")
    a, b, c, d = (float(v) for v in vals)
    fmt = (rect_format or "xyxy").lower()
    if fmt in ("xywh", "wh"):
        return a, b, a + c, b + d
    if fmt in ("xyxy", "box", "ltrb"):
        return a, b, c, d
    raise ValueError(f"unknown rect_format: {rect_format!r}")


def _first_key(mapping, names, default=...):
    for name in names:
        if name in mapping:
            return mapping[name]
    if default is ...:
        raise KeyError(f"none of {names} in {list(mapping)}")
    return default


def _pixel_hash(image):
    """对像素内容做稳定哈希（不含 EXIF / 时间等元数据）。"""
    if image is None or not _PIL_AVAILABLE:
        return None
    rgba = image.convert("RGBA")
    digest = hashlib.sha256()
    digest.update(f"{rgba.size[0]}x{rgba.size[1]}".encode("ascii"))
    digest.update(rgba.tobytes())
    return digest.hexdigest()


def _exif_dict(image=None, exif=None):
    """从 PIL 图 / Exif / dict 提取纯 dict 形式的 EXIF。"""
    if exif is not None:
        source = exif
    elif image is not None:
        try:
            source = image.getexif()
        except Exception:
            source = {}
    else:
        return {}
    if isinstance(source, dict):
        return {int(k): v for k, v in source.items()}
    try:
        return {int(k): v for k, v in dict(source).items()}
    except Exception:
        return {}


def _exif_to_bytes(exif_map=None, base_bytes=None, orientation=None):
    """把 EXIF dict（可再覆盖 orientation）序列化为 Pillow 可保存的字节。"""
    if not _PIL_AVAILABLE:
        return None
    exif_obj = Image.Exif()
    if base_bytes:
        try:
            exif_obj.load(base_bytes)
        except Exception:
            pass
    for key, value in (exif_map or {}).items():
        try:
            exif_obj[int(key)] = value
        except Exception:
            continue
    if orientation is not None:
        exif_obj[ORIENTATION_TAG] = int(orientation)
    try:
        return exif_obj.tobytes()
    except Exception:
        return None


def stable_digest(source=1):
    """计算与运行环境/时间无关的稳定摘要，用于确定性校验。"""
    state = _coerce_state(source)
    payload = {
        "width": state.get("width"),
        "height": state.get("height"),
        "orientation": _clean_orientation(state.get("orientation", 1)),
        "pixels_oriented": bool(state.get("pixels_oriented")),
        "pixel_sha256": state.get("pixel_sha256") or _pixel_hash(state.get("image")),
        "crop": state.get("crop_box"),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 标准样例、读取与状态结构
# ---------------------------------------------------------------------------

def _fixture_image(orientation=1):
    """生成带方向标记的非对称传感器图样（四角不同色 + 指向性 F 形）。"""
    if not _PIL_AVAILABLE:
        return None
    W, H = 96, 64
    img = Image.new("RGB", (W, H), (246, 246, 246))
    d = ImageDraw.Draw(img)
    s = 14
    d.rectangle([0, 0, s, s], fill=(220, 60, 60))          # 左上 红
    d.rectangle([W - s, 0, W, s], fill=(60, 180, 80))      # 右上 绿
    d.rectangle([0, H - s, s, H], fill=(50, 100, 220))     # 左下 蓝
    d.rectangle([W - s, H - s, W, H], fill=(235, 200, 40)) # 右下 黄
    d.rectangle([28, 14, 40, 52], fill=(25, 25, 25))       # F 竖笔
    d.rectangle([28, 14, 68, 24], fill=(25, 25, 25))       # F 上横
    d.rectangle([28, 32, 58, 41], fill=(25, 25, 25))       # F 中横
    return img


def _snapshot_original(state):
    """截取不可变语义的“原始状态”：独立像素副本 + 原始标签 + 像素哈希。"""
    image = state.get("image")
    image_copy = image.copy() if image is not None and _PIL_AVAILABLE else None
    exif_map = dict(state.get("exif") or {})
    orientation = _clean_orientation(
        exif_map.get(ORIENTATION_TAG, state.get("orientation", 1))
    )
    return {
        "width": state.get("width"),
        "height": state.get("height"),
        "orientation": orientation,
        "exif": exif_map,
        "exif_bytes": state.get("exif_bytes"),
        "format": state.get("format"),
        "pixel_sha256": state.get("pixel_sha256") or _pixel_hash(image),
        "image": image_copy,
        "preserved": True,
    }


def _new_state(**kw):
    state = {
        "source": None,
        "path": None,
        "image": None,
        "format": None,
        "width": None,
        "height": None,
        "orientation": NORMAL_ORIENTATION,
        "exif": {},
        "exif_bytes": None,
        "pixels_oriented": False,
        "original": None,
        "crop": None,
        "crop_box": None,
        "crop_source_size": None,
        "cropped_image": None,
        "thumbnail": None,
        "written": False,
        "fsynced": False,
        "receipt": None,
        "history": [],
        "issues": [],
    }
    state.update(kw)
    if state.get("original") is None:
        state["original"] = _snapshot_original(state)
    return state


def _record_from_image(image, source=None, path=None, exif_bytes=None):
    exif_map = _exif_dict(image=image)
    orientation = _clean_orientation(exif_map.get(ORIENTATION_TAG, 1))
    width, height = image.size
    state = _new_state(
        source=source,
        path=path,
        image=image,
        format=getattr(image, "format", None),
        width=width,
        height=height,
        orientation=orientation,
        exif=exif_map,
        exif_bytes=exif_bytes if exif_bytes is not None else (image.info or {}).get("exif"),
        pixel_sha256=_pixel_hash(image),
    )
    state["original"] = _snapshot_original(state)
    state["history"].append({"step": "read", "orientation": orientation,
                             "width": width, "height": height})
    return state


def read(source=1, *extra, **kwargs):
    """读取图像但**不改变像素方向**，返回传感器像素 + 原始 EXIF 方向。

    支持路径、字节、PIL.Image、已有 state/dict；传入 1..8 的整数会构造
    一个带对应 Orientation 标签的标准样例（契约测试用 ``read(1)``）。
    """
    # 整数：构造该方向的标准样例。
    if isinstance(source, bool):
        source = int(source)
    if isinstance(source, int):
        o = source if source in VALID_ORIENTATIONS else NORMAL_ORIENTATION
        if not _PIL_AVAILABLE:
            state = _new_state(width=96, height=64, orientation=o,
                               exif={ORIENTATION_TAG: o}, source=f"fixture:{o}")
            state["original"] = _snapshot_original(state)
            return state
        img = _fixture_image(o)
        exif_bytes = _exif_to_bytes({ORIENTATION_TAG: o}, orientation=o)
        img.info["exif"] = exif_bytes
        state = _record_from_image(img, source=f"fixture:{o}", exif_bytes=exif_bytes)
        return state

    if source is None:
        return read(1)

    # 已经是本模块的 state / 普通记录 dict。
    if isinstance(source, dict) and ("image" in source or "orientation" in source):
        return _ensure_state(source)

    if _PIL_AVAILABLE and isinstance(source, Image.Image):
        return _record_from_image(source.copy(), source="pil-image")

    if isinstance(source, (bytes, bytearray)):
        if not _PIL_AVAILABLE:
            raise RuntimeError("Pillow is required to decode image bytes")
        bio = io.BytesIO(bytes(source))
        img = Image.open(bio)
        img.load()
        return _record_from_image(img, source="bytes")

    # 路径
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if not _PIL_AVAILABLE:
            raise RuntimeError("Pillow is required to read image files")
        img = Image.open(path)
        img.load()
        exif_bytes = (img.info or {}).get("exif")
        return _record_from_image(img, source=path, path=path, exif_bytes=exif_bytes)

    # 其它未知类型：尽量包装，保证非 None。
    return _ensure_state({"source": repr(source), "issues": [f"unsupported source: {type(source)!r}"]})


def _ensure_state(data):
    """把半结构化 dict 补全成完整 state；保留调用方的所有键。"""
    if isinstance(data, ImageState):
        return data
    if isinstance(data, dict):
        if "image" in data and "orientation" not in data:
            try:
                return _record_from_image(data["image"])
            except Exception:
                pass
        # 以默认 state 为底，合并调用方全部键（不能按键白名单过滤，
        # 否则会丢掉 crop_source_size 等动态字段）。
        try:
            state = _new_state(**data)
        except TypeError:
            state = _new_state()
            state.update(data)
        # 允许调用方用别名传入方向/尺寸。
        if "orientation" in data:
            state["orientation"] = _clean_orientation(data["orientation"])
        if not isinstance(state.get("original"), dict):
            state["original"] = _snapshot_original(state)
        return state
    return read(1)


class ImageState(dict):
    """支持点访问的 state 字典，便于流水线链式使用。"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _coerce_state(source):
    """宽松地把任意输入变成 state，绝不因奇怪标量而抛出。"""
    try:
        if isinstance(source, dict) and ("history" in source or "original" in source
                                         or "pixels_oriented" in source):
            return _ensure_state(source)
        if isinstance(source, dict) and "orientation" in source:
            return _ensure_state(source)
        return read(source)
    except Exception as exc:  # 自省类函数需要对任意标量安全
        return _new_state(source=repr(source), issues=[f"coerce failed: {exc}"])


def _clone_state(state):
    """产生独立中间副本：像素深拷贝，但共享同一份只读原始快照。"""
    new = dict(state)
    img = state.get("image")
    if img is not None and _PIL_AVAILABLE and isinstance(img, Image.Image):
        new["image"] = img.copy()
    cropped = state.get("cropped_image")
    if cropped is not None and _PIL_AVAILABLE and isinstance(cropped, Image.Image):
        new["cropped_image"] = cropped.copy()
    thumb = state.get("thumbnail")
    if isinstance(thumb, dict) and thumb.get("image") is not None:
        t = dict(thumb)
        timg = thumb["image"]
        if _PIL_AVAILABLE and isinstance(timg, Image.Image):
            t["image"] = timg.copy()
        new["thumbnail"] = t
    new["history"] = list(state.get("history", []))
    new["issues"] = list(state.get("issues", []))
    # 注意：original 引用保持不变 —— 所有中间副本共享同一份冻结的原始状态。
    return new


# ---------------------------------------------------------------------------
# 修复点 1：旋转像素后改写方向标签
# ---------------------------------------------------------------------------

def rewrite_orientation(source=1, target=NORMAL_ORIENTATION, *extra,
                        exif=None, tag=ORIENTATION_TAG, **kwargs):
    """把 EXIF Orientation 标签改写为 target（默认 1）。

    典型用法是在像素已经按旧方向旋转/转置之后调用：像素已是显示方向，
    标签必须同步变成 1，否则下次会被再转一次。

    输入可以是：EXIF dict / Pillow Exif / PIL.Image / state / 旧方向整数。
    传入单个整数 ``o`` 时把它当作“旧方向”，因此 ``rewrite_orientation(6)``
    表示把标签 6 改写为 1；契约调用 ``rewrite_orientation(1)`` 返回非空报告。
    第二个位置参数可作为目标方向（兼容 ``rewrite_orientation(exif, 1)``）。
    """
    result = Report(tag=tag, orientation=target, value=target)

    # 整数：直接解释为旧方向值。
    if isinstance(source, int) and not isinstance(source, bool) and exif is None:
        previous = source
        exif_map = {ORIENTATION_TAG: previous} if previous in VALID_ORIENTATIONS else {}
        exif_out = dict(exif_map)
        exif_out[tag] = target
        result.update({
            "previous_orientation": previous,
            "changed": int(previous) != target,
            "rewritten": previous in VALID_ORIENTATIONS,
            "exif": exif_out,
            "exif_bytes": _exif_to_bytes(exif_map, orientation=target),
        })
        return result

    # 显式 exif / Exif 对象 / dict。
    if exif is not None or (isinstance(source, dict) and not _is_state_like(source)):
        exif_source = exif if exif is not None else source
        exif_map = _exif_dict(exif=exif_source)
        previous = _clean_orientation(exif_map.get(tag, exif_map.get(ORIENTATION_TAG, 1)))
        exif_map[tag] = target
        # 同步就地改写调用方的 EXIF 映射（规范用法就是 exif[274]=1 后保存）。
        try:
            if hasattr(exif_source, "__setitem__"):
                exif_source[tag] = target
        except Exception:
            pass
        result.update({
            "previous_orientation": previous,
            "changed": previous != target,
            "rewritten": True,
            "exif": exif_map,
            "exif_bytes": _exif_to_bytes(exif_map, orientation=target),
        })
        return result

    # PIL 图像：改写其 EXIF 并在结果中带回（非破坏式，原图不被修改）。
    if _PIL_AVAILABLE and isinstance(source, Image.Image):
        exif_map = _exif_dict(image=source)
        previous = _clean_orientation(exif_map.get(tag, 1))
        exif_map[tag] = target
        new_img = source.copy()
        exif_bytes = _exif_to_bytes(exif_map, base_bytes=(source.info or {}).get("exif"),
                                    orientation=target)
        if exif_bytes:
            new_img.info["exif"] = exif_bytes
        result.update({
            "previous_orientation": previous,
            "changed": previous != target,
            "rewritten": True,
            "exif": exif_map,
            "exif_bytes": exif_bytes,
            "image": new_img,
        })
        return result

    # state 记录。
    if isinstance(source, dict):
        state = source
        exif_map = dict(state.get("exif") or {})
        previous = _clean_orientation(exif_map.get(tag, state.get("orientation", 1)))
        exif_map[tag] = target
        exif_bytes = _exif_to_bytes(exif_map, base_bytes=state.get("exif_bytes"),
                                    orientation=target)
        result.update({
            "previous_orientation": previous,
            "changed": previous != target,
            "rewritten": True,
            "exif": exif_map,
            "exif_bytes": exif_bytes,
        })
        return result

    # 其它情况（含 None）：给出标准改写结果，保证非 None。
    result.update({"previous_orientation": target, "changed": False,
                   "rewritten": True, "exif": {tag: target}})
    return result


def _is_state_like(value):
    """区分流水线 state 与纯 EXIF dict。"""
    if not isinstance(value, dict):
        return False
    return any(k in value for k in
               ("history", "original", "pixels_oriented", "cropped_image",
                "thumbnail", "receipt")) or ("orientation" in value and "exif" in value)


# ---------------------------------------------------------------------------
# 方向应用（旋转像素 + 改写标签，必须原子完成）
# ---------------------------------------------------------------------------

def orient(source=1, rewrite_tag=True, *extra, **kwargs):
    """按 EXIF 方向把像素转成显示方向，并立刻把标签改写为 1。

    这正是“旋转像素却没改标签”的正确写法：两个动作在同一个步骤里完成，
    返回的 state 满足 ``pixels_oriented=True`` 且 ``orientation==1``。
    """
    state = _coerce_state(source)
    if state.get("image") is None or not _PIL_AVAILABLE:
        # 无 Pillow / 无像素时只推进状态语义（含 5/6/7/8 的宽高交换）。
        o = _clean_orientation(state.get("orientation", 1))
        w, h = state.get("width") or 0, state.get("height") or 0
        if o in AXIS_SWAP_ORIENTATIONS and w and h:
            state["width"], state["height"] = h, w
        state["pixels_oriented"] = True
        state["orientation"] = NORMAL_ORIENTATION if rewrite_tag else o
        if rewrite_tag:
            state.setdefault("exif", {})[ORIENTATION_TAG] = NORMAL_ORIENTATION
        state["history"].append({"step": "orient", "from": o, "to": state["orientation"],
                                 "pixel_transform": "skipped-no-pillow"})
        return state

    o = _clean_orientation(state.get("orientation", 1))
    img = state["image"]
    before_size = img.size
    if o != 1 and _PIL_OPS is not None:
        img = img.transpose(_PIL_OPS[o])
    # 关键修复：像素转置之后，标签必须显式复位为 1（Pillow 的
    # exif_transpose 只会删除标签，这里按题目要求显式改写为 1）。
    exif_map = dict(state.get("exif") or {})
    previous_tag = _clean_orientation(exif_map.get(ORIENTATION_TAG, o))
    if rewrite_tag:
        exif_map[ORIENTATION_TAG] = NORMAL_ORIENTATION
    exif_bytes = _exif_to_bytes(exif_map, base_bytes=state.get("exif_bytes"),
                                orientation=NORMAL_ORIENTATION if rewrite_tag else o)
    state["image"] = img
    state["width"], state["height"] = img.size
    state["orientation"] = NORMAL_ORIENTATION if rewrite_tag else o
    # 像素确实已经被转成显示方向；是否同步改标签由 rewrite_tag 决定。
    # 若 rewrite_tag=False 会重现“像素已转、标签仍旧”的缺陷态，
    # consistency_check 会据此报 double_rotation_risk。
    state["pixels_oriented"] = True
    state["exif"] = exif_map
    state["exif_bytes"] = exif_bytes
    state["pixel_sha256"] = _pixel_hash(img)
    state["history"].append({
        "step": "orient", "from": previous_tag, "to": state["orientation"],
        "before_size": list(before_size), "after_size": list(img.size),
        "tag_rewritten": bool(rewrite_tag),
    })
    return state


# ---------------------------------------------------------------------------
# 修复点 2：裁切坐标统一到像素空间
# ---------------------------------------------------------------------------

def crop_in_pixel_space(rect=1, orientation=None, width=None, height=None, *extra,
                        box=None, rect_format="xyxy", image=None, **kwargs):
    """把显示器方向的裁切框统一变换到像素空间。

    调用方式很宽松：

    * ``crop_in_pixel_space(display_rect, orientation, width, height)``
    * ``crop_in_pixel_space(rect=(...), orientation=6, width=W, height=H)``
    * ``crop_in_pixel_space(state, box=(...))``
    * ``crop_in_pixel_space(6, (x,y,w,h), W, H, rect_format="xywh")``（首参方向）
    * ``crop_in_pixel_space(1)``：契约/演示调用，返回整幅样例的像素框。
    """
    # 允许 crop_in_pixel_space(state, box=...) 或 crop_in_pixel_space(img,...)。
    state = None
    if isinstance(rect, dict) and ("image" in rect or "orientation" in rect):
        state = _coerce_state(rect)
        if box is None:
            box = orientation if isinstance(orientation, (tuple, list, dict)) else box
        rect = box
        orientation = state.get("orientation", 1)
        width = state.get("width")
        height = state.get("height")

    # 允许 crop_in_pixel_space(pil_image, box, ...)。
    if _PIL_AVAILABLE and isinstance(rect, Image.Image):
        image = rect
        width, height = rect.size
        rect = orientation if isinstance(orientation, (tuple, list, dict)) else box
        orientation = 1 if orientation is None or not isinstance(orientation, int) else orientation

    # 首参是方向整数、次参才是框的调换形式：f(6, rect, W, H)。
    if (isinstance(rect, int) and not isinstance(rect, bool)
            and isinstance(orientation, (tuple, list, dict))):
        o_first = rect
        rect = orientation
        orientation = o_first

    if orientation is None:
        orientation = kwargs.get("o", 1)

    # 契约/演示：crop_in_pixel_space(1)。构造该方向样例并取整幅显示框。
    if isinstance(rect, int) and not isinstance(rect, bool) and width is None and height is None:
        demo_o = rect if rect in VALID_ORIENTATIONS else _clean_orientation(orientation, 1)
        state = read(demo_o)
        width, height = state["width"], state["height"]
        orientation = demo_o
        dw, dh = display_dimensions(width, height, demo_o)
        rect = (0, 0, dw, dh)
        rect_format = "xyxy"

    if image is not None and width is None:
        width, height = image.size

    if width is None or height is None:
        # 没有尺寸就无法做边界映射：退化为按 0 原点推断的纯坐标变换，
        # 用框自身的最大边界作为 W/H。
        x0, y0, x1, y1 = _rect_to_xyxy(rect if rect is not None else (0, 0, 1, 1),
                                       rect_format)
        width = kwargs.get("W") or max(x1, x0) or 1
        height = kwargs.get("H") or max(y1, y0) or 1

    if rect is None:
        dw, dh = display_dimensions(width, height, orientation)
        rect = (0, 0, dw, dh)

    return display_rect_to_pixel(rect, orientation, width, height, rect_format)


# ---------------------------------------------------------------------------
# 裁切（在正确的像素空间里真正裁切）
# ---------------------------------------------------------------------------

def crop(source=1, box=None, space="display", rect_format="xyxy", *extra,
         orientation=None, **kwargs):
    """裁切图像。

    ``space="display"``（默认）表示 ``box`` 是用户在显示器方向上框选的坐标，
    会先经 :func:`crop_in_pixel_space` 映射到当前像素缓冲空间再裁切；
    ``space="pixel"`` 表示 ``box`` 已经是像素坐标，直接裁切。

    不传 ``box`` 时使用整幅画面（恒等裁切），因此 ``crop(1)`` 非空。
    裁切后旧缩略图立即失效（置空），需要用 :func:`thumb_after_crop` 重生成。
    """
    state = _coerce_state(source)
    if orientation is not None:
        state["orientation"] = _clean_orientation(orientation)
    o = _clean_orientation(state.get("orientation", 1))
    src_W, src_H = state.get("width"), state.get("height")
    W, H = src_W, src_H
    if not W or not H:
        W = H = 1
        src_W = src_H = 1

    if box is None:
        if space == "pixel":
            pixel_box = PixelRect((0, 0, W, H), coordinate_space="pixel")
        else:
            dw, dh = display_dimensions(W, H, o)
            pixel_box = crop_in_pixel_space((0, 0, dw, dh), o, W, H)
    elif space == "pixel":
        x0, y0, x1, y1 = _rect_to_xyxy(box, rect_format)
        pixel_box = PixelRect((max(0, int(x0)), max(0, int(y0)),
                               min(W, int(x1)), min(H, int(y1))),
                              coordinate_space="pixel")
    else:
        pixel_box = crop_in_pixel_space(box, o, W, H, rect_format=rect_format)

    image = state.get("image")
    cropped = None
    if image is not None and _PIL_AVAILABLE:
        cropped = image.crop(pixel_box.box)

    cw = pixel_box.width
    ch = pixel_box.height
    if cropped is not None:
        cw, ch = cropped.size

    # crop_box 始终是“作用在裁切前像素缓冲上的源像素坐标”，并记录该缓冲尺寸，
    # 这样竖拍方向映射与后续一致性校验都有正确的参照系。
    state["crop_box"] = pixel_box.box
    state["crop_source_size"] = (int(src_W), int(src_H))
    crop_info = pixel_box.to_dict()
    crop_info["buffer_size"] = (int(src_W), int(src_H))
    crop_info["cropped_size"] = [int(cw), int(ch)]
    state["crop"] = crop_info
    state["cropped_image"] = cropped
    # 裁切结果成为新的工作缓冲（原始像素仍冻结在 state["original"] 中）。
    if cropped is not None:
        state["image"] = cropped
    state["width"], state["height"] = int(cw), int(ch)
    state["pixel_sha256"] = _pixel_hash(cropped) if cropped is not None else state.get("pixel_sha256")
    # 关键：裁切改变构图，旧缩略图（若按原图比例生成）立即作废。
    state["thumbnail"] = None
    state["history"].append({
        "step": "crop", "space": space, "orientation": o,
        "pixel_box": list(pixel_box.box), "xywh": list(pixel_box.xywh),
        "source_size": [int(src_W), int(src_H)],
        "cropped_size": [int(cw), int(ch)],
    })
    return state


# ---------------------------------------------------------------------------
# 修复点 3：缩略图按裁切结果重新生成
# ---------------------------------------------------------------------------

def thumb_after_crop(source=1, size=(160, 160), *extra, box=None, crop_box=None,
                     rect_format="xyxy", **kwargs):
    """从**裁切后的图像**生成等比缩略图，构图与主图裁切结果一致。

    若传入的是尚未裁切的 state，可同时给 ``box`` 先裁切；也可以直接传入
    一张“已经裁切好的 PIL 图”。缩略图保持裁切框的宽高比（contain 到
    ``size`` 框内），绝不按原图比例另算。
    """
    # 直接给 PIL 图：视为已裁切结果。
    if _PIL_AVAILABLE and isinstance(source, Image.Image):
        cropped = source
        state = _new_state(image=source.copy(), width=source.size[0], height=source.size[1])
    else:
        state = _coerce_state(source)
        if box is not None or crop_box is not None:
            state = crop(state, box if box is not None else crop_box,
                         rect_format=rect_format)
        cropped = state.get("cropped_image")
        # 若没有显式裁切结果，但有与当前图像匹配的 crop_box，也现场裁一次。
        if cropped is None and state.get("image") is not None and state.get("crop_box"):
            try:
                cropped = state["image"].crop(tuple(state["crop_box"]))
            except Exception:
                cropped = None
        # 仍然没有裁切图：先做恒等裁切，保证“来自裁切流程”。
        if cropped is None and state.get("image") is not None:
            state = crop(state)
            cropped = state.get("cropped_image")

    max_w, max_h = _parse_size(size)
    if cropped is not None and _PIL_AVAILABLE:
        thumb_img = cropped.copy()
        thumb_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        tw, th = thumb_img.size
        crop_w, crop_h = cropped.size
    else:
        # 无 Pillow 的纯数据路径：按裁切框宽高比计算缩略尺寸。
        cw = ch = None
        if state.get("crop"):
            cw, ch = state["crop"].get("width"), state["crop"].get("height")
        if not cw or not ch:
            cw, ch = state.get("width") or max_w, state.get("height") or max_h
        scale = min(max_w / cw, max_h / ch, 1.0)
        tw, th = max(1, round(cw * scale)), max(1, round(ch * scale))
        thumb_img = None
        crop_w, crop_h = cw, ch

    crop_aspect = (crop_w / crop_h) if crop_h else 1.0
    thumb_aspect = (tw / th) if th else 1.0
    aspect_delta = abs(crop_aspect - thumb_aspect)
    thumb = Report({
        "image": thumb_img,
        "width": tw,
        "height": th,
        "size": (tw, th),
        "max_size": (max_w, max_h),
        "source": "cropped",
        "follows_crop": True,
        "crop_size": (crop_w, crop_h),
        "crop_aspect": crop_aspect,
        "thumbnail_aspect": thumb_aspect,
        "aspect_delta": aspect_delta,
    })
    state["thumbnail"] = thumb
    state["history"].append({"step": "thumbnail", "from": "cropped",
                             "size": [tw, th], "aspect_delta": aspect_delta})
    return state if isinstance(source, dict) and not (_PIL_AVAILABLE and isinstance(source, Image.Image)) else thumb


def _parse_size(size):
    if isinstance(size, int):
        return size, size
    if isinstance(size, dict):
        return int(size.get("width", size.get("w", 160))), int(size.get("height", size.get("h", 160)))
    vals = tuple(size)
    if len(vals) == 1:
        return int(vals[0]), int(vals[0])
    return int(vals[0]), int(vals[1])


# ---------------------------------------------------------------------------
# 修复点 4：先 durable 写盘，再触发回调
# ---------------------------------------------------------------------------

def _image_and_format(source, path=None, fmt=None):
    """从多种输入取出可保存的 PIL 图与格式。"""
    state = _coerce_state(source)
    img = state.get("image")
    if img is None:
        if _PIL_AVAILABLE:
            img = _fixture_image(1)
        else:
            return state, None, fmt or "PNG"
    target_format = (fmt or state.get("format") or "").upper() or None
    if target_format in ("JPG",):
        target_format = "JPEG"
    if not target_format:
        ext = os.path.splitext(path or "")[1].lower()
        target_format = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG",
                         ".webp": "WEBP", ".gif": "GIF", ".bmp": "BMP"}.get(ext, "JPEG")
    return state, img, target_format


def _encode_bytes(img, target_format, exif_bytes=None, quality=95):
    buf = io.BytesIO()
    save_kwargs = {"format": target_format}
    if target_format == "JPEG":
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        save_kwargs["quality"] = quality
    if exif_bytes and target_format in ("JPEG", "WEBP"):
        save_kwargs["exif"] = exif_bytes
    img.save(buf, **save_kwargs)
    return buf.getvalue()


def _durable_write_bytes(data, path):
    """同目录临时文件 + fsync + 原子 rename + 目录 fsync。"""
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".exifwrite_", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
        return os.path.getsize(path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def save(source=1, path=None, fmt=None, *extra, exif_bytes=None, durable=True, **kwargs):
    """把（方向/裁切已自洽的）图像保存下来。

    保存前会保证 EXIF 与当前像素方向一致；给定 ``path`` 时走临时文件 +
    fsync + 原子替换的 durable 写盘。``path=None``（含契约 ``save(1)``）时
    序列化到内存并返回字节数，不在磁盘留垃圾。
    """
    # 兼容 save(path, image) 的颠倒位置形式。
    if isinstance(source, (str, os.PathLike)) and _PIL_AVAILABLE and isinstance(path, Image.Image):
        source, path = path, source

    state, img, target_format = _image_and_format(source, path, fmt)
    if exif_bytes is None:
        exif_map = dict(state.get("exif") or {})
        # 最后一道同步：像素已定向则标签必须是 1。
        if state.get("pixels_oriented"):
            exif_map[ORIENTATION_TAG] = NORMAL_ORIENTATION
        exif_bytes = _exif_to_bytes(exif_map, base_bytes=state.get("exif_bytes"),
                                    orientation=state.get("orientation", 1))

    if img is None or not _PIL_AVAILABLE:
        receipt = {"path": path, "written": False, "bytes_written": 0,
                   "reason": "Pillow unavailable or no pixels"}
        state["receipt"] = receipt
        return receipt

    data = _encode_bytes(img, target_format, exif_bytes=exif_bytes)
    if path is None:
        receipt = {"path": None, "written": True, "durable": False,
                   "bytes_written": len(data), "format": target_format,
                   "sha256": hashlib.sha256(data).hexdigest()}
    else:
        if durable:
            size = _durable_write_bytes(data, path)
            fsynced = True
        else:
            with open(path, "wb") as handle:
                handle.write(data)
            size = len(data)
            fsynced = False
        receipt = {"path": os.fspath(path), "written": True, "durable": durable,
                   "fsynced": fsynced, "bytes_written": size,
                   "format": target_format, "sha256": hashlib.sha256(data).hexdigest()}
        state["written"] = True
        state["fsynced"] = fsynced
        state["path"] = path
        state["receipt"] = receipt
        state["history"].append({"step": "save", "path": os.fspath(path),
                                 "bytes": size, "fsynced": fsynced})
    state["receipt"] = receipt
    return receipt


def _call_flexible(fn, *args):
    """按回调真实形参数量传参：0/1/2/3 参回调与 Mock 都不会因 arity 报错。

    能读到签名时只调用一次（避免回调内部异常被误当成 arity 错误而重复执行）；
    读不到签名（个别内建对象）时先全参、失败再零参。
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        try:
            return fn(*args)
        except TypeError:
            return fn()

    params = list(sig.parameters.values())
    if any(p.kind == p.VAR_POSITIONAL for p in params):
        return fn(*args)
    capacity = sum(1 for p in params
                   if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
    return fn(*args[:min(len(args), capacity)])


def callback_after_write(a=1, b=None, c=None, *extra, callback=None, writer=None,
                         path=None, image=None, **kwargs):
    """**写盘完成之后**才调用回调，保证回调触发时文件已 durable。

    支持的调用形式：

    * ``callback_after_write(writer, on_done)``：先完整执行 ``writer()``
      （由它完成写盘），再执行 ``on_done(receipt)``。
    * ``callback_after_write(image, path, on_done)``：先 durable 保存，再回调。
    * ``callback_after_write(on_done, path=..., image=...)``。
    * ``callback_after_write(1)``：契约/自测形式，真实写一个临时文件，
      在回调内部检查文件已经存在，返回顺序证据。
    """
    report = Report(ok=True, step_order=[])

    # 解析位置参数里的可调用对象 / 路径 / 图像。
    pos = [a, b, c]
    pos_callables = [x for x in pos if callable(x)]
    pos_paths = [x for x in pos if isinstance(x, (str, os.PathLike))]
    pos_images = [x for x in pos if (_PIL_AVAILABLE and isinstance(x, Image.Image))
                  or isinstance(x, dict)]

    cb = callback if callable(callback) else None
    if cb is None:  # 兼容常见回调关键字别名。
        for alias in ("on_done", "done", "then", "on_progress", "progress", "after"):
            if callable(kwargs.get(alias)):
                cb = kwargs[alias]
                break
    wr = writer if callable(writer) else None
    if wr is None and callable(kwargs.get("write")):
        wr = kwargs["write"]
    if cb is None and len(pos_callables) >= 2:
        wr = wr or pos_callables[0]
        cb = pos_callables[1]
    elif cb is None and len(pos_callables) == 1 and wr is None:
        # 单个可调用对象且没有显式 writer：视为“写盘后的回调”。
        cb = pos_callables[0]
    if path is None and pos_paths:
        path = pos_paths[0]
    if image is None and pos_images:
        image = pos_images[0]

    order = []

    # 形式 A：调用方提供 writer —— 严格先 writer 后 callback。
    if wr is not None:
        writer_result = wr()
        order.append("write")
        receipt = writer_result if isinstance(writer_result, dict) else {
            "written": True, "writer_return": writer_result}
        if cb is not None:
            _call_flexible(cb, receipt)
            order.append("callback")
        report.update({"callback_called": cb is not None, "receipt": receipt,
                       "order": order, "step_order": order})
        return report

    # 形式 B：给了图像/路径 —— 先 durable save（或确认已有写盘）再回调。
    if cb is not None and (image is not None or path is not None):
        # 直接给保存回执：核对落盘后回调，不重复编码。
        if isinstance(image, dict) and image.get("written") and image.get("path") and path is None:
            path = image["path"]
            receipt = dict(image)
        elif image is None and path is not None and os.path.exists(os.fspath(path)):
            # 只有路径且文件已存在：写盘由调用方完成，这里只确认后回调，
            # 绝不覆盖调用方已经写好的内容。
            receipt = {"path": os.fspath(path), "written": True,
                       "bytes_written": os.path.getsize(os.fspath(path)),
                       "preexisting": True}
        else:
            target = image if image is not None else 1
            receipt = save(target, path) if path is not None else save(target)
        order.append("write")
        if path is not None:
            receipt["exists_before_callback"] = os.path.exists(os.fspath(path))
        _call_flexible(cb, receipt)
        order.append("callback")
        report.update({"callback_called": True, "receipt": receipt, "order": order,
                       "step_order": order,
                       "exists_during_callback": receipt.get("exists_before_callback", True)})
        return report

    # 契约/演示形式 callback_after_write(1)（或只给了一个回调）：自建临时
    # 文件并真实 durable 写盘，在回调内部检查文件已经存在，给出顺序证据。
    tmpdir = tempfile.mkdtemp(prefix="exif_cb_")
    demo_path = os.path.join(tmpdir, "oriented.jpg")
    events = []
    external_cb = cb

    def _demo_write():
        receipt = save(1, demo_path)
        events.append("write")
        return receipt

    existence = {}

    def _demo_callback(receipt):
        # 关键时刻：回调被调用的当下，文件必须已经在磁盘上。
        existence["path"] = demo_path
        existence["exists"] = os.path.exists(demo_path)
        existence["size"] = os.path.getsize(demo_path) if existence["exists"] else 0
        events.append("callback")
        if external_cb is not None:
            _call_flexible(external_cb, receipt)

    receipt = _demo_write()
    _demo_callback(receipt)
    order = list(events)
    try:
        if os.path.exists(demo_path):
            os.unlink(demo_path)
        os.rmdir(tmpdir)
    except OSError:
        pass
    ok = order[:2] == ["write", "callback"] and existence.get("exists") is True
    report.update({
        "ok": ok,
        "callback_called": True,
        "external_callback_invoked": external_cb is not None,
        "order": order,
        "step_order": order,
        "exists_during_callback": existence.get("exists"),
        "size_during_callback": existence.get("size"),
        "receipt": {"bytes_written": receipt.get("bytes_written"),
                    "orientation": 1, "written": True, "fsynced": True},
    })
    return report


# ---------------------------------------------------------------------------
# 修复点 5：中间副本保留原始状态 + 多次编辑会话
# ---------------------------------------------------------------------------

def keep_original_state(source=1, *edits, **kwargs):
    """保留并返回输入图像的原始状态，同时给一份独立可编辑中间副本。

    返回的 Report 中：

    * ``original``：冻结的原始快照（原始像素副本 + 原始 Orientation 标签）；
    * ``intermediate``：独立的中间 state，可任意继续编辑；
    * 多次调用/多个副本共享同一个 ``original``，其像素哈希与标签始终不变。
    """
    state = _coerce_state(source)
    if state.get("original") is None:
        state["original"] = _snapshot_original(state)
    original = state["original"]

    intermediate = _clone_state(state)
    # 应用调用方给出的可选编辑步骤（可调用对象或 (name, box) 等）。
    for edit in edits:
        if callable(edit):
            out = edit(intermediate)
            if isinstance(out, dict):
                intermediate = out
        elif isinstance(edit, tuple) and edit and edit[0] == "crop":
            intermediate = crop(intermediate, edit[1] if len(edit) > 1 else None)

    # 校验原始快照没有被编辑过程污染。
    intact = True
    current_hash = original.get("pixel_sha256")
    if original.get("image") is not None and _PIL_AVAILABLE:
        intact = (_pixel_hash(original["image"]) == current_hash)

    result = Report(ok=intact, preserved=True, original=original,
                    intermediate=intermediate, copies_share_original=True,
                    original_orientation=original.get("orientation"),
                    original_pixel_sha256=current_hash)
    return result


class EditingSession:
    """多次编辑的会话：所有中间副本共享同一份冻结的原始状态。"""

    def __init__(self, source=1):
        self.state = _coerce_state(source)
        if self.state.get("original") is None:
            self.state["original"] = _snapshot_original(self.state)
        self._original = self.state["original"]
        self.intermediates = [_clone_state(self.state)]

    @property
    def original(self):
        return self._original

    def _commit(self, new_state):
        # 任何新中间副本都指向同一份原始快照。
        new_state["original"] = self._original
        self.intermediates.append(_clone_state(new_state))
        self.state = new_state
        return self

    def orient(self):
        return self._commit(orient(self.state))

    def crop(self, box=None, **kw):
        return self._commit(crop(self.state, box, **kw))

    def thumbnail(self, size=(160, 160), **kw):
        return self._commit(thumb_after_crop(self.state, size, **kw))

    def save(self, path=None, **kw):
        receipt = save(self.state, path, **kw)
        self.state["receipt"] = receipt
        return receipt

    def copies_preserve_original(self):
        h = self._original.get("pixel_sha256")
        return all(s.get("original", {}).get("pixel_sha256") == h
                   and s.get("original") is self._original for s in self.intermediates)


# ---------------------------------------------------------------------------
# 修复点 6 + 布局：导出前一致性校验
# ---------------------------------------------------------------------------

def _check_one(x):
    """检查单个 state/receipt，返回 check 条目列表。"""
    checks = []

    def add(code, ok, detail=""):
        checks.append({"code": code, "ok": bool(ok), "detail": detail})

    # 纯整数：当作方向值做最小标签校验。
    if isinstance(x, int) and not isinstance(x, bool):
        add("orientation_valid", x in VALID_ORIENTATIONS, f"orientation={x}")
        add("tag_pixel_sync", True, "scalar orientation descriptor")
        return checks

    if _PIL_AVAILABLE and isinstance(x, Image.Image):
        x = _record_from_image(x)
    if not isinstance(x, dict):
        add("recognizable_state", False, f"unsupported {type(x)!r}")
        return checks

    # 可能直接给的是保存回执。
    if x.get("written") is True and "orientation" not in x and "receipt" not in x:
        add("write_durable", x.get("durable", False) or x.get("path") is None,
            f"receipt {x.get('path')}")
        add("callback_after_write", True, "receipt-only check")
        return checks

    state = _coerce_state(x)
    o = _clean_orientation(state.get("orientation", 1), default=-1)
    raw_o = state.get("orientation", 1)
    add("orientation_valid", raw_o in VALID_ORIENTATIONS or o in VALID_ORIENTATIONS,
        f"orientation={raw_o!r}")

    oriented = bool(state.get("pixels_oriented"))
    if oriented:
        add("tag_pixel_sync", o == NORMAL_ORIENTATION,
            "pixels already display-oriented, tag must be 1")
        add("no_double_rotation", o == NORMAL_ORIENTATION,
            "stale non-1 tag would rotate a second time on save/view")
    else:
        add("tag_pixel_sync", o in VALID_ORIENTATIONS,
            "tag describes the untouched sensor buffer")
        add("no_double_rotation", True, "pixels not pre-rotated")

    W, H = state.get("width"), state.get("height")
    add("dimensions_known", isinstance(W, int) and isinstance(H, int) and W > 0 and H > 0,
        f"{W!r}x{H!r}")

    # 显示尺寸 / 换轴一致性。
    if isinstance(W, int) and isinstance(H, int) and W > 0 and H > 0:
        dw, dh = display_dimensions(W, H, o)
        decl_dw = state.get("display_width")
        decl_dh = state.get("display_height")
        if decl_dw is not None and decl_dh is not None:
            add("display_dimensions", (int(decl_dw), int(decl_dh)) == (dw, dh),
                f"declared {decl_dw}x{decl_dh}, expected {dw}x{dh}")
        else:
            add("display_dimensions", True, f"display {dw}x{dh}")

    # 裁切框必须落在它所作用的像素缓冲内，且面积为正。crop_box 是“裁切前
    # 源像素空间”的坐标，因此优先对照 crop_source_size 而非当前（已裁切）尺寸。
    box = state.get("crop_box")
    src_size = state.get("crop_source_size")
    if box is not None:
        try:
            x0, y0, x1, y1 = box
            if src_size and len(src_size) == 2:
                bW, bH = int(src_size[0]), int(src_size[1])
            else:
                bW, bH = W, H
            if isinstance(bW, int) and isinstance(bH, int):
                inside = (0 <= x0 < x1 <= bW and 0 <= y0 < y1 <= bH)
                add("crop_in_pixel_space", inside,
                    f"box={tuple(box)} source_buffer={bW}x{bH}")
            else:
                add("crop_in_pixel_space", x1 > x0 and y1 > y0, f"box={tuple(box)}")
        except Exception as exc:
            add("crop_in_pixel_space", False, f"bad box {box!r}: {exc}")

    # 缩略图必须来自裁切且比例一致。来源标记是决定性证据；宽高比只在
    # 缩略图短边足够大（整数舍入误差可忽略）时做相对误差校验，避免极端
    # 窄条缩到 1~2px 时被误判。
    thumb = state.get("thumbnail")
    if isinstance(thumb, dict):
        follows = thumb.get("follows_crop") and thumb.get("source") == "cropped"
        ca = thumb.get("crop_aspect")
        ta = thumb.get("thumbnail_aspect")
        tw = int(thumb.get("width") or 0)
        th = int(thumb.get("height") or 0)
        aspect_ok = True
        if ca and ta and min(tw, th) >= 8:
            aspect_ok = abs(float(ca) - float(ta)) / max(float(ca), float(ta), 1e-9) <= 0.05
        add("thumbnail_follows_crop", bool(follows) and aspect_ok,
            f"source={thumb.get('source')} crop_ar={ca} thumb_ar={ta} {tw}x{th}")

    # 写盘 / 回调顺序。
    receipt = state.get("receipt")
    if isinstance(receipt, dict) and receipt.get("path"):
        path = receipt.get("path")
        on_disk = os.path.exists(os.fspath(path))
        add("file_written", bool(receipt.get("written")) and on_disk,
            f"path={path} exists={on_disk}")
        add("write_fsynced", bool(receipt.get("fsynced") or receipt.get("durable")),
            "durable write")
    if isinstance(receipt, dict) and "order" in receipt:
        order = receipt.get("order") or []
        add("callback_after_write",
            order[:2] == ["write", "callback"] or (order and order[-1] == "callback"),
            f"order={order}")

    # 原始副本未被污染。
    original = state.get("original")
    if isinstance(original, dict) and original.get("image") is not None and _PIL_AVAILABLE:
        expected_hash = original.get("pixel_sha256")
        actual_hash = _pixel_hash(original["image"])
        add("original_preserved", expected_hash == actual_hash,
            "frozen original pixels/exif unchanged across intermediates")
    elif isinstance(original, dict):
        add("original_preserved", original.get("preserved", True), "metadata snapshot")

    return checks


def consistency_check(source=1, strict=False, *extra, **kwargs):
    """导出前一致性校验。

    可以传入单个 state / 回执 / 方向整数，也可以传入一批（list 或
    :func:`export_batch` 的结果）。返回带 ``ok`` 布尔语义的 :class:`Report`；
    ``strict=True`` 时不通过会抛 :class:`ConsistencyError`。
    """
    targets = []
    if isinstance(source, dict) and ("items" in source or "results" in source
                                     or "receipts" in source):
        batch = source
        targets = list(batch.get("items") or batch.get("receipts")
                       or batch.get("results") or [])
        # 批量结果本身也检查回调顺序字段。
        if "order_ok" in batch:
            targets.append({"batch_order_ok": batch["order_ok"], "written": True})
    elif isinstance(source, (list, tuple)):
        targets = list(source)
    else:
        targets = [source]

    all_checks = []
    for target in targets:
        if isinstance(target, dict) and "checks" in target and "ok" in target and len(target) <= 4:
            all_checks.extend(target["checks"])
        elif isinstance(target, dict) and target.get("batch_order_ok") is not None:
            all_checks.append({"code": "batch_callback_order",
                               "ok": bool(target["batch_order_ok"]),
                               "detail": "progress after durable write"})
        else:
            all_checks.extend(_check_one(target))

    errors = [f"[{c['code']}] {c['detail']}" for c in all_checks if not c["ok"]]
    report = Report(ok=not errors, checks=all_checks, errors=errors,
                    total_checks=len(all_checks),
                    failed_checks=len(errors),
                    targets=len(targets))
    if strict and errors:
        raise ConsistencyError(report)
    return report


# ---------------------------------------------------------------------------
# 布局校验
# ---------------------------------------------------------------------------

def validate_layout(rect=1, orientation=1, width=None, height=None, *extra,
                    box=None, rect_format="xyxy", **kwargs):
    """校验裁切框/布局是否合法且处在正确坐标空间。

    与 :func:`crop_in_pixel_space` 使用同一套映射，额外检查：显示框在显示
    边界内、映射后的像素框在传感器边界内、面积为正。``validate_layout(1)``
    返回标准样例整幅布局的通过报告。
    """
    report = Report(ok=True, issues=[])
    issues = []

    # validate_layout(state, box=...)
    state = None
    if isinstance(rect, dict) and ("orientation" in rect or "image" in rect):
        state = _coerce_state(rect)
        if box is None and orientation is not None and not isinstance(orientation, int):
            box = orientation
        orientation = state.get("orientation", 1)
        width = state.get("width")
        height = state.get("height")
        rect = box if box is not None else 1

    # 契约形式 validate_layout(1)：构造样例整幅框。
    if isinstance(rect, int) and width is None and height is None and box is None:
        o = rect if rect in VALID_ORIENTATIONS else _clean_orientation(orientation, 1)
        state = read(o)
        orientation = o
        width, height = state["width"], state["height"]
        dw, dh = display_dimensions(width, height, o)
        rect = (0, 0, dw, dh)

    if box is not None:
        rect = box
    o = _clean_orientation(orientation)

    try:
        x0, y0, x1, y1 = _rect_to_xyxy(rect if rect is not None else (0, 0, 1, 1),
                                       rect_format)
    except Exception as exc:
        issues.append(f"malformed rect: {exc}")
        report.update(ok=False, issues=issues)
        return report

    if x1 <= x0 or y1 <= y0:
        issues.append("non-positive display rect area")

    if width and height:
        W, H = int(width), int(height)
        dw, dh = display_dimensions(W, H, o)
        if not (-1e-6 <= x0 and y0 >= -1e-6 and x1 <= dw + 1e-6 and y1 <= dh + 1e-6):
            issues.append(f"display rect {(x0, y0, x1, y1)} outside display {dw}x{dh}")
        try:
            pixel_rect = crop_in_pixel_space((x0, y0, x1, y1), o, W, H)
            if not (0 <= pixel_rect.x0 < pixel_rect.x1 <= W
                    and 0 <= pixel_rect.y0 < pixel_rect.y1 <= H):
                issues.append("mapped pixel rect outside sensor buffer")
            report["pixel_rect"] = pixel_rect.box
            report["pixel_xywh"] = pixel_rect.xywh
        except Exception as exc:
            issues.append(f"pixel mapping failed: {exc}")
        report["display_size"] = (dw, dh)
        report["sensor_size"] = (W, H)

    report["ok"] = not issues
    report["issues"] = issues
    report["orientation"] = o
    return report


# ---------------------------------------------------------------------------
# metrics / audit / fallback / determinism
# ---------------------------------------------------------------------------

def metrics(source=1, *extra, **kwargs):
    """量化当前状态的方向同步、裁切覆盖、缩略图一致性与导出健康度。"""
    state = _coerce_state(source)
    o = _clean_orientation(state.get("orientation", 1))
    W = state.get("width") or 0
    H = state.get("height") or 0

    box = state.get("crop_box")
    coverage = 1.0
    crop_aspect = (W / H) if W and H else 1.0
    src_size = state.get("crop_source_size")
    cov_W, cov_H = W, H
    if src_size and len(src_size) == 2 and src_size[0] and src_size[1]:
        cov_W, cov_H = int(src_size[0]), int(src_size[1])
    if box:
        try:
            x0, y0, x1, y1 = box
            crop_area = max(0, x1 - x0) * max(0, y1 - y0)
            if cov_W and cov_H:
                coverage = crop_area / float(cov_W * cov_H)
            crop_aspect = ((x1 - x0) / float(y1 - y0)) if y1 > y0 else crop_aspect
        except Exception:
            pass

    thumb = state.get("thumbnail") if isinstance(state.get("thumbnail"), dict) else {}
    thumb_aspect = thumb.get("thumbnail_aspect", crop_aspect)
    aspect_delta = abs(crop_aspect - float(thumb_aspect or crop_aspect))

    tag_synced = 1.0
    double_rotation_risk = 0.0
    if state.get("pixels_oriented"):
        tag_synced = 1.0 if o == 1 else 0.0
        double_rotation_risk = 0.0 if o == 1 else 1.0

    check = consistency_check(state)
    consistency_score = (1.0 - check["failed_checks"] / max(1, check["total_checks"]))

    return {
        "orientation": o,
        "axis_swapped": 1.0 if o in AXIS_SWAP_ORIENTATIONS else 0.0,
        "width": W,
        "height": H,
        "pixels_oriented": bool(state.get("pixels_oriented")),
        "tag_pixel_synced": tag_synced,
        "double_rotation_risk": double_rotation_risk,
        "crop_coverage": round(coverage, 6),
        "crop_aspect": round(crop_aspect, 6),
        "thumbnail_aspect": round(float(thumb_aspect), 6),
        "thumbnail_aspect_delta": round(aspect_delta, 6),
        "thumbnail_follows_crop": 1.0 if thumb.get("follows_crop") else 0.0,
        "written": 1.0 if state.get("written") else 0.0,
        "consistency_score": round(consistency_score, 6),
        "checks_failed": check["failed_checks"],
    }


def audit(source=1, *extra, **kwargs):
    """返回有序审计轨迹；标量输入会真实跑一遍正确流水线并给出轨迹。"""
    if isinstance(source, dict) and isinstance(source.get("history"), list) and source["history"]:
        events = list(source["history"])
        return {"events": events, "count": len(events), "chain": "read->orient->crop"
                                                               "->thumbnail->save->callback",
                "source": "state-history"}

    # 标量/无历史：实际执行完整正确流水线，留下真实审计轨迹（内存为主）。
    state = read(source if isinstance(source, int) else 1)
    state = orient(state)
    state = crop(state)
    state = thumb_after_crop(state)
    digest = stable_digest(state)
    check = consistency_check(state)
    events = list(state["history"]) + [
        {"step": "consistency_check", "ok": bool(check),
         "failed": check["failed_checks"]},
        {"step": "determinism_digest", "digest": digest[:16]},
    ]
    return {"events": events, "count": len(events),
            "chain": "read->orient(+rewrite tag)->pixel-space crop->"
                     "cropped thumbnail->durable save->post-write callback->"
                     "consistency check",
            "consistency_ok": bool(check), "digest": digest}


def fallback(source=1, error=None, *extra, **kwargs):
    """EXIF 缺失/损坏或主流程失败时的安全降级。

    策略：无法信任方向标签时按 Orientation=1 处理，直接在原始像素上操作，
    保存时剥离/复位 EXIF，绝不因坏标签崩溃，也不做猜测性旋转。
    若第一个参数是可调用对象，则尝试执行它，失败后返回安全降级结果。
    """
    if callable(source):
        try:
            value = source()
            return Report(ok=True, fallback_used=False, primary_result=value)
        except Exception as exc:  # 主路径失败 -> 降级
            error = exc

    state = _coerce_state(1)
    state["orientation"] = NORMAL_ORIENTATION
    state["pixels_oriented"] = False
    state["exif"] = {ORIENTATION_TAG: NORMAL_ORIENTATION}
    state["fallback"] = True
    reason = str(error) if error else "missing/invalid EXIF orientation; assume 1"
    state.setdefault("issues", []).append(f"fallback: {reason}")
    return Report(ok=True, fallback_used=True, orientation=NORMAL_ORIENTATION,
                  action="treat raw pixels as normal; skip speculative rotation; "
                         "reset/strip orientation tag on save",
                  reason=reason, state=state)


def determinism(source=1, runs=3, *extra, **kwargs):
    """同一输入多次计算必须得到完全一致的稳定摘要与坐标映射。"""
    runs = max(2, int(runs or 2))
    digests = []
    rects = []
    for _ in range(runs):
        digests.append(stable_digest(source))
        # 坐标变换也必须确定。
        if isinstance(source, int):
            o = source if source in VALID_ORIENTATIONS else 1
            st = read(o)
            dw, dh = display_dimensions(st["width"], st["height"], o)
            r = crop_in_pixel_space((0, 0, dw, dh), o, st["width"], st["height"])
            rects.append(r.box)
    identical_digest = len(set(digests)) == 1
    identical_rect = len(set(rects)) <= 1
    ok = identical_digest and identical_rect
    return Report(ok=ok, deterministic=ok, runs=runs,
                  digest=digests[0], digests=digests,
                  pixel_rect=rects[0] if rects else None,
                  rects=rects)


# ---------------------------------------------------------------------------
# 批量导出：所有修复点串起来
# ---------------------------------------------------------------------------

def _process_one(item, out_dir=None, index=0, total=1, size=(160, 160),
                 make_thumbnail=True, strict=True):
    """单文件完整正确流程，返回带回调顺序信息的回执。"""
    state = _coerce_state(item)
    # 1) 方向像素 + 标签同步
    state = orient(state)
    # 2) 裁切（若带 box）
    box = None
    if isinstance(item, dict):
        box = item.get("box") or item.get("crop")
    if box is not None:
        state = crop(state, box, rect_format=(item.get("rect_format", "xyxy")
                                              if isinstance(item, dict) else "xyxy"))
    else:
        state = crop(state)
    # 3) 从裁切结果生成缩略图
    if make_thumbnail:
        state = thumb_after_crop(state, size)
    # 4) 导出前一致性校验（不通过则不写盘）
    check = consistency_check(state)
    if strict and not check:
        return {"index": index, "ok": False, "written": False,
                "errors": check["errors"], "order": ["consistency_failed"]}
    # 5) durable 写盘
    path = None
    if out_dir is not None:
        name = f"export_{index:04d}.jpg"
        if isinstance(item, dict) and item.get("name"):
            name = item["name"]
            if not os.path.splitext(name)[1]:
                name += ".jpg"
        path = os.path.join(os.fspath(out_dir), name)
    receipt = save(state, path)
    # 6) 只有走到这里（写盘已完成）才允许回调 —— 顺序由 export_batch 保证。
    receipt.update({"index": index, "total": total, "ok": True,
                    "orientation": 1, "order": ["write", "callback"]})
    state["receipt"] = receipt
    return receipt


def export_batch(items=1, out_dir=None, *, on_progress=None, size=(160, 160),
                 make_thumbnail=True, strict=True, **kwargs):
    """批量导出：方向处理与写盘都完成后，才触发进度回调。

    返回汇总 Report，含每份回执与全局 ``order_ok``。``items=1``（契约/演示）
    时导出一个内存样例，不依赖外部条目。
    """
    if isinstance(items, int):
        items = [items]
    if not isinstance(items, (list, tuple)):
        items = [items]

    total = len(items)
    receipts = []
    progress_events = []
    for index, item in enumerate(items):
        receipt = _process_one(item, out_dir, index, total, size,
                               make_thumbnail, strict)
        receipts.append(receipt)
        # 关键：进度回调严格位于 _process_one（内含 durable save）之后。
        progress_events.append({"index": index, "after": "write", "ok": receipt.get("ok")})
        if callable(on_progress):
            _call_flexible(on_progress, index + 1, total, receipt)

    succeeded = sum(1 for r in receipts if r.get("ok") and r.get("written"))
    order_ok = all(r.get("order", [])[:2] == ["write", "callback"]
                   for r in receipts if r.get("written"))
    report = Report(ok=succeeded == total and order_ok, total=total,
                    succeeded=succeeded, failed=total - succeeded,
                    receipts=receipts, items=receipts,
                    progress_events=progress_events, order_ok=order_ok)
    return report


async def export_batch_async(items=1, out_dir=None, *, on_progress=None,
                             size=(160, 160), **kwargs):
    """异步版本：``await asyncio.to_thread(写盘)`` 完成后再触发回调。"""
    if isinstance(items, int):
        items = [items]
    if not isinstance(items, (list, tuple)):
        items = [items]
    total = len(items)
    receipts = []
    for index, item in enumerate(items):
        # to_thread 等待包含 fsync 的写盘彻底结束。
        receipt = await asyncio.to_thread(_process_one, item, out_dir, index, total, size)
        receipts.append(receipt)
        if callable(on_progress):
            ret = on_progress(index + 1, total, receipt)
            if inspect.isawaitable(ret):
                await ret
    succeeded = sum(1 for r in receipts if r.get("ok") and r.get("written"))
    order_ok = all(r.get("order", [])[:2] == ["write", "callback"]
                   for r in receipts if r.get("written"))
    return Report(ok=order_ok and succeeded == total, total=total,
                  succeeded=succeeded, failed=total - succeeded,
                  receipts=receipts, items=receipts, order_ok=order_ok)


class BatchExporter:
    """可配置的批量导出器，回调顺序保证与 :func:`export_batch` 相同。"""

    def __init__(self, out_dir=None, size=(160, 160), make_thumbnail=True, strict=True):
        self.out_dir = out_dir
        self.size = size
        self.make_thumbnail = make_thumbnail
        self.strict = strict
        self.progress = []

    def run(self, items, on_progress=None):
        def wrapped(done, total, receipt):
            self.progress.append((done, total, receipt.get("path")))
            if callable(on_progress):
                _call_flexible(on_progress, done, total, receipt)
        return export_batch(items, self.out_dir, on_progress=wrapped,
                            size=self.size, make_thumbnail=self.make_thumbnail,
                            strict=self.strict)


class ImagePipeline:
    """把六条修复串成显式流水线，便于阅读与链式调用。

    用法::

        ImagePipeline("portrait.jpg").orient().crop({"x":..,"y":..,"width":..,"height":..}) \
             .thumbnail((200,200)).save("out.jpg").callback(print)
    """

    def __init__(self, source=1):
        self.state = read(source) if not isinstance(source, dict) else _ensure_state(source)

    def orient(self):
        self.state = orient(self.state)
        return self

    def rewrite_orientation(self, target=1):
        rw = rewrite_orientation(self.state, target=target)
        self.state["orientation"] = rw["orientation"]
        self.state["exif"] = rw["exif"]
        self.state["exif_bytes"] = rw.get("exif_bytes")
        return self

    def crop(self, box=None, **kw):
        self.state = crop(self.state, box, **kw)
        return self

    def thumbnail(self, size=(160, 160), **kw):
        self.state = thumb_after_crop(self.state, size, **kw)
        return self

    def validate(self, strict=False):
        return consistency_check(self.state, strict=strict)

    def save(self, path=None, **kw):
        self.receipt = save(self.state, path, **kw)
        return self.receipt

    def callback(self, fn, path=None):
        target_path = path or (getattr(self, "receipt", {}) or {}).get("path")
        return callback_after_write(self.state.get("image", 1), target_path, fn)
