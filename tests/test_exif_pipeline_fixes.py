"""EXIF 方向 / 像素空间裁切 / 写盘回调一致性的回归测试。

这些测试对应题目描述的六个事故：
旋转后标签未改写、竖拍裁切空间不一致、缩略图不跟随裁切、
回调早于写盘、中间副本污染原始状态、导出前缺少一致性校验。
"""

import io
import os

import pytest

PIL = pytest.importorskip("PIL")  # 真实像素/EXIF 测试需要 Pillow
from PIL import Image  # noqa: E402

import imaging.exif_orient_crop as exif  # noqa: E402


W, H = 96, 64  # 传感器方向样例尺寸（横版）；6/8 方向显示为 64x96。


def _jpeg_with_orientation(o, color=(30, 90, 160)):
    img = Image.new("RGB", (W, H), color)
    exif_bytes = exif._exif_to_bytes({exif.ORIENTATION_TAG: o}, orientation=o)
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif_bytes)
    buf.seek(0)
    return buf


# -- 修复点 1：旋转像素后改写标签 -------------------------------------------

def test_rewrite_orientation_resets_tag_to_one():
    assert exif.rewrite_orientation(6)["orientation"] == 1
    out = exif.rewrite_orientation({274: 8, 306: "2024:01:01"})
    assert out["exif"][274] == 1
    assert out["exif"][306] == "2024:01:01"  # 其它 EXIF 标签不丢


def test_orient_rotates_pixels_and_rewrites_tag(tmp_path):
    path = tmp_path / "p6.jpg"
    path.write_bytes(_jpeg_with_orientation(6).getvalue())

    state = exif.orient(exif.read(path))
    assert (state["width"], state["height"]) == (64, 96)  # 轴交换
    assert state["orientation"] == 1
    assert state["exif"][274] == 1

    out = tmp_path / "out.jpg"
    exif.save(state, out)
    reloaded = Image.open(out)
    assert reloaded.getexif().get(274, 1) == 1
    assert reloaded.size == (64, 96)


def test_orient_is_idempotent_and_avoids_double_rotation():
    once = exif.orient(exif.read(6))["image"]
    twice = exif.orient(exif.orient(exif.read(6)))["image"]
    assert once.size == twice.size == (64, 96)
    assert once.tobytes() == twice.tobytes()


# -- 修复点 2：裁切坐标统一到像素空间 ---------------------------------------

def test_portrait_display_box_maps_to_sensor_box():
    # 方向 6：显示左上 20x20 -> 传感器坐标 (0, H-20, 20, H) = (0,44,20,64)
    rect = exif.crop_in_pixel_space((0, 0, 20, 20), 6, W, H)
    assert rect.box == (0, 44, 20, 64)
    assert rect.xywh == (0, 44, 20, 20)


def test_xywh_and_dict_forms_agree():
    a = exif.crop_in_pixel_space((10, 10, 20, 30), 6, W, H, rect_format="xywh")
    b = exif.crop_in_pixel_space({"x": 10, "y": 10, "width": 20, "height": 30}, 6, W, H)
    assert a.box == b.box == (10, 34, 40, 54)


@pytest.mark.parametrize("o", range(1, 9))
def test_crop_mapping_matches_display_space_crop(o):
    """映射->传感器裁切->定向 必须与 先定向->显示空间裁切 逐像素一致。"""
    import random

    ops = exif._PIL_OPS

    def orient_full(img):
        return img if o == 1 else img.transpose(ops[o])

    dw, dh = exif.display_dimensions(W, H, o)
    base = Image.new("RGB", (W, H))
    base.putdata([(x * 4 % 256, y * 8 % 256, (x + y) % 256)
                  for y in range(H) for x in range(W)])
    rng = random.Random(o)
    for _ in range(40):
        x0 = rng.randint(0, dw - 2)
        y0 = rng.randint(0, dh - 2)
        x1 = rng.randint(x0 + 1, dw)
        y1 = rng.randint(y0 + 1, dh)
        box = exif.crop_in_pixel_space((x0, y0, x1, y1), o, W, H).box
        expected = orient_full(base).crop((x0, y0, x1, y1))
        actual = orient_full(base.crop(box))
        assert actual.size == expected.size
        assert actual.tobytes() == expected.tobytes()


def test_portrait_crop_end_to_end_hits_right_region():
    state = exif.crop(exif.read(6), (0, 0, 20, 20))  # 显示坐标框
    actual = exif.orient(state)["image"]
    expected = exif.orient(exif.read(6))["image"].crop((0, 0, 20, 20))
    assert actual.size == expected.size == (20, 20)
    assert actual.tobytes() == expected.tobytes()


# -- 修复点 3：缩略图按裁切结果重生成 ---------------------------------------

def test_thumbnail_follows_cropped_composition():
    state = exif.orient(exif.read(6))                 # 64x96 显示缓冲
    state = exif.crop(state, (0, 33, 64, 63), space="pixel")  # 64x30 宽条
    state = exif.thumb_after_crop(state, (160, 160))
    thumb = state["thumbnail"]
    assert thumb["source"] == "cropped"
    cw, ch = state["cropped_image"].size
    assert abs(thumb["width"] / thumb["height"] - cw / ch) < 0.03
    assert abs(thumb["width"] / thumb["height"] - 64 / 96) > 0.3


def test_thumbnail_from_direct_image_keeps_aspect():
    thumb = exif.thumb_after_crop(Image.new("RGB", (300, 100)), (120, 120))
    assert (thumb["width"], thumb["height"]) == (120, 40)


# -- 修复点 4：写盘后再回调 --------------------------------------------------

def test_callback_runs_strictly_after_writer():
    events = []
    report = exif.callback_after_write(
        lambda: events.append("write"),
        lambda receipt=None: events.append("callback"),
    )
    assert events == ["write", "callback"]
    assert report["order"] == ["write", "callback"]


def test_file_is_durable_before_callback(tmp_path):
    path = tmp_path / "cb.jpg"
    seen = {}

    def on_done(receipt=None):
        seen["exists"] = path.exists()
        seen["nonempty"] = path.stat().st_size > 0 if path.exists() else False

    exif.callback_after_write(exif._fixture_image(1), os.fspath(path), on_done)
    assert seen == {"exists": True, "nonempty": True}


def test_contract_callback_one_has_write_then_callback_evidence():
    report = exif.callback_after_write(1)
    assert report is not None
    assert report["order"][:2] == ["write", "callback"]
    assert report["exists_during_callback"] is True


# -- 修复点 5：中间副本保留原始状态 ------------------------------------------

def test_intermediate_copies_preserve_original_state():
    kept = exif.keep_original_state(6)
    digest = kept["original"]["pixel_sha256"]
    assert kept["original"]["orientation"] == 6

    mid = exif.orient(kept["intermediate"])
    mid = exif.crop(mid, (2, 2, 40, 40))
    assert exif._pixel_hash(kept["original"]["image"]) == digest


def test_editing_session_shares_one_frozen_original():
    session = exif.EditingSession(6)
    session.orient().crop((0, 0, 30, 30)).thumbnail()
    assert session.original["orientation"] == 6
    assert session.copies_preserve_original() is True


# -- 修复点 6：导出前一致性校验 ---------------------------------------------

def test_consistency_detects_rotated_pixels_with_stale_tag():
    bad = exif.read(6)
    bad["pixels_oriented"] = True  # 像素已转，标签却仍是 6
    report = exif.consistency_check(bad)
    assert not report
    assert any(c["code"] == "tag_pixel_sync" and not c["ok"] for c in report["checks"])
    with pytest.raises(exif.ConsistencyError):
        exif.consistency_check(bad, strict=True)


def test_consistency_accepts_correct_pipeline():
    state = exif.read(6)
    state = exif.orient(state)
    state = exif.crop(state, (0, 0, 20, 20))
    state = exif.thumb_after_crop(state)
    assert exif.consistency_check(state)


def test_consistency_rejects_thumbnail_from_original():
    state = exif.crop(exif.orient(exif.read(1)), (0, 0, 50, 20))
    state["thumbnail"] = {"source": "original", "follows_crop": False, "aspect_delta": 0.9}
    assert not exif.consistency_check(state)


# -- 批量导出端到端 ----------------------------------------------------------

def test_export_batch_writes_tag_one_and_callbacks_after_write(tmp_path):
    out = tmp_path / "export"
    progress = []
    report = exif.export_batch(
        [exif.read(o) for o in (1, 3, 6, 8)],
        os.fspath(out),
        on_progress=lambda i, total, receipt: progress.append((i, receipt["path"])),
    )
    assert report["succeeded"] == 4 and report["failed"] == 0
    assert report["order_ok"] is True
    assert len(progress) == 4
    for name in os.listdir(out):
        tag = Image.open(out / name).getexif().get(274, 1)
        assert tag == 1


# -- 其余入口的基本正确性 ----------------------------------------------------

def test_validate_layout_bounds():
    assert exif.validate_layout((0, 0, 20, 20), 6, W, H)
    assert not exif.validate_layout((0, 0, 500, 500), 1, W, H)


def test_determinism_is_stable():
    report = exif.determinism(6, runs=3)
    assert report["deterministic"] is True
    assert len(set(report["digests"])) == 1


def test_fallback_normalizes_orientation():
    report = exif.fallback(1)
    assert report["fallback_used"] is True
    assert report["orientation"] == 1
