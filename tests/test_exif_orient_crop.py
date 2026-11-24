import pytest
import imaging.exif_orient_crop as mod

CASES = [
    ("rewrite_orientation", (1,)),
    ("crop_in_pixel_space", (1,)),
    ("thumb_after_crop", (1,)),
    ("callback_after_write", (1,)),
    ("keep_original_state", (1,)),
    ("consistency_check", (1,)),
    ("read", (1,)),
    ("orient", (1,)),
    ("crop", (1,)),
    ("save", (1,)),
    ("validate_layout", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
