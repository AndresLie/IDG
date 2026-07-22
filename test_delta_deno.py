import torch
from PIL import Image
from iadgen_v2.auto_masks import write_delta_deno_bbox_masks
from pathlib import Path

img = Image.new("RGB", (1024, 1024), (128, 128, 128))
try:
    write_delta_deno_bbox_masks(
        img, "custom_part", "scratch", (100, 100, 200, 200), Path("/tmp"), "test",
        auto={"sd_device": "cuda" if torch.cuda.is_available() else "cpu"},
        text_hint="a scratch", padding=5
    )
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
