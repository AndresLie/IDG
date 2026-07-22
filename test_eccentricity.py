from PIL import Image
import cv2
import numpy as np

def check(path):
    print("Checking", path)
    mask = Image.open(path).convert("L")
    arr = np.asarray(mask)
    coords = cv2.findNonZero(arr)
    if coords is not None and len(coords) >= 5:
        rect = cv2.minAreaRect(coords)
        width, height = rect[1]
        angle = rect[2]
        
        print(f"Original WxH: {width}x{height}, angle={angle}")
        if width < height:
            width, height = height, width
            angle += 90
            
        eccentricity = width / max(height, 1)
        print(f"Eccentricity: {eccentricity}")

check("outputs/auto_mask_smoke/auto_masks/qwen/masks/custom_part/scratch/custom_part_scratch_000_refined.png")
check("outputs/auto_mask_smoke/auto_masks/qwen/masks/custom_part/scratch/custom_part_scratch_001_refined.png")
check("outputs/auto_mask_smoke/auto_masks/qwen/masks/custom_part/scratch/custom_part_scratch_002_refined.png")
