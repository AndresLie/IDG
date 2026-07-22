import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

def inject_defect(
    source_path: Path,
    mask_path: Path,
    target_path: Path,
    output_path: Path,
    center_x: int | None = None,
    center_y: int | None = None,
    harmonize: bool = False,
) -> None:
    """Inject a defect from a source image into a target image using Poisson Blending."""
    src = cv2.imread(str(source_path))
    if src is None:
        raise ValueError(f"Could not read source: {source_path}")
    dst = cv2.imread(str(target_path))
    if dst is None:
        raise ValueError(f"Could not read target: {target_path}")
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
        
    # Find bounding box of mask to determine center if not provided
    coords = cv2.findNonZero(mask)
    if coords is None:
        raise ValueError("Mask is empty")
        
    x, y, w, h = cv2.boundingRect(coords)
    
    # If no center provided, use the center of the defect in the source image
    if center_x is None:
        center_x = x + w // 2
    if center_y is None:
        center_y = y + h // 2
        
    # OpenCV seamless clone expects the center of the src patch in the dst image
    center = (center_x, center_y)
    
    # Ensure center is within bounds of dst
    h_dst, w_dst = dst.shape[:2]
    center_x = max(0, min(w_dst - 1, center_x))
    center_y = max(0, min(h_dst - 1, center_y))
    center = (center_x, center_y)
    
    # MIXED_CLONE preserves textures better for scratches, NORMAL_CLONE for opaque patches
    cloned = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
    
    if harmonize:
        print("Harmonizing with Latent Diffusion (ControlNet Canny)...")
        import torch
        from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
        
        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16
        )
        
        # Load SD inpainting model with ControlNet
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        
        # Generate Canny edge map from the clean target image
        target_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(target_gray, 100, 200)
        edges = np.stack([edges]*3, axis=-1)
        canny_pil = Image.fromarray(edges)
        
        cloned_pil = Image.fromarray(cv2.cvtColor(cloned, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask).convert("RGB")
        
        # Very low strength to just harmonize lighting and color without destroying defect geometry
        harmonized_pil = pipe(
            prompt="",
            image=cloned_pil,
            mask_image=mask_pil,
            control_image=canny_pil,
            strength=0.20,
            num_inference_steps=20,
        ).images[0]
        
        cloned = cv2.cvtColor(np.array(harmonized_pil), cv2.COLOR_RGB2BGR)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cloned)
    print(f"Successfully injected defect into {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Inject a defect into a target image using Poisson Blending.")
    parser.add_argument("--source", type=Path, required=True, help="Path to the source defect image")
    parser.add_argument("--mask", type=Path, required=True, help="Path to the source defect mask (inpaint_soft variant)")
    parser.add_argument("--target", type=Path, required=True, help="Path to the clean target image")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the synthetic output")
    parser.add_argument("--cx", type=int, default=None, help="Target center X (defaults to same position as source)")
    parser.add_argument("--cy", type=int, default=None, help="Target center Y (defaults to same position as source)")
    parser.add_argument("--harmonize", action="store_true", help="Run Latent Diffusion over the cloned defect to harmonize lighting")
    
    args = parser.parse_args()
    inject_defect(args.source, args.mask, args.target, args.output, args.cx, args.cy, args.harmonize)

if __name__ == "__main__":
    main()
