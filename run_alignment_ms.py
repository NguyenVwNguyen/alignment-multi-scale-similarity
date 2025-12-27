import os
import torch
import torch.nn.functional as F
from PIL import Image

from clip_encoder import CLIPEncoder
from prompt_ensemble import build_prompts
from multiscale_features import extract_multiscale_features


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET = r"D:\File Downloads\BIG_DATA_LYTHUYET\VisA_20220922"
MAX_IMAGES = 1   # demo đúng 1 ảnh


encoder = CLIPEncoder(DEVICE)
clip_model = encoder.model

print("=== ALIGNMENT & MULTI-SCALE SIMILARITY DEMO ===")

img_count = 0  # đếm số ảnh đã chạy

for obj in os.listdir(DATASET):
    img_root = os.path.join(DATASET, obj, "Data", "Images")
    if not os.path.isdir(img_root):
        continue

    print(f"\nObject: {obj}")

   
    prompts = build_prompts(obj)
    text_feats = encoder.encode_text(prompts)     # [K, D]
    text_feats = F.normalize(text_feats, dim=-1)

    for defect in os.listdir(img_root):
        defect_dir = os.path.join(img_root, defect)
        if not os.path.isdir(defect_dir):
            continue

        for img_name in os.listdir(defect_dir):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            
            img_path = os.path.join(defect_dir, img_name)
            image = encoder.preprocess_image(
                Image.open(img_path).convert("RGB")
            )

            
            patch_maps = extract_multiscale_features(image, clip_model)

            for scale_id, patch_map in enumerate(patch_maps):
                # patch_map: [1, C, H, W]
                B, C, H, W = patch_map.shape

                # flatten patches
                patches = patch_map.view(B, C, H * W).permute(0, 2, 1)  # [1, HW, C]
                patches = F.normalize(patches, dim=-1)

                # cosine similarity (patch-level)
                sim = patches @ text_feats.T    # [1, HW, K]
                sim_map = sim.view(H, W, -1)    # [H, W, K]

                print(
                    f"{obj} | {defect} | {img_name} | "
                    f"scale-{scale_id} similarity map {sim_map.shape}"
                )

            img_count += 1
            if img_count >= MAX_IMAGES:
                break

        if img_count >= MAX_IMAGES:
            break

    if img_count >= MAX_IMAGES:
        break

print("\n=== DONE ===")

