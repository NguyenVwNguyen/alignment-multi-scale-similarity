import torch
import math

def extract_multiscale_features(image, model, layers=(6, 12, 18, 24)):
    """
    Extract multi-scale spatial feature maps from CLIP ViT
    Return: list of tensors [B, C, H, W]
    """

    with torch.no_grad():
        # forward_intermediates là API đúng của open_clip ViT
        outputs = model.visual.forward_intermediates(
            image,
            layers=layers,
            return_prefix_tokens=False,   # bỏ CLS
            norm=False
        )

    feats = outputs["intermediates"]   # list of [B, HW, C]

    patch_maps = []

    for tokens in feats:
        # tokens: [B, HW, C]
        B, HW, C = tokens.shape
        H = W = int(math.sqrt(HW))

        tokens = tokens.transpose(1, 2)      # [B, C, HW]
        patch_map = tokens.reshape(B, C, H, W)

        patch_maps.append(patch_map)

    return patch_maps
