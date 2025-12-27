import torch.nn.functional as F
import torch

def compute_similarity_map(patch_map, text_features):
    """
    patch_map: [B, H, W, D]
    text_features: [K, D]
    return: [B, K, H, W]
    """
    patch_map = F.normalize(patch_map, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    sim_map = torch.einsum("bhwd,kd->bkhw", patch_map, text_features)
    return sim_map
