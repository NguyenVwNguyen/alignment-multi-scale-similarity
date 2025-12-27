import torch
import open_clip

class CLIPEncoder:
    def __init__(self, device):
        self.device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained="openai",
            img_size=224
        )
        self.model = self.model.to(device).eval()

        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")

    def encode_text(self, prompts):
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            text_feat = self.model.encode_text(tokens)
        return text_feat

    def preprocess_image(self, pil_image):
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        return image
