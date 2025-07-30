import open_clip
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "model/model_openclip.bin"
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained=model_path
)
model = model.to(device).eval()



def load_image(image_path_or_pil):
    if isinstance(image_path_or_pil, str):
        image = Image.open(image_path_or_pil).convert("RGB")
    else:
        image = image_path_or_pil.convert("RGB")
    return preprocess(image).unsqueeze(0).to(device)


def find_similarity(image1_tensor, image2_tensor):
    with torch.no_grad():
        f1 = model.encode_image(image1_tensor)
        f2 = model.encode_image(image2_tensor)

    f1 = f1 / f1.norm(dim=-1, keepdim=True)
    f2 = f2 / f2.norm(dim=-1, keepdim=True)
    return F.cosine_similarity(f1, f2).item()


