from typing import List, Tuple
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_img_list(root: Path):
    if root.is_dir():
        files = list(sorted(root.glob('*.png'))) \
                + list(sorted(root.glob('*.jpg'))) \
                + list(sorted(root.glob('*.jpeg')))
    else:
        files = [root]

    for f in files:
        yield f

def load_img(img_path: Path, img_size:Tuple[int, int]=(1024, 1024)) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    img = tf(Image.open(img_path).convert('RGB')).unsqueeze(0)
    return img

def freq_decompose(latent, sigma, kernel_size):
    _, c, h, w = latent.shape

    factor = h//64
    k = kernel_size * factor + ((factor + 1) % 2)
    sigma = sigma * factor

    # decompose
    lp_latent = TF.gaussian_blur(latent, k, sigma)
    hp_latent = latent - lp_latent

    return lp_latent, hp_latent


def precompute_text_embedding(model, prompts: List[torch.Tensor], device: torch.device):
    for att in dir(model):
        if att.startswith('text_enc'):
            getattr(model, att).to(device)

    outputs = []
    with torch.no_grad():
        for prompt in prompts:
            outputs.append(model.encode_prompt(prompt, batch_size=1))

    for att in dir(model):
        if att.startswith('text_enc'):
            delattr(model, att)

    torch.cuda.empty_cache()

    return outputs