import io

import torch
import numpy as np

from PIL import Image
from albumentations import RandomCrop
from torchvision.models import inception_v3

try:
    from models import GoogleNet
except ImportError:
    from .models import GoogleNet


def get_model(weights_path: str, cpu: bool = False):
    model = GoogleNet(inception_v3(pretrained=True))
    if not torch.cuda.is_available() or cpu:
        model = load_weights(model, weights_path, cpu=True)
    else:
        model = load_weights(model, weights_path)
    model.eval()
    return model


def get_tensor(image: bytes):
    with Image.open(io.BytesIO(image)) as img:
        img = np.array(img)[..., :3]
    rc = RandomCrop(299, 299)
    batch = np.array([rc(image=img)['image'] for i in range(2)])
    batch = torch.from_numpy(np.moveaxis(batch.astype(np.float32) / 255., -1, 1))
    return batch


def load_weights(model: torch.nn.Module, weights_file: str = None, cpu: bool = False):
    pretrained_weights = torch.load(weights_file, map_location='cpu')
    processed = {}
    model_state_dict = model.state_dict()
    for k, v in pretrained_weights.items():
        if k.split('.')[0] == 'module' and not isinstance(model, torch.nn.DataParallel):
            k = '.'.join(k.split('.')[1:])
        elif isinstance(model, torch.nn.DataParallel) and k.split('.')[0] != 'module':
            k = 'module.' + k
        if k in model_state_dict:
            if v.device != model_state_dict[k].device:
                v.to(model_state_dict[k].device)
            processed[k] = v
    model.load_state_dict(processed)
    return model
