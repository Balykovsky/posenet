import io

import torch
import numpy as np

from PIL import Image
from albumentations import RandomCrop
from torchvision.models import inception_v3

from models import GoogleNet


def get_model():
    weights_path = 'weights/weights.pth'
    model = GoogleNet(inception_v3(pretrained=True))
    if not torch.cuda.is_available():
        pretrained_dict = torch.load(weights_path, map_location='cpu')
    else:
        pretrained_dict = torch.load(weights_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = torch.nn.DataParallel(model)
    model.eval()
    return model


def get_tensor(image):
    with Image.open(io.BytesIO(image)) as img:
        img = np.array(img)[..., :3]
    rc = RandomCrop(299, 299)
    batch = np.array([rc(image=img)['image'] for i in range(10)])
    batch = torch.from_numpy(np.moveaxis(batch.astype(np.float32) / 255., -1, 1))
    return batch
