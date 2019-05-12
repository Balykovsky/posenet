import torch
import cv2
import numpy as np

from torchvision.models import inception_v3
from albumentations import RandomCrop

from neural_pipeline import Predictor, FileStructManager

from models import GoogleNet


def predict_img(img: np.ndarray, predictor: Predictor, crops: int = 10):
    rc = RandomCrop(299, 299)
    batch = np.array([rc(image=img)['image'] for i in range(crops)])
    batch = torch.from_numpy(np.moveaxis(batch.astype(np.float32) / 255., -1, 1))
    output = predictor.predict({'data': batch})
    pos = np.mean(output[0].cpu().data.numpy().copy(), axis=0)
    qtn = np.mean(output[1].cpu().data.numpy().copy(), axis=0)
    print('position: {}, quaternion: {}'.format(pos, qtn))
    return pos, qtn


if __name__ == "__main__":
    img = cv2.imread('/mnt/tb_storage/uprojects/dataset/images/img_0_0_1542108891812919700.png')
    model = GoogleNet(inception_v3(pretrained=True))
    model = torch.nn.DataParallel(model)
    checkpoints_dir = 'experiments/exp_initial'
    fsm = FileStructManager(base_dir=checkpoints_dir, is_continue=True)
    predictor = Predictor(model, fsm=fsm, from_best_state=False, device=torch.device('cuda'))
    predict_img(img, predictor)
