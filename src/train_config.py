import os
import torch
import numpy as np

from neural_pipeline import TrainConfig, ValidationStage, TrainStage, DataProducer


from albumentations import Compose, RandomGamma, RandomBrightnessContrast, RGBShift, \
    RandomCrop, OneOf, IAAAdditiveGaussianNoise, GridDistortion, IAAPiecewiseAffine, \
    GaussNoise, MotionBlur, MedianBlur, Blur, OpticalDistortion

from sklearn.model_selection import train_test_split
from torchvision.models import inception_v3

from dataset import PoseNetDataset
from models import GoogleNet
from metrics import PosQtnMetricsProcessor, PoseNetLoss


base_dir = '/mnt/tb_storage/uprojects/posenet/dataset'
img_dir = os.path.join(base_dir, 'images')
csv_data = os.path.join(base_dir, 'info.csv')

BATCH_SIZE = 4
LEARN_LOSS = True
POS_WT = 0.5
QTN_WT = 0.5

augmentations = Compose([RandomCrop(299, 299),
                         RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                         RandomGamma(),
                         RGBShift(),
                         OneOf([
                            IAAAdditiveGaussianNoise(p=1.),
                            GaussNoise(p=1.)], p=0.27),
                         OneOf([
                             MotionBlur(p=1.),
                             MedianBlur(blur_limit=3, p=1.),
                             Blur(blur_limit=3, p=1.)], p=0.27),
                         OneOf([
                             OpticalDistortion(p=1.),
                             GridDistortion(p=1.),
                             IAAPiecewiseAffine(p=1.)], p=0.27)])


def augmentate_train(img: np.ndarray):
    res = augmentations(image=img)
    return torch.from_numpy(np.moveaxis(res['image'].astype(np.float32) / 255., -1, 0))


def augmentate_val(img: np.ndarray):
    res = RandomCrop(299, 299)(image=img)
    return torch.from_numpy(np.moveaxis(res['image'].astype(np.float32) / 255., -1, 0))


class PoseNetTrainConfig(TrainConfig):
    experiment_name = 'exp_ll'
    experiment_dir = os.path.join('experiments', experiment_name)

    def __init__(self):
        train_ids, val_ids = train_test_split(range(900), shuffle=True, test_size=0.2)
        train_dataset = PoseNetDataset(csv_data, img_dir, train_ids, augmentate_train)
        val_dataset = PoseNetDataset(csv_data, img_dir, val_ids, augmentate_val)

        train_data_producer = DataProducer([train_dataset], batch_size=BATCH_SIZE, num_workers=3)
        val_data_producer = DataProducer([val_dataset], batch_size=BATCH_SIZE, num_workers=3)

        model = GoogleNet(inception_v3(pretrained=True))
        model = torch.nn.DataParallel(model)

        if LEARN_LOSS:
            loss_func = PoseNetLoss(pos_wt=POS_WT, qtn_wt=QTN_WT, learn_flag=True)
            optimizer = torch.optim.Adam([{'params': model.parameters()},
                                          {'params': [loss_func.pos_wt, loss_func.qtn_wt]}], lr=1e-4)
        else:
            loss_func = PoseNetLoss(pos_wt=POS_WT, qtn_wt=QTN_WT)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        self.train_stage = TrainStage(train_data_producer, PosQtnMetricsProcessor('train'))
        self.train_stage.enable_hard_negative_mining(0.1)
        self.val_stage = ValidationStage(val_data_producer, PosQtnMetricsProcessor('validation'))

        super().__init__(model, [self.train_stage, self.val_stage], loss_func, optimizer)
