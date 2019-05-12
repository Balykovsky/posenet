import os
import torch
import cv2

import numpy as np
import pandas as pd


class PoseNetDataset:
    def __init__(self, path: str, img_dir: str, ids: [], aug: callable):
        self._path = path
        self._img_dir = img_dir
        self.__items_df = pd.read_csv(self._path)
        self.__ids = ids
        self.__aug = aug

    def __len__(self):
        return len(self.__ids)

    def __getitem__(self, idx):
        item = self.__items_df.iloc[self.__ids[idx]]
        target = np.array(item.get_values()[3:10], dtype=np.float32)
        img = cv2.imread(os.path.join(self._img_dir, item['ImageFile']))
        return {'data': self.__aug(img), 'target': torch.from_numpy(target)}
