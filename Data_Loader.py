import torch
import numpy as np
import random
import PIL.Image as Image
from skimage import io
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast
from torchvision import transforms
from torch import Tensor
from functools import partial
from operator import itemgetter
from utils.losses import one_hot2dist, class2one_hot
import os

palette = [[0], [1], [2]]
num_classes = 3
D = Union[Image.Image, np.ndarray, Tensor]


def gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
    return transforms.Compose([
        lambda img: np.array(img)[...],
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
        partial(class2one_hot, K=K),
        itemgetter(0)  # Then pop the element to go back to img shape
    ])


def dist_map_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
    return transforms.Compose([
        gt_transform(resolution, K),
        lambda t: t.cpu().numpy(),
        partial(one_hot2dist, resolution=resolution),
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])


def transform(img):
    img = (img - np.amin(img)) * 1.0 / (np.amax(img) - np.amin(img))  # img*1.0 transform array to double
    img = img * 1.0 / np.median(img)
    img_h = img.shape[0]
    img_w = img.shape[1]
    return np.reshape(img, (1, img_h, img_w))


class Dataset_folder(torch.utils.data.Dataset):
    def __init__(self, dic,  labels):
        self.dic = dic
        self.labels = labels.astype(int)
        # self.labels = sorted(os.listdir(labels_dir))
        self.disttransform = dist_map_transform([1, 1], 3)

    def __len__(self):
        return len(self.dic)

    def __getitem__(self, i):
        dic = self.dic[i]
        label = self.labels[i]
        # label = np.array(io.imread(self.labels_dir + self.labels[i]))

        seed = np.random.randint(0, 2 ** 32)  # make a seed with numpy generator
        torch.manual_seed(seed)

        dic_trans = transform(dic).astype(np.float32)

        label_h = label.shape[0]
        label_w = label.shape[1]
        label_map = np.zeros([3, label_h, label_w])
        for r in range(label_h):
            for c in range(label_w):
                label_map[label[r][c], r, c] = 1

        # #get mask 1 is inside
        # mask = np.zeros([img_h, img_w])    
        # for r in range(label_h):
        #     for c in range(label_w):
        #         if label[r][c] == 1:
        #             mask[r, c] = 1
        #         else:
        #             mask[r, c] = 0

        # #get boundary (2)
        # boundary = np.zeros([img_h, img_w])    
        # for r in range(label_h):
        #     for c in range(label_w):
        #         if label[r][c] == 2:
        #             boundary[r, c] = 1
        #         else:
        #             boundary[r, c] = 0

        # #get boundary map
        # boundary_map = np.zeros([2, img_h, img_w])    
        # for r in range(label_h):
        #     for c in range(label_w):
        #         if label[r][c] == 2:
        #             boundary_map[1, r, c] = 1
        #         else:
        #             boundary_map[0, r, c] = 1

        # apply this seed to target/label tranfsorms  
        label = torch.tensor(label, dtype=torch.float)
        label_map = torch.tensor(label_map, dtype=torch.float)
        dist_map = self.disttransform(label)
        # boundary_map = torch.tensor(boundary_map,dtype=torch.float)
        # boundary = boundary.astype(np.float32)
        # boundary = torch.tensor(boundary,dtype=torch.float).unsqueeze(dim=0)
        # mask = mask.astype(np.float32)
        # mask = torch.tensor(mask,dtype=torch.float).unsqueeze(dim=0)

        return dic_trans,label_map, dist_map, label