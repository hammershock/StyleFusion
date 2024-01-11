import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class COCODataset(Dataset):
    def __init__(self, root, transform=None):
        """
        初始化 COCO 数据集
        :param root: 存储数据集的路径
        :param train: 是否使用训练集
        :param transform: 应用于图像的转换
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.dataset = datasets.ImageFolder(root, transform=transform)
        # self.dataset = datasets.CocoDetection(root=os.path.join(root, 'coco'),
        #                                       annFile=os.path.join(root,
        #                                                            'annotations/instances_train2017.json' if train else 'annotations/instances_val2017.json'),
        #                                       transform=self.transform)

    def __getitem__(self, index):
        """
        获取一个数据点和其标签
        """
        image, _ = self.dataset[index]
        return image

    def __len__(self):
        """
        返回数据集中的数据点数
        """
        return len(self.dataset)
