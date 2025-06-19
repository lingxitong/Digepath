from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
from .common_utils import load_class2id_mapping

class ROIDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, csv_path,domain,transform,class2id_txt):
        self.csv_path = csv_path
        self.domain = domain
        self.df = pd.read_csv(csv_path)
        self.images_path = self.df[f'{domain}_path'].dropna().tolist()
        self.images_class = self.df[f'{domain}_label'].dropna().tolist()
        self.transform = transform
        self.class2id_dict = load_class2id_mapping(class2id_txt)
        self.id2class_dict = {v: k for k, v in self.class2id_dict.items()}

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, int(label)
    
    def get_imgs_from_idxs(self, idxs):
        img_paths = [self.images_path[idx] for idx in idxs]
        labels = [self.images_class[idx] for idx in idxs]
        return img_paths, labels

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
    
class FeatDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, all_feats,all_labels):
        self.all_feats = all_feats
        self.all_labels = all_labels

    def __len__(self):
        return len(self.all_feats)

    def __getitem__(self, item):
        feat = self.all_feats[item]
        label = self.all_labels[item]
        return feat, int(label)
    
    # def get_imgs_from_idxs(self, idxs):
    #     img_paths = [self.images_path[idx] for idx in idxs]
    #     labels = [self.images_class[idx] for idx in idxs]
    #     return img_paths, labels

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
