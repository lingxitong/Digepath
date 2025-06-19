import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import math
import torch
import torchvision.transforms
import PIL.Image
import numpy as np

class Seg_Dataset(Dataset):
    def __init__(self, csv_path: str, group, transforms=None,slide_window_val = True):
        super(Seg_Dataset, self).__init__()
        self.transforms = transforms
        self.dataset_df = pd.read_csv(csv_path)
        self.img_path_list = self.dataset_df[f'{group}_img_path'].dropna().to_list()
        self.mask_path_list = self.dataset_df[f'{group}_mask_path'].dropna().to_list()
        assert len(self.img_path_list) == len(self.mask_path_list)
        self.slide_window_val = False
        if group != 'train':
            if slide_window_val == True:
                self.slide_window_val = True
    def __getitem__(self, idx):
        img = Image.open(self.img_path_list[idx]).convert('RGB')
        mask = Image.open(self.mask_path_list[idx]).convert('L')
        if self.slide_window_val:
            width, height = img.size
            target_size = max(width, height)
            target_size = math.ceil(target_size / 224) * 224
            new_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))  
            new_mask = Image.new('L', (target_size, target_size), 255)  
            offset_x = (target_size - width) // 2
            offset_y = (target_size - height) // 2
            new_img.paste(img, (offset_x, offset_y))
            new_mask.paste(mask, (offset_x, offset_y))
            img = new_img
            mask = new_mask
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        if self.slide_window_val:
            _, height, width = img.shape
            patch_size = 224

            N = height // patch_size

            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            img_patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            mask_patches = mask.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

            img_patches = img_patches.contiguous().view(-1, 3, patch_size, patch_size)
            mask_patches = mask_patches.contiguous().view(-1, patch_size, patch_size)

            img = img_patches
            mask = mask_patches


        return img, mask

    def __len__(self):
        return len(self.img_path_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets
    


class Seg_Dataset_2(Dataset):
    def __init__(self, group, transforms=None,img_path_list=None,mask_path_list=None,slide_window_val = True):
        super(Seg_Dataset_2, self).__init__()
        # self.dataset_df = pd.read_csv(csv_path)
        self.img_path_list = img_path_list
        self.mask_path_list = mask_path_list
        self.transforms = transforms
        assert len(self.img_path_list) == len(self.mask_path_list)
        self.slide_window_val = False
        if group != 'train':
            if slide_window_val == True:
                self.slide_window_val = True
    def __getitem__(self, idx):
        img = Image.open(self.img_path_list[idx]).convert('RGB')
        mask = Image.open(self.mask_path_list[idx]).convert('L')
        if self.slide_window_val:
            width, height = img.size
            target_size = max(width, height)
            target_size = math.ceil(target_size / 224) * 224
            new_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))  
            new_mask = Image.new('L', (target_size, target_size), 255)  
            offset_x = (target_size - width) // 2
            offset_y = (target_size - height) // 2
            new_img.paste(img, (offset_x, offset_y))
            new_mask.paste(mask, (offset_x, offset_y))

            img = new_img
            mask = new_mask
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        if self.slide_window_val:
            _, _, height, width = img.shape
            patch_size = 224
            
            N = height // patch_size

            img_patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            mask_patches = mask.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

            img_patches = img_patches.contiguous().view(-1, 3, patch_size, patch_size)
            mask_patches = mask_patches.contiguous().view(-1, 1, patch_size, patch_size)

            img = img_patches
            mask = mask_patches.squeeze(1)


        return img, mask

    def __len__(self):
        return len(self.img_path_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs



class PanNukeDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, group):
        super(PanNukeDataset, self).__init__()
        self.dataset_df = pd.read_csv(csv_path)
        self.img_path_list = self.dataset_df[f'{group}_img_path'].dropna().to_list()
        self.mask_path_list = self.dataset_df[f'{group}_mask_path'].dropna().to_list()
        self.img_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224,224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )    

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, item):
        try:
            img = Image.open(self.img_path_list[item]).convert('RGB')
        except Exception as e:
            print(f"[ERROR] Failed to load image: {self.img_path_list[item]}")
            raise e
        mask = Image.open(self.mask_path_list[item]).convert('L')
        mask = mask.resize((224, 224), PIL.Image.NEAREST)
        mask = (np.array(mask) / 255).astype(np.uint8)
        img = self.img_transform(img)
        mask = torch.tensor(mask).long()
        return img, mask
    
    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets