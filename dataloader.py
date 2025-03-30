import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os

class MaskGenerator:
    def __init__(self, target_bgr_arr):
        self.target_bgr_arr = target_bgr_arr
    
    def generate_mask(self, image, target_bgr):
        lower_bound = np.array(target_bgr)
        upper_bound = np.array(target_bgr)
        mask = cv2.inRange(image, lower_bound, upper_bound)
        return mask
    
    def split_to_classes(self, image):
        class_masks = []
        for bgr in self.target_bgr_arr:
            mask = self.generate_mask(image, bgr)
            class_masks.append(mask)
        return np.stack(class_masks, axis=0)

class SegmentationDataset(Dataset):
    def __init__(self,images_dir,masks_dir, rgb_to_class = {
            (155, 38, 182): 0,    # obstacles
            (14, 135, 204): 1,    # water
            (124, 252, 0): 2,     # soft-surfaces
            (255, 20, 147): 3,    # moving-objects
            (169, 169, 169): 4    # landing-zones
        }
, img_size = (512,512), transform = None,augmentations = None):
        self.img_size = img_size
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augmentations = augmentations

        self.rgb_to_class = {
            (155, 38, 182): 0,    # obstacles
            (14, 135, 204): 1,    # water
            (124, 252, 0): 2,     # soft-surfaces
            (255, 20, 147): 3,    # moving-objects
            (169, 169, 169): 4    # landing-zones
        }

    def __len__(self):
        return len(self.images_dir)

    def rgb_to_mask(self, rgb_mask):
        height, width, _ = rgb_mask.shape
        mask = np.zeros((height, width), dtype=np.int64)

        for rgb, class_idx in self.rgb_to_class.items():
            lower_bound = np.array(rgb, dtype=np.uint8)
            upper_bound = np.array(rgb, dtype=np.uint8)
            color_mask = cv2.inRange(rgb_mask, lower_bound, upper_bound)
            mask[color_mask > 0] = class_idx
    
        return mask

    def __getitem__(self, idx):
        image = cv2.imread(self.images_dir[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        mask = cv2.imread(self.masks_dir[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        mask = self.rgb_to_mask(mask)
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        image = image / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        
        return image, mask