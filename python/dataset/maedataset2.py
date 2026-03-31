import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset



class MAEDataset2(Dataset):
    def __init__(
        self,
        root_dir,
        global_size=1024,
        global_scale_aug=(0.4, 1.0)
    ):
        self.root_dir = root_dir
        self.global_size = global_size
        self.global_scale_aug = global_scale_aug


        self.image_paths = []
        exts = ["*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff",
                "*.JPG", "*.JPEG", "*.BMP", "*.TIF", "*.TIFF"]

        for ext in exts:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, ext)))

        self.image_paths = list(set(self.image_paths))
        self.image_paths = sorted(self.image_paths)

        if len(self.image_paths) == 0:
            raise ValueError(f"이미지를 찾지 못했습니다: {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    

    def _apply_basic_flip(self, img):
        """상하좌우 반전을 적용합니다. 제조업 데이터는 방향성이 없는 경우가 많습니다."""
        if random.random() < 0.5:
            img = cv2.flip(img, 1) # 좌우 반전
        if random.random() < 0.5:
            img = cv2.flip(img, 0) # 상하 반전 (파티클/스크래치 형태 불변)
        return img

    # -------------------------------------

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"이미지 로드 실패: {path}")


        # [C] 리사이즈 및 텐서화
        crop = cv2.resize(img, (self.global_size, self.global_size), interpolation=cv2.INTER_CUBIC)
        crop = torch.from_numpy(crop.astype(np.float32) / 255.0).unsqueeze(0).float()

        return {
            "global_crops": crop,
            "path": path
        }