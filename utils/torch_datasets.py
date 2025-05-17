# Utility file containing Dataset and DataModule classes for PV segmentation
import os
import warnings
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2 as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
import cv2

# Suppress rasterio warnings for non-georeferenced TIFFs
warnings.filterwarnings("ignore", category=UserWarning, module="rasterio")
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

class PVSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, mask_transform=None,
                 target_size=(832, 832), in_channels=3):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform
        self.target_size = target_size  # Expects a tuple (height, width)
        self.in_channels = in_channels
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(f"Mismatch between number of images ({len(self.image_paths)}) and masks ({len(self.mask_paths)})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_paths[idx], self.mask_paths[idx]
        try:
            if str(img_path).lower().endswith(('.tif', '.tiff')):
                with rasterio.open(img_path) as src:
                    num_bands_to_read = min(self.in_channels, src.count)
                    image_data = src.read(list(range(1, num_bands_to_read + 1)))
                    if image_data.shape[0] < self.in_channels:
                        padding = np.zeros((self.in_channels - image_data.shape[0], src.height, src.width), dtype=image_data.dtype)
                        image_data = np.concatenate((image_data, padding), axis=0)
                    image = np.moveaxis(image_data, 0, -1)  # HWC
                    # Scale to 0-255 if not already, common for satellite imagery
                    if image.max() > 255.0 or image.dtype != np.uint8:
                        if image.max() > 1.0:  # Heuristic for scaling larger integer types
                             image = (image / image.max() * 255).astype(np.uint8)
                        else:  # Assume float 0-1
                             image = (image * 255).astype(np.uint8)
            else:  # PNG, JPG
                image = Image.open(img_path)
                # Ensure correct number of channels
                if self.in_channels == 3 and image.mode != 'RGB': image = image.convert('RGB')
                elif self.in_channels == 1 and image.mode != 'L': image = image.convert('L')
                elif self.in_channels == 4 and image.mode != 'RGBA': image = image.convert('RGBA')

            image_np = np.asarray(image)
            if image_np.shape != (self.target_size[1], self.target_size[0]):  # PIL size is (width, height)
                image_np = cv2.resize(image_np, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)

            # Ensure correct number of channels if model expects 3 from grayscale
            if image_np.ndim == 2 and self.in_channels == 3: # Grayscale to RGB
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif image_np.ndim == 3 and image_np.shape[2] == 1 and self.in_channels == 3: # H,W,1 to H,W,3
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            # If image_np has more channels than self.in_channels (e.g. RGBA but model wants RGB)
            elif image_np.ndim == 3 and image_np.shape[2] > self.in_channels and self.in_channels == 3:
                image_np = image_np[:, :, :3] # Take first 3 channels (e.g. RGB from RGBA)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Using zeros.")
            zero_img_data = np.zeros((self.target_size[0], self.target_size[1], self.in_channels), dtype=np.uint8)
            image_pil = Image.fromarray(zero_img_data)
        # end load image

        try:
            mask_pil = Image.open(mask_path).convert('L')
            if mask_pil.size != (self.target_size[1], self.target_size[0]):
                mask_pil = mask_pil.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
            mask_np = np.array(mask_pil)
            # Ensure mask is 2D (H, W) for Albumentations if it's grayscale.
            # if mask_np.ndim == 2:
            #     mask_np = np.expand_dims(mask_np, axis=-1)  # Add channel dimension
            mask_np = (mask_np > 127).astype(np.float32)  # Threshold common for masks
            # mask_pil = Image.fromarray(mask_np, mode='F')
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}. Using zeros.")
            mask_pil = Image.fromarray(np.zeros(self.target_size, dtype=np.float32), mode='F')

        # image_tensor = self.transform(image_pil) if self.transform else transforms.ToTensor()(image_pil)
        # mask_tensor = self.mask_transform(mask_pil) if self.mask_transform else transforms.ToTensor()(mask_pil)
        augmented = self.transform(image=image_np, mask=mask_np)
        image_tensor = augmented['image']
        mask_tensor = augmented['mask']
        # ensure mask has shape (1, H, W) for PyTorch
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        return image_tensor, mask_tensor

class PVSegmentationDataModule(pl.LightningDataModule):
    def __init__(self, image_dir: str, mask_dir: str, batch_size: int, num_workers: int,
                 val_split_ratio: float, seed: int = 42,
                 patch_size_pixels_dm: int = 256, in_channels_dm: int = 3):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_ratio = val_split_ratio
        self.seed = seed
        self.patch_size_tuple = (patch_size_pixels_dm, patch_size_pixels_dm)
        self.in_channels = in_channels_dm

        self.imagenet_mean = [0.485, 0.456, 0.406][:self.in_channels]
        self.imagenet_std = [0.229, 0.224, 0.225][:self.in_channels]
        if self.in_channels == 1:
            self.imagenet_mean, self.imagenet_std = [0.449], [0.226]

        # self.train_transform = Compose([
        #     RandomChoice([
        #         RandomHorizontalFlip(p=0.5),
        #         RandomVerticalFlip(p=0.5),
        #         ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #         RandomRotation(degrees=180),
        #         RandomResizedCrop(size=(PATCH_SIZE_PIXELS, PATCH_SIZE_PIXELS), scale=(0.8, 1.0)),
        #         RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        #     ]), # Apply one or more of these transforms with 80% probability
        #     ToTensor(), Normalize(mean=self.imagenet_mean, std=self.imagenet_std)])
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.SomeOf([ # Apply a random number of these transforms
                A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.75, 1.5), saturation=(0.7, 1.1), hue=(-0.25, 0.25)),
                # see here: https://web.archive.org/web/20190424180810/http://tanbakuchi.com/posts/comparison-of-openv-interpolation-algorithms/
                # and here: https://iq.opengenus.org/different-interpolation-methods-in-opencv/
                A.SafeRotate(limit=(-180, 180), interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, mask_interpolation=cv2.INTER_LINEAR, rotate_method='ellipse'),
                A.RandomResizedCrop(size=self.patch_size_tuple, scale=(0.7, 1)), A.CLAHE(clip_limit=(1,4)), A.GaussianBlur(blur_limit=5),
                A.RandomSunFlare(flare_roi=(0.1, 0.1, 0.9, 0.9), src_radius=60, num_flare_circles_range=(2, 5), method='physics_based')
                # A.Affine(translate_percent=(0.1, 0.2), scale=(0.8, 1.2), shear=10),
            ], n=3, p=0.9), # Apply 3 random transforms from the list with 90% prob (* prob of each transform; default is 0.5)
            A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std),
            ToTensorV2(),
        ])
        # self.val_transform = transforms.Compose([
        #     transforms.ToTensor(), transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)])
        self.val_transform = A.Compose([
            A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std),
            ToTensorV2(),
        ])
        self.mask_transform = A.Compose([ToTensorV2()])

    def prepare_data(self):
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask dir not found: {self.mask_dir}")
        print(f"Data check: Image dir '{self.image_dir}', Mask dir '{self.mask_dir}' exist.")

    def setup(self, stage: str = None):
        img_exts = ['*.tif', '*.png', '*.jpg', '*.jpeg']
        all_img_paths = []
        # print(f"Searching for images in {self.image_dir}")
        for ext in img_exts:
            all_img_paths.extend(list(self.image_dir.glob(ext)))
        all_img_paths = sorted([p for p in all_img_paths if p.is_file()])

        valid_img_p, all_mask_p, skipped = [], [], 0
        for img_p in all_img_paths:
            found_m = next((self.mask_dir / (img_p.stem + m_ext) for m_ext in ['.png', '.tif', '.jpg', '.jpeg']
                          if (self.mask_dir / (img_p.stem + m_ext)).exists()), None)
            if found_m:
                valid_img_p.append(img_p)
                all_mask_p.append(found_m)
            else:
                skipped += 1

        if skipped > 0:
            print(f"Warning: Skipped {skipped} images due to missing masks during DataModule setup.")
        if not valid_img_p:
            raise ValueError("No valid image/mask pairs found in DataModule setup.")

        ds_size = len(valid_img_p)
        val_s = int(ds_size * self.val_split_ratio)
        train_s = ds_size - val_s

        if train_s <= 0:
            raise ValueError(f"Not enough data for training. Train size: {train_s}")

        indices = list(range(ds_size))
        if val_s > 0:
            train_idx, val_idx = random_split(indices, [train_s, val_s], generator=torch.Generator().manual_seed(self.seed))
        else:
            print("Warning: Validation split resulted in 0 validation samples. Using all data for training. No validation will be performed.")
            train_idx = indices
            val_idx = []

        train_img_list = [valid_img_p[i] for i in train_idx]
        train_mask_list = [all_mask_p[i] for i in train_idx]
        val_img_list = [valid_img_p[i] for i in val_idx]
        val_mask_list = [all_mask_p[i] for i in val_idx]

        self.train_dataset = PVSegmentationDataset(
            train_img_list, train_mask_list,
            self.train_transform, self.mask_transform,
            self.patch_size_tuple, self.in_channels
        )

        self.val_dataset = PVSegmentationDataset(
            val_img_list, val_mask_list,
            self.val_transform, self.mask_transform,
            self.patch_size_tuple, self.in_channels
        ) if val_img_list else None

        print(f"Setup: Train: {len(self.train_dataset)}, Val: {len(self.val_dataset) if self.val_dataset else 'None'}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # Always set pin_memory to True for faster data transfer
            persistent_workers=(self.num_workers > 0)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0)
        ) if self.val_dataset else None

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0)
        ) if self.val_dataset else None