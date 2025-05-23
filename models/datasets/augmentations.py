from typing import List

import albumentations as A
from albumentations.pytorch import ToTensorV2

class Transformer:
    def __init__(self, dataset_name):
        affine = A.Affine(
            translate_px=(-16, 16),
            rotate=(-5, 5),
            scale=(0.95, 1.05),
            p=0.25,
        )

        random_brightness_contrast = A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.25,
        )

        blur = A.Blur(
            blur_limit=3,
            p=0.25,
        )

        sharpen = A.Sharpen(
            alpha=(0.1, 0.3),
            lightness=(0.5, 1.0),
            p=0.25,
        )

        flip = A.OneOf(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ],
            p=0.25,
        )

        if dataset_name == "visa":
            self.transforms = [affine, random_brightness_contrast, blur, sharpen, flip]  # change for MVTec
        elif dataset_name == "mvtec":
            self.transforms = [affine, random_brightness_contrast, blur]
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported")

    def get_transforms_with_index(self, resize, imagesize, list_index: List[int]) -> A:
        return A.Compose(
            [A.Resize(resize, resize),
            A.CenterCrop(imagesize, imagesize),
            A.OneOf(
                [self.transforms[index] for index in list_index],
                p=1.0,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),]
        )
