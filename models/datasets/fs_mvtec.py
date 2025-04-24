import os
from enum import Enum

import PIL
import cv2
import torch
from torchvision import transforms
import random
from torch.utils.data import DataLoader
import numpy as np

_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class MVTec_Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """
    def __init__(
            self,
            source,
            classname,
            resize=256,
            imagesize=224,
            split=DatasetSplit.TRAIN,
            shot=2,
            seed=26,
            train_val_split=1.0,
            **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classname = classname
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split
        self.shot = shot
        self.seed = seed
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path, fgmask_path = self.data_to_iterate[idx]

        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if fgmask_path is not None:
            fg_mask = cv2.imread(fgmask_path, -1)
            fg_mask = PIL.Image.fromarray(np.uint8(fg_mask))
            fg_mask = self.transform_mask(fg_mask)
        else:
            fg_mask = torch.zeros([1, *image.size()[1:]])

        if self.split == DatasetSplit.TEST:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path)
                mask = self.transform_mask(mask)
            else:
                mask = torch.zeros([1, *image.size()[1:]])
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "fg_mask": fg_mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-1:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}
        fgmask_paths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            fg_mask_path = os.path.join(self.source, "fg_mask", classname)
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}
            fgmask_paths_per_class[classname] = {}

            for anomaly in anomaly_types:
                # class/good
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                                                     classname
                                                                 ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                                                     classname
                                                                 ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST:
                    pass
                    # anomaly_fgmask_path = os.path.join(fg_mask_path, anomaly)
                    # anomaly_fgmasks = sorted(os.listdir(anomaly_fgmask_path))
                    # fgmask_paths_per_class[classname][anomaly] = [
                    #     os.path.join(anomaly_fgmask_path, x) for x in anomaly_fgmasks
                    # ]
                    if anomaly != "good":
                        anomaly_mask_path = os.path.join(maskpath, anomaly)
                        anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                        maskpaths_per_class[classname][anomaly] = [
                            os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                        ]
                    else:
                        maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        random.seed(self.seed)

        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                if self.split == DatasetSplit.TRAIN:
                    for k in range(self.shot):
                        random_choose = random.randint(0, (len(imgpaths_per_class[classname][anomaly]) - 1))
                        image_path = imgpaths_per_class[classname][anomaly][random_choose]
                        data_tuple = [classname, anomaly, image_path]
                        data_tuple.append(None)
                        filename = os.path.basename(image_path)
                        filename = filename.split(".")[0]
                        fg_mask = os.path.join(fg_mask_path, filename+".png")
                        if os.path.exists(fg_mask):
                            data_tuple.append(fg_mask)
                        else:
                            data_tuple.append(None)

                        data_to_iterate.append(data_tuple)

                if self.split == DatasetSplit.TEST:
                    for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                        data_tuple = [classname, anomaly, image_path]

                        if anomaly != "good":
                            data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                        else:
                            data_tuple.append(None)
                        data_tuple.append(None)
                        data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate