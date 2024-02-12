# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from typing import Any, Callable, List, Optional, Tuple, Dict
import torch
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pandas as pd

class RegressionDataset(datasets.VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform)
        self.samples, self.norm_params = self.make_dataset(self.root)
        self.loader = datasets.folder.default_loader

    @staticmethod
    def make_dataset(
        directory: str,
    ) -> Tuple[List[Tuple[str, int]], Dict[str, torch.Tensor]]:
        directory = os.path.expanduser(directory)

        targets_df = pd.read_csv(directory + ".csv", header=0,
                                 dtype={"file": str, "target": float})
        targets = dict(zip(targets_df["file"], targets_df["target"]))

        instances = list()
        for target_file, target_value in targets.items():
            path = os.path.join(directory, target_file)
            item = path, target_value
            instances.append(item)

        target_values = torch.tensor(list(targets.values()))
        dataset_mean = torch.mean(target_values)
        dataset_std = torch.std(target_values)
        norm_params = {"mean": dataset_mean, "std": dataset_std}

        return instances, norm_params
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, target
    
    def __len__(self) -> int:
        return len(self.samples)
    
class RegressionStackDataset(datasets.VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform)
        self.samples, self.norm_params = self.make_dataset(self.root)
        self.loader = datasets.folder.default_loader

    @staticmethod
    def make_dataset(
        directory: str,
    ) -> Tuple[List[Tuple[str, int]], Dict[str, torch.Tensor]]:
        directory = os.path.expanduser(directory)

        targets_df = pd.read_csv(directory + ".csv", header=0,
                                 dtype={"stack": str, "target": float})
        targets = dict(zip(targets_df["stack"], targets_df["target"]))

        instances = list()
        for target_stack, target_value in targets.items():
            stack_paths = list()
            for target_file in target_stack.split(":"):
                path = os.path.join(directory, target_file)
                stack_paths.append(path)
            item = stack_paths, target_value
            instances.append(item)

        target_values = torch.tensor(list(targets.values()))
        dataset_mean = torch.mean(target_values)
        dataset_std = torch.std(target_values)
        norm_params = {"mean": dataset_mean, "std": dataset_std}

        return instances, norm_params
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        stack_paths, target = self.samples[index]
        sample = [self.loader(path) for path in stack_paths]

        if self.transform is not None:
            for i in range(len(sample)):
                sample[i] = self.transform(sample[i])

        sample = torch.stack(sample)
        return sample, target
    
    def __len__(self) -> int:
        return len(self.samples)

def build_dataset(is_train, args):
    
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)

    if not args.stack:
        dataset = RegressionDataset(root, transform=transform)
    else:
        dataset = RegressionStackDataset(root, transform=transform)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train=='train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
