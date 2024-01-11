# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Thought: Create another function, build_combined_dataset().
#          - This function is for concatenate datasets from different sources.
#          - You should pass in global normalization stats to this function.
#          - If this function is built, we should modify main_finetune_modified.py. 

def build_dataset(is_train, args, data_source=None, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):

    transform = build_transform(is_train, mean, std, args)

    if args.normalization == "N/A":
        # print("Global normalization is False!!!")
        root = os.path.join(args.data_path, is_train)
    else: # args.normalization == "local" or args.normalization == "global"
        # print("Global normalization is True!!!")
        root = os.path.join(args.data_path, data_source, is_train)

    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset


def build_transform(is_train, mean, std, args):
    mean = mean
    std = std
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
            # Thought: I only have to change mean & std parts.
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

    if args.normalization == "global":
        t.append(transforms.Normalize(args.global_mean, args.global_std))
    else:
        t.append(transforms.Normalize(mean, std))
    
    return transforms.Compose(t)
