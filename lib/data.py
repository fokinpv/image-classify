import os

import torch
import torchvision as tv


def transforms():
    data_transforms = {
        "train": tv.transforms.Compose(
            [
                tv.transforms.RandomRotation(30),
                tv.transforms.RandomResizedCrop(224),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        ),
        "valid": tv.transforms.Compose(
            [
                tv.transforms.Resize(256),
                tv.transforms.CenterCrop(224),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        ),
    }
    return data_transforms


def datasets(data_dir, transforms):
    image_datasets = {
        x: tv.datasets.ImageFolder(os.path.join(data_dir, x), transforms[x])
        for x in ["train", "valid"]
    }
    return image_datasets


def dataloaders(datasets, batch_size=64, num_workers=4):
    image_dataloaders = {
        x: torch.utils.data.DataLoader(
            datasets[x], batch_size=batch_size,
            shuffle=True, num_workers=num_workers
        )
        for x in ["train", "valid"]
    }
    return image_dataloaders
