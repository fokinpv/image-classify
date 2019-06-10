import torch

from . import data, model


def start(data_dir, model_name):
    print("Create image transformations")
    image_transforms = data.transforms()
    print(image_transforms)

    print("Create image datasets")
    image_datasets = data.datasets(data_dir, image_transforms)
    print(image_datasets)

    print("Create image dataloaders")
    image_dataloaders = data.dataloaders(image_datasets)
    print(image_dataloaders)

    print("Create model")
    num_in_features = model.num_in_features(model_name)
    classifier = model.classifier(
        num_in_features, hidden_layers=None, num_out_features=102
    )
    model_ = model.create_model(model_name)
    model_.classifier = classifier
    #  print(model_)
    criterion = model.criterion(model_name)
    optimizer = model.optimizer(model_)
    sched = model.scheduler(optimizer)

    print("Train model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train(device, image_dataloaders, model_, criterion, optimizer, sched)
