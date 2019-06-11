from . import data, model


def start(data_dir, model_name, epochs):
    print("Create image transformations")
    image_transforms = data.transforms()

    print("Create image datasets")
    image_datasets = data.datasets(data_dir, image_transforms)

    print("Create image dataloaders")
    image_dataloaders = data.dataloaders(image_datasets)

    print("Create model")
    num_in_features = model.num_in_features(model_name)
    classifier = model.classifier(
        num_in_features, hidden_layers=None, num_out_features=102
    )
    model_ = model.create_model(model_name)

    for param in model_.parameters():
        param.requires_grad = False

    model_.classifier = classifier
    criterion = model.criterion(model_name)
    optimizer = model.optimizer(model_)
    sched = model.scheduler(optimizer)

    print("Train model")
    model.train(
        image_dataloaders, model_, criterion, optimizer, sched, epochs
    )

    print("Save model")
    model.save(
        image_dataloaders, model_, classifier, optimizer, sched, epochs
    )
