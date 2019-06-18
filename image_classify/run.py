from . import data, model


def start(data_dir, netname, epochs):
    print("Create image transformations")
    image_transforms = data.transforms()

    print("Create image datasets")
    image_datasets = data.datasets(data_dir, image_transforms)

    print("Create image dataloaders")
    image_dataloaders = data.dataloaders(image_datasets)

    print("Create model")
    num_in_features = model.num_in_features(netname)
    classifier = model.classifier(
        num_in_features, hidden_layers=None, num_out_features=102
    )
    net = model.create_model(netname)

    for param in net.parameters():
        param.requires_grad = False

    net.classifier = classifier
    criterion = model.criterion(netname)
    optimizer = model.optimizer(net)
    sched = model.scheduler(optimizer)

    print("Train model")
    model.train(
        image_dataloaders, net, criterion, optimizer, sched, epochs
    )

    print("Save model")
    model.save(
        image_dataloaders, net, classifier, optimizer, sched, epochs
    )
