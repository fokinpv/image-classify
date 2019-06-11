import copy
import time

import torch
import torch.nn as nn
import torchvision as tv
from torch import optim


def num_in_features(model_name):
    if model_name == 'densenet':
        return 2208
    if model_name == 'vgg':
        return 25088


def create_model(model_name):
    if model_name == "densenet":
        model = tv.models.densenet161(pretrained=True)
    elif model_name == "vgg":
        model = tv.models.vgg19(pretrained=True)
    else:
        print("Unknown model, please choose 'densenet' or 'vgg'")
        return
    return model


def criterion(model_name):
    if model_name == 'densenet':
        return nn.CrossEntropyLoss()
    if model_name == 'vgg':
        return nn.NLLLoss()


def optimizer(model):
    return optim.Adadelta(model.parameters())


def scheduler(optimizer):
    return optim.lr_scheduler.StepLR(optimizer, step_size=4)


def classifier(num_in_features, hidden_layers, num_out_features):
    classifier = nn.Sequential()
    if hidden_layers is None:
        classifier.add_module("fc0", nn.Linear(num_in_features, 102))
    else:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        classifier.add_module(
            "fc0", nn.Linear(num_in_features, hidden_layers[0])
        )
        classifier.add_module("relu0", nn.ReLU())
        classifier.add_module("drop0", nn.Dropout(0.6))
        classifier.add_module("relu1", nn.ReLU())
        classifier.add_module("drop1", nn.Dropout(0.5))
        for i, (h1, h2) in enumerate(layer_sizes):
            classifier.add_module("fc" + str(i + 1), nn.Linear(h1, h2))
            classifier.add_module("relu" + str(i + 1), nn.ReLU())
            classifier.add_module("drop" + str(i + 1), nn.Dropout(0.5))
        classifier.add_module(
            "output", nn.Linear(hidden_layers[-1], num_out_features)
        )

    return classifier


# Adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train(
        dataloaders, model, criterion, optimizer, sched, num_epochs
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_sizes = {
        x: len(dataloaders[x].dataset) for x in ["train", "valid"]
    }
    model.to(device)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        # sched.step()
                        loss.backward()

                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(
                "{} Loss: {:.4f} Acc: {:.4f}"
                .format(phase, epoch_loss, epoch_acc)
            )

            # deep copy the model
            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def save(dataloaders, model, classifier, optimizer, scheduler, *, epochs):
    checkpoint = {
        'input_size': 2208,
        'output_size': 102,
        'epochs': epochs,
        'batch_size': 64,
        #  'model': models.densenet161(pretrained=True),
        'model': model,
        'classifier': classifier,
        'scheduler': scheduler,
        'optimizer': optimizer.state_dict(),
        'state_dict': model.state_dict(),
        'class_to_idx': dataloaders['train'].dataset.class_to_idx,
        'classes': dataloaders['train'].dataset.classes,
    }
    torch.save(checkpoint, 'checkpoint.pth')
