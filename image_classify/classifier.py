import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


def create_model(classifier, model_name='densenet'):
    if model_name == "densenet":
        model = models.densenet161(pretrained=True)
        num_in_features = 2208
        print(model)
    elif model_name == "vgg":
        model = models.vgg19(pretrained=True)
        num_in_features = 25088
        print(model.classifier)
    else:
        print("Unknown model, please choose 'densenet' or 'vgg'")
        return

    # Only train the classifier parameters, feature parameters are frozen
    if model_name == "densenet":
        model.classifier = classifier
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adadelta(
            model.parameters()
        )  # Adadelta #weight optim.Adam(model.parameters(), lr=0.001, momentum=0.9)
        # optimizer_conv = optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.001, momentum=0.9)
        sched = optim.lr_scheduler.StepLR(optimizer, step_size=4)
    elif model_name == "vgg":
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)
        sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    else:
        pass
    return model


def save_checkpoint(model, classifier, optimizer, sched, epochs):
    checkpoint = {
        "input_size": 2208,
        "output_size": 102,
        "epochs": epochs,
        "batch_size": 64,
        "model": models.densenet161(pretrained=True),
        "classifier": classifier,
        "scheduler": sched,
        "optimizer": optimizer.state_dict(),
        "state_dict": model.state_dict(),
        "class_to_idx": model.class_to_idx,
    }

    torch.save(checkpoint, "checkpoint_ic_d161.pth")
