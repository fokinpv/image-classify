import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

from PIL import Image


# Load a checkpoint and rebuild the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location="cpu")
    model = checkpoint["model"]
    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["state_dict"])
    print(checkpoint["class_to_idx"])

    for param in model.parameters():
        param.requires_grad = False

    return model, checkpoint["classes"]


def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    # Process a PIL image for use in a PyTorch model
    # tensor.numpy().transpose(1, 2, 0)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
              mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    image = preprocess(image)
    return image


def do(image_path, model, classes, topk=2):
    """
    Predict the class (or classes) of an image
    using a trained deep learning model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Implement the code to predict the class from an image file
    img = Image.open(image_path)
    img = process_image(img)

    # Convert 2D image to 1D vector
    img = np.expand_dims(img, 0)

    img = torch.from_numpy(img)

    model.eval()
    inputs = Variable(img).to(device)
    logits = model.forward(inputs)

    ps = F.softmax(logits, dim=1)
    topk = ps.cpu().topk(topk)

    probs, preds = (e.data.numpy().squeeze().tolist() for e in topk)
    print(probs, preds)
    return classes[preds[0]]
