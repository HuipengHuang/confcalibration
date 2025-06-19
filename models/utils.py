import os
import torch
import torchvision.models as models
from .naive_model import NaiveModel
from .ts import TemperatureScaling
from .vs import VectorScaling
from .ps import PlattScaling
from .conftr import ConfTr
from .linear_probing import LinearProbing

def build_net(device, args):
    model_type = args.model
    pretrained = (args.pretrained == "True")
    num_classes = args.num_classes

    if model_type == 'resnet18':
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnet34":
        net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnet101":
        net = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnet152":
        net = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "densenet121":
        net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "densenet161":
        net = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnext50":
        net = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if args.dataset != "imagenet":
        if hasattr(net, "fc"):
            net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
        else:
            net.classifier = torch.nn.Linear(net.classifier.in_features, num_classes)

    if args.load == "True":
        load_model(args, net)
    return net.to(device)

def build_model(device, args):
    net = build_net(device, args)
    method = args.method
    if method is None:
        model = NaiveModel(net, device, args)
    elif method == "ts":
        model = TemperatureScaling(net, device, args)
    elif method == "vs":
        model = VectorScaling(net, device, args)
    elif method == "ps":
        model = PlattScaling(net, device, args)
    elif method == "conftr":
        model = ConfTr(net, device, args)
    elif method == "linear_probing":
        model = LinearProbing(net, device, args)
    else:
        raise NotImplementedError
    return model

def load_model(args, net):
        p = f"./data/{args.dataset}_{args.model}{0}net.pth"


        net.load_state_dict(torch.load(p))

def save_model(args, model):
    i = 0
    while (True):
        p = f"./data/{args.dataset}_{args.model}{i}net.pth"

        if os.path.exists(p):
            i += 1
            continue
        torch.save(model.net.state_dict(), p)
        break

