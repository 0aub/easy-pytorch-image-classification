#  +-+ +-+ +-+ +-+ +-+ +-+
#  |i| |m| |p| |o| |r| |t|
#  +-+ +-+ +-+ +-+ +-+ +-+

import torch
from torchvision import models


#  +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+
#  |n| |e| |t| |w| |o| |r| |k| |s|
#  +-+ +-+ +-+ +-+ +-+ +-+ +-+ +-+


def prepare_model(model_name, model, n_classes):
    """
    Prepares a pre-trained neural network model for transfer learning by freezing all of its layers, 
    except for the final classification layer, which is replaced with a new one with the specified number of classes.

    Args:
        model_name (str): The name of the pre-trained model to prepare.
        model (torch.nn.Module): The pre-trained model to prepare.
        n_classes (int): The number of classes for the new classification layer.

    Returns:
        The prepared model with the new classification layer.

    """
    # Freeze training for all layers
    for param in model.parameters():
        param.require_grad = False

    # Replace the final classification layer with a new one with the specified number of classes
    if model_name in ['alexnet', 'vgg']:
        num_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_features, n_classes)
    elif model_name in ['convnext', 'resnet', 'resnext', 'shufflenet', 'googlenet', 'inception']:
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, n_classes)
    elif model_name in ['mnasnet', 'densenet']:
        num_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_features, n_classes)
    elif model_name in ['mobilenet_v3']:
        num_features = model.classifier[1].channels
        model.classifier[1] = torch.nn.Linear(num_features, n_classes)
    elif model_name in ['efficientnet', 'mobilenet_v2']:
        num_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_features, n_classes)
    elif model_name in ['swin', 'regnet']:
        num_features = model.head.in_features
        model.head = torch.nn.Linear(num_features, n_classes)
    elif model_name in ['vit']:
        num_features = model.heads[0].in_features
        model.heads[0] = torch.nn.Linear(num_features, n_classes)
    elif model_name in ['maxvit']:
        num_features = model.classifier[-1].in_features
        model.head = torch.nn.Linear(num_features, n_classes)
    elif model_name in ['squeezenet']:
        # replace the last convolutional layer with a new one with the desired number of output channels
        # keep the same kernel size and stride as the original one
        # see https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py#L104-L105
        # and https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py#L118-L119
        # for reference
        last_conv_layer_index = -2 if '1_0' else -3 # depends on squeezenet version 1.0 or 1.1
        last_conv_layer_old = list(model.classifier.children())[last_conv_layer_index]
        last_conv_layer_new = torch.nn.Conv2d(
            last_conv_layer_old.in_channels,
            n_classes,
            kernel_size=last_conv_layer_old.kernel_size,
            stride=last_conv_layer_old.stride,
            padding=last_conv_layer_old.padding,
            bias=True,
            groups=last_conv_layer_old.groups
        )
        model.classifier[last_conv_layer_index] = last_conv_layer_new
    else:
        raise ValueError(f'[ERROR]  could not load {model_name}. An error occur in replacing the output layer.')

    return model


def load_pretrained_model(model_name):
    """
    Loads a pretrained deep learning model from the `models` module in the `torchvision` library.

    Args:
        model_name (str): A string indicating the name of the pretrained model to load. 
            The function supports a specific set of model names, including alexnet, convnext_tiny,
            convnext_small, convnext_base, convnext_large, densenet121, densenet161, densenet169,
            densenet201, efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, 
            efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7, efficientnet_v2_s,
            efficientnet_v2_m, efficientnet_v2_l, googlenet, inception_v3, maxvit_t, mnasnet0_5,
            mnasnet0_75, mnasnet1_0, mnasnet1_3, mobilenet_v2, mobilenet_v3_small, and 
            mobilenet_v3_large.

    Returns:
        A PyTorch model instance loaded with pretrained weights.

    Raises:
        ValueError: If the `model_name` argument is not one of the supported model names.

    Example:
        To load the pretrained `alexnet` model, call the function as follows:
        >>> model = load_pretrained_model('alexnet')
    """
    if 'alexnet' in model_name:
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        if model_name.strip() == 'alexnet':
            return models.alexnet(pretrained=True)
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'convnext' in model_name:
        if model_name.strip() == 'convnext_tiny':
            return models.convnext_tiny(pretrained=True)
        elif model_name.strip() == 'convnext_small':
            return models.convnext_small(pretrained=True)
        elif model_name.strip() == 'convnext_base':
            return models.convnext_base(pretrained=True)
        elif model_name.strip() == 'convnext_large':
            return models.convnext_large(pretrained=True)
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'densenet' in model_name:
        if model_name.strip() == 'densenet121':
            return models.densenet121(pretrained=True)
        elif model_name.strip() == 'densenet161':
            return models.densenet161(pretrained=True)
        elif model_name.strip() == 'densenet169':
            return models.densenet169(pretrained=True)
        elif model_name.strip() == 'densenet201':
            return models.densenet201(pretrained=True)
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'efficientnet' in model_name:
        if model_name.strip() == 'efficientnet_b0':
            return models.efficientnet_b0(pretrained=True)
        elif model_name.strip() == 'efficientnet_b1':
            return models.efficientnet_b1(pretrained=True)
        elif model_name.strip() == 'efficientnet_b2':
            return models.efficientnet_b2(pretrained=True)
        elif model_name.strip() == 'efficientnet_b3':
            return models.efficientnet_b3(pretrained=True)
        elif model_name.strip() == 'efficientnet_b4':
            return models.efficientnet_b4(pretrained=True)
        elif model_name.strip() == 'efficientnet_b5':
            return models.efficientnet_b5(pretrained=True)
        elif model_name.strip() == 'efficientnet_b6':
            return models.efficientnet_b6(pretrained=True)
        elif model_name.strip() == 'efficientnet_b7':
            return models.efficientnet_b7(pretrained=True)
        elif model_name.strip() == 'efficientnet_v2_s':
            return models.efficientnet_v2_s(pretrained=True)
        elif model_name.strip() == 'efficientnet_v2_m':
            return models.efficientnet_v2_m(pretrained=True)
        elif model_name.strip() == 'efficientnet_v2_l':
            return models.efficientnet_v2_l(pretrained=True)
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'googlenet' in model_name:
        if model_name.strip() == 'googlenet':
            return models.googlenet(pretrained=True)
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'inception' in model_name:
        if model_name.strip() == 'inception_v3':
            return models.inception_v3(pretrained=True)
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'maxvit' in model_name:
        if model_name.strip() == 'maxvit_t':
            return models.maxvit_t(pretrained=True)
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'mnasnet' in model_name:
        if model_name.strip() == 'mnasnet0_5':
            return models.mnasnet0_5(pretrained=True)
        elif model_name.strip() == 'mnasnet0_75':
            return models.mnasnet0_75(pretrained=True)
        elif model_name.strip() == 'mnasnet1_0':
            return models.mnasnet1_0(pretrained=True)
        elif model_name.strip() == 'mnasnet1_3':
            return models.mnasnet1_3(pretrained=True)
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'mobilenet' in model_name:
        if model_name.strip() == 'mobilenet_v2':
            return models.mobilenet_v2(pretrained=True)
        elif model_name.strip() == 'mobilenet_v3_small':
            return models.mobilenet_v3_small(pretrained=True)
        elif model_name.strip() == 'mobilenet_v3_large':
            return models.mobilenet_v3_large(pretrained=True)
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'regnet' in model_name:
        if model_name.strip() == 'regnet_y_400mf':
            return models.regnet_y_400mf(pretrained=True)
        elif model_name.strip() == 'regnet_y_800mf':
            return models.regnet_y_800mf(pretrained=True)
        elif model_name.strip() == 'regnet_y_1_6gf':
            return models.regnet_y_1_6gf(pretrained=True)
        elif model_name.strip() == 'regnet_y_3_2gf':
            return models.regnet_y_3_2gf(pretrained=True)
        elif model_name.strip() == 'regnet_y_8gf':
            return models.regnet_y_8gf(pretrained=True)
        elif model_name.strip() == 'regnet_y_16gf':
            return models.regnet_y_16gf(pretrained=True)
        elif model_name.strip() == 'regnet_y_32gf':
            return models.regnet_y_32gf(pretrained=True)
        elif model_name.strip() == 'regnet_y_128gf':
            return models.regnet_y_128gf(pretrained=True)
        elif model_name.strip() == 'regnet_x_400mf':
            return models.regnet_x_400mf(pretrained=True)
        elif model_name.strip() == 'regnet_x_800mf':
            return models.regnet_x_800mf(pretrained=True)
        elif model_name.strip() == 'regnet_x_1_6gf':
            return models.regnet_x_1_6gf(pretrained=True)
        elif model_name.strip() == 'regnet_x_3_2gf':
            return models.regnet_x_3_2gf(pretrained=True)
        elif model_name.strip() == 'regnet_x_8gf':
            return models.regnet_x_8gf(pretrained=True)
        elif model_name.strip() == 'regnet_x_16gf':
            return models.regnet_x_16gf(pretrained=True)
        elif model_name.strip() == 'regnet_x_32gf':
            return models.regnet_x_32gf(pretrained=True)
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'resnet' in model_name:
        if model_name.strip() == 'resnet18':
            return models.resnet18(pretrained=True)
        elif model_name.strip() == 'resnet34':
            return models.resnet34(pretrained=True)
        elif model_name.strip() == 'resnet50':
            return models.resnet50(pretrained=True)
        elif model_name.strip() == 'resnet101':
            return models.resnet101(pretrained=True)
        elif model_name.strip() == 'resnet152':
            return models.resnet152(pretrained=True)
        elif model_name.strip() == 'resnext50_32x4d':
            return models.resnext50_32x4d(pretrained=True)
        elif model_name.strip() == 'resnext101_32x8d':
            return models.resnext101_32x8d(pretrained=True)
        elif model_name.strip() == 'resnext101_64x4d':
            return models.resnext101_64x4d(pretrained=True)
        elif model_name.strip() == 'wide_resnet50_2':
            return models.wide_resnet50_2(pretrained=True)
        elif model_name.strip() == 'wide_resnet101_2':
            return models.wide_resnet101_2(pretrained=True)
    elif 'resnext' in model_name:
        if model_name.strip() == 'resnext50_32x4d':
            return models.resnext50_32x4d(pretrained=True)
        elif model_name.strip() == 'resnext101_32x8d':
            return models.resnext101_32x8d(pretrained=True)
        elif model_name.strip() == 'resnext101_64x4d':
            return models.resnext101_64x4d(pretrained=True)
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'shufflenet' in model_name:
        if model_name.strip() == 'shufflenet_v2_x0_5':
            return models.shufflenet_v2_x0_5(pretrained=True)
        elif model_name.strip() == 'shufflenet_v2_x1_0':
            return models.shufflenet_v2_x1_0(pretrained=True)
        elif model_name.strip() == 'shufflenet_v2_x1_5':
            return models.shufflenet_v2_x1_5(pretrained=True)
        elif model_name.strip() == 'shufflenet_v2_x2_0':
            return models.shufflenet_v2_x2_0(pretrained=True)
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'squeezenet' in model_name:
        if model_name.strip() == 'squeezenet1_0':
            return models.squeezenet1_0(pretrained=True)
        elif model_name.strip() == 'squeezenet1_1':
            return models.squeezenet1_1(pretrained=True)
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')        
    elif 'swin' in model_name:
        if model_name.strip() == 'swin_t':
            return models.swin_t(pretrained=True)
        elif model_name.strip() == 'swin_s':
            return models.swin_s(pretrained=True)
        elif model_name.strip() == 'swin_b':
            return models.swin_b(pretrained=True)
        elif model_name.strip() == 'swin_v2_t':
            return models.swin_v2_t(pretrained=True)
        elif model_name.strip() == 'swin_v2_s':
            return models.swin_v2_s(pretrained=True)
        elif model_name.strip() == 'swin_v2_b':
            return models.swin_v2_b(pretrained=True)
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'vgg' in model_name:
        if model_name.strip() == 'vgg11':
            return models.vgg11(pretrained=True)
        elif model_name.strip() == 'vgg13':
            return models.vgg13(pretrained=True)
        elif model_name.strip() == 'vgg16':
            return models.vgg16(pretrained=True)
        elif model_name.strip() == 'vgg19':
            return models.vgg19(pretrained=True)
        elif model_name.strip() == 'vgg11_bn':
            return models.vgg11_bn(pretrained=True)
        elif model_name.strip() == 'vgg13_bn':
            return models.vgg13_bn(pretrained=True)
        elif model_name.strip() == 'vgg16_bn':
            return models.vgg16_bn(pretrained=True)
        elif model_name.strip() == 'vgg19_bn':
            return models.vgg19_bn(pretrained=True)
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')        
    elif 'vit' in model_name:
        if model_name.strip() == 'vit_b_16':
            return models.vit_b_16(pretrained=True)
        elif model_name.strip() == 'vit_b_32':
            return models.vit_b_32(pretrained=True)
        elif model_name.strip() == 'vit_l_16':
            return models.vit_l_16(pretrained=True)
        elif model_name.strip() == 'vit_l_32':
            return models.vit_l_32(pretrained=True)
        elif model_name.strip() == 'vit_h_14':
            return models.vit_h_14(pretrained=True)
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    else:
        raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    

def pretrained_network(model_name, n_classes, **kwargs):
    """
    Returns a pretrained neural network model for a specified number of output classes.

    Args:
    model_name (str): Name of the pretrained model to use. Supported options include "resnet18", "resnet34",
                      "resnet50", "resnet101", and "resnet152".
    n_classes (int): Number of output classes for the model.
    **kwargs: Additional arguments that will be passed to the `prepare_model` function.

    Returns:
    A pretrained neural network model with a specified number of output classes.
    """
    model_name = model_name.lower()
    return prepare_model(model_name, load_pretrained_model(model_name), n_classes)
