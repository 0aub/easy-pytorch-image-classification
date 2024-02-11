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
        if model_name == 'alexnet':
            return models.alexnet(weights='AlexNet_Weights.DEFAULT')
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'convnext' in model_name:
        if model_name == 'convnext_tiny':
            return models.convnext_tiny(weights='ConvNeXt_Tiny_Weights.DEFAULT')
        elif model_name == 'convnext_small':
            return models.convnext_small(weights='ConvNeXt_Small _Weights.DEFAULT')
        elif model_name == 'convnext_base':
            return models.convnext_base(weights='ConvNeXt_Base_Weights.DEFAULT')
        elif model_name == 'convnext_large':
            return models.convnext_large(weights='ConvNeXt_Large_Weights.DEFAULT')
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'densenet' in model_name:
        if model_name == 'densenet121':
            return models.densenet121(weights='DenseNet121_Weights.DEFAULT')
        elif model_name == 'densenet161':
            return models.densenet161(weights='DenseNet161_Weights.DEFAULT')
        elif model_name == 'densenet169':
            return models.densenet169(weights='DenseNet169_Weights.DEFAULT')
        elif model_name == 'densenet201':
            return models.densenet201(weights='DenseNet201_Weights.DEFAULT')
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'efficientnet' in model_name:
        if model_name == 'efficientnet_b0':
            return models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
        elif model_name == 'efficientnet_b1':
            return models.efficientnet_b1(weights='EfficientNet_B1_Weights.DEFAULT')
        elif model_name == 'efficientnet_b2':
            return models.efficientnet_b2(weights='EfficientNet_B2_Weights.DEFAULT')
        elif model_name == 'efficientnet_b3':
            return models.efficientnet_b3(weights='EfficientNet_B3_Weights.DEFAULT')
        elif model_name == 'efficientnet_b4':
            return models.efficientnet_b4(weights='EfficientNet_B4_Weights.DEFAULT')
        elif model_name == 'efficientnet_b5':
            return models.efficientnet_b5(weights='EfficientNet_B5_Weights.DEFAULT')
        elif model_name == 'efficientnet_b6':
            return models.efficientnet_b6(weights='EfficientNet_B7_Weights.DEFAULT')
        elif model_name == 'efficientnet_b7':
            return models.efficientnet_b7(weights='EfficientNet_B7_Weights.DEFAULT')
        elif model_name == 'efficientnet_v2_s':
            return models.efficientnet_v2_s(weights='EfficientNet_V2_S_Weights.DEFAULT')
        elif model_name == 'efficientnet_v2_m':
            return models.efficientnet_v2_m(weights='EfficientNet_V2_M_Weights.DEFAULT')
        elif model_name == 'efficientnet_v2_l':
            return models.efficientnet_v2_l(weights='EfficientNet_V2_L_Weights.DEFAULT')
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'googlenet' in model_name:
        if model_name == 'googlenet':
            return models.googlenet(weights='GoogLeNet_Weights.DEFAULT')
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'inception' in model_name:
        if model_name == 'inception_v3':
            return models.inception_v3(weights='Inception_V3_Weights.DEFAULT')
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'maxvit' in model_name:
        if model_name == 'maxvit_t':
            return models.maxvit_t(weights='MaxVit_T_Weights.DEFAULT')
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'mnasnet' in model_name:
        if model_name == 'mnasnet0_5':
            return models.mnasnet0_5(weights='MNASNet0_5_Weights.DEFAULT')
        elif model_name == 'mnasnet0_75':
            return models.mnasnet0_75(weights='MNASNet0_75_Weights.DEFAULT')
        elif model_name == 'mnasnet1_0':
            return models.mnasnet1_0(weights='MNASNet1_0_Weights.DEFAULT')
        elif model_name == 'mnasnet1_3':
            return models.mnasnet1_3(weights='MNASNet1_3_Weights.DEFAULT')
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'mobilenet' in model_name:
        if model_name == 'mobilenet_v2':
            return models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
        elif model_name == 'mobilenet_v3_small':
            return models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.DEFAULT')
        elif model_name == 'mobilenet_v3_large':
            return models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.DEFAULT')
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'regnet' in model_name:
        if model_name == 'regnet_y_400mf':
            return models.regnet_y_400mf(weights='RegNet_Y_400MF_Weights.DEFAULT')
        elif model_name == 'regnet_y_800mf':
            return models.regnet_y_800mf(weights='RegNet_Y_800MF_Weights.DEFAULT')
        elif model_name == 'regnet_y_1_6gf':
            return models.regnet_y_1_6gf(weights='RegNet_Y_1_6GF_Weights.DEFAULT')
        elif model_name == 'regnet_y_3_2gf':
            return models.regnet_y_3_2gf(weights='RegNet_Y_3_2GF_Weights.DEFAULT')
        elif model_name == 'regnet_y_8gf':
            return models.regnet_y_8gf(weights='RegNet_Y_8GF_Weights.DEFAULT')
        elif model_name == 'regnet_y_16gf':
            return models.regnet_y_16gf(weights='RegNet_Y_16GF_Weights.DEFAULT')
        elif model_name == 'regnet_y_32gf':
            return models.regnet_y_32gf(weights='RegNet_Y_32GF_Weights.DEFAULT')
        elif model_name == 'regnet_y_128gf':
            return models.regnet_y_128gf(weights='RegNet_Y_128GF_Weights.DEFAULT')
        elif model_name == 'regnet_x_400mf':
            return models.regnet_x_400mf(weights='RegNet_X_400MF_Weights.DEFAULT')
        elif model_name == 'regnet_x_800mf':
            return models.regnet_x_800mf(weights='RegNet_X_800MF_Weights.DEFAULT')
        elif model_name == 'regnet_x_1_6gf':
            return models.regnet_x_1_6gf(weights='RegNet_X_1_6GF_Weights.DEFAULT')
        elif model_name == 'regnet_x_3_2gf':
            return models.regnet_x_3_2gf(weights='RegNet_X_3_2GF_Weights.DEFAULT')
        elif model_name == 'regnet_x_8gf':
            return models.regnet_x_8gf(weights='RegNet_X_8GF_Weights.DEFAULT')
        elif model_name == 'regnet_x_16gf':
            return models.regnet_x_16gf(weights='RegNet_X_16GF_Weights.DEFAULT')
        elif model_name == 'regnet_x_32gf':
            return models.regnet_x_32gf(weights='RegNet_X_32GF_Weights.DEFAULT')
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'resnet' in model_name:
        if model_name == 'resnet18':
            return models.resnet18(weights='ResNet18_Weights.DEFAULT')
        elif model_name == 'resnet34':
            return models.resnet34(weights='ResNet34_Weights.DEFAULT')
        elif model_name == 'resnet50':
            return models.resnet50(weights='ResNet50_Weights.DEFAULT')
        elif model_name == 'resnet101':
            return models.resnet101(weights='ResNet101_Weights.DEFAULT')
        elif model_name == 'resnet152':
            return models.resnet152(weights='ResNet152_Weights.DEFAULT')
        elif model_name == 'resnext50_32x4d':
            return models.resnext50_32x4d(weights='ResNet50_32x4d_Weights.DEFAULT')
        elif model_name == 'resnext101_32x8d':
            return models.resnext101_32x8d(weights='ResNet101_32x8d_Weights.DEFAULT')
        elif model_name == 'resnext101_64x4d':
            return models.resnext101_64x4d(weights='ResNet101_64x4d_Weights.DEFAULT')
        elif model_name == 'wide_resnet50_2':
            return models.wide_resnet50_2(weights='Wide_ResNet50_2_Weights.DEFAULT')
        elif model_name == 'wide_resnet101_2':
            return models.wide_resnet101_2(weights='Wide_ResNet101_2_Weights.DEFAULT')
    elif 'resnext' in model_name:
        if model_name == 'resnext50_32x4d':
            return models.resnext50_32x4d(weights='ResNeXt50_32X4D_Weights.DEFAULT')
        elif model_name == 'resnext101_32x8d':
            return models.resnext101_32x8d(weights='ResNeXt101_32X8D_Weights.DEFAULT')
        elif model_name == 'resnext101_64x4d':
            return models.resnext101_64x4d(weights='ResNeXt101_64X8D_Weights.DEFAULT')
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'shufflenet' in model_name:
        if model_name == 'shufflenet_v2_x0_5':
            return models.shufflenet_v2_x0_5(weights='ShuffleNet_V2_X0_5_Weights.DEFAULT')
        elif model_name == 'shufflenet_v2_x1_0':
            return models.shufflenet_v2_x1_0(weights='ShuffleNet_V2_X1_0_Weights.DEFAULT')
        elif model_name == 'shufflenet_v2_x1_5':
            return models.shufflenet_v2_x1_5(weights='ShuffleNet_V2_X1_5_Weights.DEFAULT')
        elif model_name == 'shufflenet_v2_x2_0':
            return models.shufflenet_v2_x2_0(weights='ShuffleNet_V2_X2_0_Weights.DEFAULT')
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'squeezenet' in model_name:
        if model_name == 'squeezenet1_0':
            return models.squeezenet1_0(weights='SqueezeNet1_0_Weights.DEFAULT')
        elif model_name == 'squeezenet1_1':
            return models.squeezenet1_1(weights='SqueezeNet1_1_Weights.DEFAULT')
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')        
    elif 'swin' in model_name:
        if model_name == 'swin_t':
            return models.swin_t(weights='Swin_T_Weights.DEFAULT')
        elif model_name == 'swin_s':
            return models.swin_s(weights='Swin_S_Weights.DEFAULT')
        elif model_name == 'swin_b':
            return models.swin_b(weights='Swin_B_Weights.DEFAULT')
        elif model_name == 'swin_v2_t':
            return models.swin_v2_t(weights='Swin_V2_T_Weights.DEFAULT')
        elif model_name == 'swin_v2_s':
            return models.swin_v2_s(weights='Swin_V2_S_Weights.DEFAULT')
        elif model_name == 'swin_v2_b':
            return models.swin_v2_b(weights='Swin_V2_B_Weights.DEFAULT')
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')
    elif 'vgg' in model_name:
        if model_name == 'vgg11':
            return models.vgg11(weights='VGG11_Weights.DEFAULT')
        elif model_name == 'vgg13':
            return models.vgg13(weights='VGG13_Weights.DEFAULT')
        elif model_name == 'vgg16':
            return models.vgg16(weights='VGG16_Weights.DEFAULT')
        elif model_name == 'vgg19':
            return models.vgg19(weights='VGG19_Weights.DEFAULT')
        elif model_name == 'vgg11_bn':
            return models.vgg11_bn(weights='VGG11_BN_Weights.DEFAULT')
        elif model_name == 'vgg13_bn':
            return models.vgg13_bn(weights='VGG13_BN_Weights.DEFAULT')
        elif model_name == 'vgg16_bn':
            return models.vgg16_bn(weights='VGG16_BN_Weights.DEFAULT')
        elif model_name == 'vgg19_bn':
            return models.vgg19_bn(weights='VGG19_BN_Weights.DEFAULT')
        else:
            raise ValueError(f'[ERROR]  could not load pretrained network because {model_name} model not found')        
    elif 'vit' in model_name:
        if model_name == 'vit_b_16':
            return models.vit_b_16(weights='ViT_B_16_Weights.DEFAULT')
        elif model_name == 'vit_b_32':
            return models.vit_b_32(weights='ViT_B_32_Weights.DEFAULT')
        elif model_name == 'vit_l_16':
            return models.vit_l_16(weights='ViT_L_16_Weights.DEFAULT')
        elif model_name == 'vit_l_32':
            return models.vit_l_32(weights='ViT_L_32_Weights.DEFAULT')
        elif model_name == 'vit_h_14':
            return models.vit_h_14(weights='ViT_H_14_Weights.DEFAULT')
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
