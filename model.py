import copy
import torch
import torch.nn as nn

from models import resnet, simplecnn3d, cnnlstm
from torchinfo import summary


def set_parameter_requires_grad(model, feature_extracting: bool, num_ft_layers: int):
    """
    Freeze the weights of the model is feature_extracting=True
    Fine tune layers >= num_ft_layers
    Batch Normalization: https://keras.io/guides/transfer_learning/

    Args:
        model: PyTorch model
        feature_extracting (bool): A bool to set all parameters to be trainable or not
        num_ft_layers (int): Number of layers to freeze and unfreezing the rest
    """
    if feature_extracting:
        if num_ft_layers != -1:
            for i, module in enumerate(model.modules()):
                if i >= num_ft_layers:
                    if not isinstance(module, nn.BatchNorm3d):
                        module.requires_grad_(True)
                else:
                    module.requires_grad_(False)
        else:
            for param in model.parameters():
                param.requires_grad = False
    # not recommended to set feature_extracting=True when use_pretrained=True
    else:
        for param in model.parameters():
            param.requires_grad = True


def _create_classifier(num_ftrs: int, embedding_size: int, num_classes: int):
    """Add a classifier head with 2 FC layers

    Args:
        num_ftrs (int): Number of features from timm models
        embedding_size (int): Number of features in penultimate layer
        num_classes (int): Number of classes
    """
    head = nn.Sequential(
        nn.Linear(num_ftrs, embedding_size),
        nn.Linear(embedding_size, num_classes),
        # nn.Sigmoid()
    )
    return head


def build_models(
        model_name: str,
        num_classes: int,
        in_channels: int,
        embedding_size: int,
        feature_extract: bool = True,
        use_pretrained: bool = True,
        num_ft_layers: int = -1,
        bst_model_weights=None
):
    """
    Build various architectures to either train from scratch, finetune or as feature extractor.

    Args:
        model_name (str) : Name of model from `timm.list_models(pretrained=use_pretrained)`
        num_classes (int) : Number of output classes added as final layer
        in_channels (int) : Number of input channels
        embedding_size (int): Size of intermediate features
        feature_extract (bool): Flag for feature extracting.
                               False = finetune the whole model,
                               True = only update the new added layers params
        use_pretrained (bool): Pretraining parameter to pass to the model or if base_model_path is given use that to
                                initialize the model weights
        num_ft_layers (int) : Number of layers to finetune
                             Default = -1 (do not finetune any layers)
        bst_model_weights : Best weights obtained after training pretrained model
                            which will be used for further finetuning.

    Returns:
        model : A pytorch model
    """
    supported_models = ['resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 
                        'resnet152', 'resnet200', 'simple_cnn3d', 'cnnlstm_resnet18', 'cnnlstm_resnet50']
    model = None
    if model_name in supported_models:
        if model_name == 'resnet10':
            model = resnet.resnet10(sample_size=384, sample_duration=6)
        if model_name == 'resnet18':
            model = resnet.resnet18(sample_size=384, sample_duration=6)
        if model_name == 'simple_cnn3d':
            model = simplecnn3d.Simple3dCNN()
        if 'cnnlstm' in model_name:
            model = cnnlstm.CNNLSTM(model_name.split('_')[-1])

        set_parameter_requires_grad(model, feature_extract, num_ft_layers)
        num_ftrs = model.fc.in_features
        model.fc = _create_classifier(num_ftrs, embedding_size, num_classes)
    else:
        print("Invalid model name, exiting...")
        exit()
    # load best model dict for further finetuning
    if bst_model_weights is not None:
        pretrain_model = torch.load(bst_model_weights)
        best_model_wts = copy.deepcopy(pretrain_model.state_dict())
        if feature_extract and num_ft_layers != -1:
            model.load_state_dict(best_model_wts)
    summary(model, input_size=(1, 3, 6, 384, 384))
    return model
