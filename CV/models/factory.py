import timm
import torch.nn as nn

def create_model_factory(model_name: str, num_classes: int, pretrained: bool = False):
    """
    timm을 사용하여 모델을 생성합니다.
    Args:
        model_name: 'resnet18', 'resnet50', 'vit_tiny_patch16_224', 'vit_small_patch16_224' 등
    """
    try:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        return model
    except Exception as e:
        raise ValueError(f"Model {model_name} not found in timm library. Error: {e}")