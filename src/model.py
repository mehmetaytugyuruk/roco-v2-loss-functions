from __future__ import annotations

from torchvision.models import (
    ResNet50_Weights, resnet50,
    EfficientNet_B0_Weights, efficientnet_b0,
    EfficientNet_B4_Weights, efficientnet_b4,
    ConvNeXt_Base_Weights, convnext_base,
    Swin_B_Weights, swin_b,
    ViT_B_16_Weights, vit_b_16,
)
from torchvision.models.vision_transformer import interpolate_embeddings
import torch.nn as nn

def build_model(
    num_classes: int,
    pretrained: bool = True,
    model_name: str = "efficientnet_b0",
    image_size: int = 224,
) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        dropout_p = model.classifier[0].p if isinstance(model.classifier[0], nn.Dropout) else 0.2
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "efficientnet_b4":
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b4(weights=weights)
        in_features = model.classifier[-1].in_features
        dropout_p = model.classifier[0].p if isinstance(model.classifier[0], nn.Dropout) else 0.4
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "convnext_base":
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_base(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_name == "swin_b":
        weights = Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
        model = swin_b(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    elif model_name == "vit_b_16":
        if pretrained and image_size != 224:
            model = vit_b_16(weights=None, image_size=image_size, num_classes=num_classes)
            state_dict = ViT_B_16_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
            state_dict = interpolate_embeddings(
                image_size=image_size,
                patch_size=16,
                model_state=state_dict,
                reset_heads=True,
            )
            model.load_state_dict(state_dict, strict=False)
        else:
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            model = vit_b_16(weights=weights, image_size=image_size, num_classes=num_classes)
            if model.heads.head.out_features != num_classes:
                in_features = model.heads.head.in_features
                model.heads.head = nn.Linear(in_features, num_classes)

    elif model_name == "deit_base_patch16_224":
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "model_name='deit_base_patch16_224' requires the 'timm' package. "
                "Install it via requirements.txt before training."
            ) from exc

        if image_size % 16 != 0:
            raise ValueError("DeiT-B/16 requires image_size divisible by 16.")
        model = timm.create_model(
            "deit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=image_size,
        )

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model
