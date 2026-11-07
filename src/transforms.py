from __future__ import annotations

from PIL import Image
from torchvision import transforms


class ResizeLongestSidePad:
    """Resize while preserving aspect ratio, then zero-pad to a square canvas."""

    def __init__(self, target_size: int) -> None:
        self.target_size = target_size

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size: {(width, height)}")

        scale = self.target_size / float(max(width, height))
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))

        resized = image.resize((new_w, new_h), resample=Image.BILINEAR)
        canvas = Image.new("RGB", (self.target_size, self.target_size), color=(0, 0, 0))

        pad_left = (self.target_size - new_w) // 2
        pad_top = (self.target_size - new_h) // 2
        canvas.paste(resized, (pad_left, pad_top))
        return canvas


def build_base_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            ResizeLongestSidePad(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_train_transform(image_size: int, color_jitter_probability: float) -> transforms.Compose:
    return transforms.Compose(
        [
            ResizeLongestSidePad(image_size),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.1, contrast=0.1)],
                p=float(color_jitter_probability),
            ),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.02, 0.02),
                scale=(0.95, 1.05),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
