"""
src/utils/image_utils.py
────────────────────────────────────────────────────────────────
Image loading, validation, and preprocessing utilities.

Chest X-rays from NIH ChestX-ray14 are 8-bit grayscale PNGs.
Before feeding them to an ImageNet pre-trained model we need to:
  1. Load as grayscale
  2. Convert to 3-channel RGB by replicating the single channel
     (pre-trained models expect 3 channels)
  3. Normalise with ImageNet mean/std

These helpers are pure functions — no side effects, safe to use
in parallel DataLoader workers.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy imports — PIL and torch are only imported when a function
# that needs them is first called.  This keeps import time fast
# in contexts that only need config or logging. The TYPE_CHECKING
# import below is never executed at runtime — it only lets type
# checkers (and ruff) resolve the string annotations below.
if TYPE_CHECKING:
    import PIL.Image
    import torch


def load_xray(path: Path | str) -> "PIL.Image.Image":
    """Load a chest X-ray image as an 8-bit grayscale PIL Image.

    Converts any mode (RGBA, L, etc.) to RGB so the image is
    compatible with ImageNet pre-trained models.

    Args:
        path: Path to a PNG or JPEG X-ray image.

    Returns:
        RGB PIL Image.

    Raises:
        FileNotFoundError: If the image does not exist.
        OSError: If the file cannot be read as an image.

    Example::

        img = load_xray("data/raw/images/00000001_000.png")
        print(img.size, img.mode)   # (1024, 1024) RGB
    """
    from PIL import Image

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"X-ray image not found: {path}")

    img = Image.open(path)

    # NIH images are 8-bit grayscale ('L').
    # Pre-trained CNNs expect 3-channel RGB.
    # We replicate the single channel across R, G, B — this preserves
    # the full dynamic range without any colour distortion.
    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def validate_image(path: Path | str) -> bool:
    """Check whether a file is a valid, readable image.

    Args:
        path: Path to the file to validate.

    Returns:
        True if the file is a valid image; False otherwise.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            img.verify()   # checks file integrity without decoding pixels
        return True
    except (UnidentifiedImageError, OSError, SyntaxError):
        return False


def image_to_tensor(
    path: Path | str,
    transform=None,
) -> "torch.Tensor":
    """Load an X-ray and convert it to a normalised PyTorch tensor.

    Args:
        path:      Path to the X-ray image.
        transform: Optional torchvision transform to apply.
                   If None, the image is returned as a raw tensor
                   in [0, 1] range.

    Returns:
        Float tensor of shape (3, H, W).

    Example::

        from src.data.transforms import get_inference_transform
        tensor = image_to_tensor("xray.png", get_inference_transform())
        print(tensor.shape)   # torch.Size([3, 224, 224])
    """
    import torchvision.transforms.functional as TF

    img = load_xray(path)

    if transform is not None:
        tensor = transform(img)
    else:
        tensor = TF.to_tensor(img)   # scales [0,255] → [0.0,1.0]

    return tensor


def tensor_to_pil(tensor: "torch.Tensor") -> "PIL.Image.Image":
    """Convert a (3, H, W) float tensor back to a PIL Image.

    Reverses ImageNet normalisation so the pixel values are
    back in [0, 255] range for display.

    Args:
        tensor: Float tensor of shape (3, H, W), ImageNet-normalised.

    Returns:
        PIL Image suitable for display or saving.
    """
    from PIL import Image
    import torch

    from src.utils.config import DataConfig

    mean = torch.tensor(DataConfig.imagenet_mean).view(3, 1, 1)
    std  = torch.tensor(DataConfig.imagenet_std).view(3, 1, 1)

    # Reverse normalisation: x = (x_norm * std) + mean
    denorm = tensor.cpu().clone() * std + mean
    denorm = denorm.clamp(0, 1)

    # Convert to uint8 numpy array then to PIL
    np_img = (denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_img)


def overlay_heatmap(
    original: "PIL.Image.Image",
    heatmap:  np.ndarray,
    alpha:    float = 0.4,
    colormap: str   = "jet",
) -> "PIL.Image.Image":
    """Overlay a Grad-CAM heatmap on the original X-ray image.

    The heatmap is rescaled to the original image size and blended
    with the original using the given alpha weight.

    Args:
        original:  Original X-ray as a PIL Image (any size).
        heatmap:   Grad-CAM activation map as a 2D float numpy array
                   in [0, 1] range.
        alpha:     Blend weight for the heatmap.
                   0.0 = heatmap only, 1.0 = original only.
                   Default 0.4 gives a clear overlay.
        colormap:  Matplotlib colormap name.  "jet" is standard for
                   medical imaging papers (red = high activation).

    Returns:
        Blended PIL Image (RGB) at the same size as the original.

    Example::

        overlay = overlay_heatmap(original_img, cam, alpha=0.4)
        overlay.save("gradcam_overlay.png")
    """
    import cv2
    from PIL import Image

    # Resize heatmap to original image dimensions
    h, w   = original.size[1], original.size[0]
    cam_resized = cv2.resize(heatmap, (w, h))

    # Apply colour map to the grayscale heatmap
    cam_uint8 = np.uint8(255 * cam_resized)
    cmap_id   = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_JET)
    cam_coloured = cv2.applyColorMap(cam_uint8, cmap_id)
    cam_coloured = cv2.cvtColor(cam_coloured, cv2.COLOR_BGR2RGB)
    cam_pil      = Image.fromarray(cam_coloured)

    # Blend heatmap with original
    original_rgb = original.convert("RGB")
    blended      = Image.blend(cam_pil, original_rgb, alpha=alpha)

    return blended


def resize_for_display(
    image: "PIL.Image.Image",
    max_size: int = 512,
) -> "PIL.Image.Image":
    """Resize an image for display while preserving aspect ratio.

    Args:
        image:    PIL Image to resize.
        max_size: Maximum dimension (width or height).

    Returns:
        Resized PIL Image.
    """
    from PIL import Image

    w, h    = image.size
    longest = max(w, h)
    if longest <= max_size:
        return image

    scale  = max_size / longest
    new_w  = int(w * scale)
    new_h  = int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def get_image_stats(path: Path | str) -> dict[str, float | int | str]:
    """Return basic statistics about an X-ray image.

    Useful for data quality checks during dataset preparation.

    Args:
        path: Path to the image.

    Returns:
        Dict with keys: width, height, mode, mean_pixel, std_pixel.
    """

    img = load_xray(path)
    arr = np.array(img)

    return {
        "width":       img.size[0],
        "height":      img.size[1],
        "mode":        img.mode,
        "mean_pixel":  float(arr.mean()),
        "std_pixel":   float(arr.std()),
        "min_pixel":   int(arr.min()),
        "max_pixel":   int(arr.max()),
    }
