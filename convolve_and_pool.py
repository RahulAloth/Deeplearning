# convolve_and_pool.py
# -*- coding: utf-8 -*-

import os
from typing import Tuple, Optional, List, Literal

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def print_env():
    """Print basic environment info."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")


def load_image_as_tensor(
    image_path: str,
    resize_to: Optional[int] = 450,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Load an image as a normalized float tensor in shape (1, C, H, W).

    Args:
        image_path: Path to the image file.
        resize_to: If provided, shorter side is resized to this value keeping aspect ratio.
        device: torch.device to place the tensor on.

    Returns:
        Tensor of shape (1, 3, H, W), float32 in [0, 1].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")

    transforms_list: List = []
    if isinstance(resize_to, int):
        # Resize the shorter side to resize_to, preserving aspect ratio.
        transforms_list.append(T.Resize(resize_to))
    transforms_list.append(T.ToTensor())  # (C,H,W), scaled to [0,1]

    transforms = T.Compose(transforms_list)
    img_tensor = transforms(img).unsqueeze(0)  # (1,C,H,W)

    if device is not None:
        img_tensor = img_tensor.to(device)

    return img_tensor


def get_same_padding(kernel_size: int, dilation: int = 1, stride: int = 1) -> int:
    """
    Compute 'same' padding for odd kernel sizes to preserve H and W.
    For odd kernels and stride=1: padding = (k - 1) / 2
    """
    effective_kernel = dilation * (kernel_size - 1) + 1
    # For stride=1, this preserves spatial size
    pad = (effective_kernel - 1) // 2
    return int(pad)


def make_depthwise_kernel_3ch(weight_2d: List[List[float]]) -> torch.Tensor:
    """
    Convert a 2D kernel (kH x kW) into a depthwise kernel suitable for RGB input:
      shape -> (out_channels=3, in_channels_per_group=1, kH, kW)
      applied with groups=3 in conv2d
    This applies the SAME kernel to each RGB channel independently.
    """
    k = torch.tensor(weight_2d, dtype=torch.float32)  # (kH, kW)
    k = k.unsqueeze(0).unsqueeze(0)  # (1,1,kH,kW)
    k = k.repeat(3, 1, 1, 1)        # (3,1,kH,kW) depthwise 3 channels
    return k


def convolve(
    img_tensor: torch.Tensor,
    kernel_2d: List[List[float]],
    padding: Literal["same", "valid"] = "valid"
) -> torch.Tensor:
    """
    Apply a depthwise 2D convolution to a 3-channel image tensor.

    Args:
        img_tensor: Input tensor of shape (1, 3, H, W).
        kernel_2d: 2D Python list kernel (kH x kW) applied per channel.
        padding: 'same' to preserve input H, W (for odd kernels), 'valid' for no padding.

    Returns:
        conv_tensor: Tensor of shape (1, 3, H_out, W_out).
    """
    assert img_tensor.dim() == 4 and img_tensor.size(1) == 3, \
        "img_tensor must be shape (1, 3, H, W)"
    k = make_depthwise_kernel_3ch(kernel_2d).to(img_tensor.device)  # (3,1,kH,kW)
    kH, kW = k.shape[-2], k.shape[-1]
    if padding == "same":
        pad_h = get_same_padding(kH)
        pad_w = get_same_padding(kW)
        pad = (pad_w, pad_w, pad_h, pad_h)  # left, right, top, bottom
        img_padded = F.pad(img_tensor, pad, mode="reflect")
        conv = F.conv2d(img_padded, k, bias=None, stride=1, padding=0, groups=3)
    else:
        conv = F.conv2d(img_tensor, k, bias=None, stride=1, padding=0, groups=3)
    return conv


def max_pool(
    feat: torch.Tensor,
    kernel_size: int = 2,
    stride: int = 2
) -> torch.Tensor:
    """
    Apply max pooling.

    Args:
        feat: Tensor (N, C, H, W).
        kernel_size: Pool window.
        stride: Pool stride.

    Returns:
        Pooled tensor.
    """
    pool = nn.MaxPool2d(kernel_size, stride)
    return pool(feat)


def to_numpy_image(t: torch.Tensor) -> np.ndarray:
    """
    Convert a (1, C, H, W) OR (C, H, W) torch tensor in [0,1] or arbitrary range
    into a displayable HxWxC numpy image with value clipping.

    - If range is not [0,1], it will be min-max normalized per-channel for visualization.

    Returns:
        np array (H, W, C) in float32 clipped to [0,1].
    """
    if t.dim() == 4:
        t = t[0]
    # (C,H,W) -> (H,W,C)
    img = t.detach().cpu().float().clone()
    # Per-channel normalization to improve visualization (esp. for edge maps)
    C, H, W = img.shape
    img = img.view(C, -1)
    min_val = img.min(dim=1, keepdim=True).values
    max_val = img.max(dim=1, keepdim=True).values
    # Avoid div-by-zero
    scale = (max_val - min_val).clamp(min=1e-8)
    img = (img - min_val) / scale
    img = img.view(C, H, W).permute(1, 2, 0).numpy()  # (H,W,C)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img


def show_side_by_side(
    title_left: str,
    img_left: np.ndarray,
    title_right: str,
    img_right: np.ndarray,
    figsize: Tuple[int, int] = (16, 6)
):
    """
    Display two images side-by-side with titles.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Convolution and Pooling Outputs", fontsize=14)
    ax1.imshow(img_left)
    ax1.set_title(title_left)
    ax1.axis("off")
    ax2.imshow(img_right)
    ax2.set_title(title_right)
    ax2.axis("off")
    plt.tight_layout()
    plt.show()


def apply_kernel_and_show(
    img_tensor: torch.Tensor,
    kernel_2d: List[List[float]],
    kernel_name: str,
    padding: Literal["same", "valid"] = "same",
    pool_kernel: int = 2,
    pool_stride: int = 2
):
    """
    Apply a 2D kernel (depthwise) + pooling, visualize both.
    """
    conv_tensor = convolve(img_tensor, kernel_2d, padding=padding)
    pool_tensor = max_pool(conv_tensor, kernel_size=pool_kernel, stride=pool_stride)

    conv_img = to_numpy_image(conv_tensor)
    pool_img = to_numpy_image(pool_tensor)

    show_side_by_side(
        title_left=f"{kernel_name} — Convolution",
        img_left=conv_img,
        title_right=f"{kernel_name} — MaxPool {pool_kernel}x{pool_kernel}",
        img_right=pool_img
    )


def main():
    print_env()

    # --- Settings ---
    image_path = "car.jpg"  # change to your image path
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Load
    img_tensor = load_image_as_tensor(image_path=image_path, resize_to=450, device=device)

    # For reference: show original image
    orig_img = to_numpy_image(img_tensor)
    plt.figure(figsize=(8, 6))
    plt.imshow(orig_img)
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

    # --- Define common kernels (2D) ---
    # Sharpen
    sharpen = [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ]

    # Vertical edges (Sobel-like)
    sobel_v = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]

    # Horizontal edges (Sobel-like)
    sobel_h = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ]

    # Simple vertical edge
    prewitt_v = [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ]

    # Simple horizontal edge
    prewitt_h = [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1],
    ]

    # Gaussian blur (3x3)
    gaussian_3x3 = [
        [1/16, 1/8, 1/16],
        [1/8,  1/4, 1/8 ],
        [1/16, 1/8, 1/16],
    ]

    # Emboss
    emboss = [
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2],
    ]

    # --- Apply and visualize ---
    apply_kernel_and_show(img_tensor, sharpen,     "Sharpen",      padding="same")
    apply_kernel_and_show(img_tensor, prewitt_v,   "Vertical Edge (Prewitt)", padding="same")
    apply_kernel_and_show(img_tensor, prewitt_h,   "Horizontal Edge (Prewitt)", padding="same")
    apply_kernel_and_show(img_tensor, sobel_v,     "Vertical Edge (Sobel)", padding="same")
    apply_kernel_and_show(img_tensor, sobel_h,     "Horizontal Edge (Sobel)", padding="same")
    apply_kernel_and_show(img_tensor, gaussian_3x3,"Gaussian Blur 3x3", padding="same")
    apply_kernel_and_show(img_tensor, emboss,      "Emboss",       padding="same")

    # --- Optional: Save one sample output to disk ---
    # Example: save the sharpened convolution result
    conv_sharp = convolve(img_tensor, sharpen, padding="same")
    img_sharp = (to_numpy_image(conv_sharp) * 255.0).astype(np.uint8)
    Image.fromarray(img_sharp).save("car_sharpened.png")
    print("Saved: car_sharpened.png")


# --- Extra Scenario (function) ---
def detect_document_edges(
    image_path: str,
    resize_to: Optional[int] = 800,
    device: Optional[torch.device] = None
):
    """
    A different scenario: document preprocessing (edge emphasis) to aid OCR or page detection.

    Steps:
    1) Load a document/photo as RGB.
    2) Apply Sobel vertical & horizontal to emphasize page/paragraph edges.
    3) Visualize results, useful for later thresholding, contour detection, or Hough transforms.
    """
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    img_tensor = load_image_as_tensor(image_path, resize_to=resize_to, device=device)

    sobel_v = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]
    sobel_h = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ]

    conv_v = convolve(img_tensor, sobel_v, padding="same")
    conv_h = convolve(img_tensor, sobel_h, padding="same")

    img_v = to_numpy_image(conv_v)
    img_h = to_numpy_image(conv_h)

    show_side_by_side("Document — Sobel Vertical", img_v, "Document — Sobel Horizontal", img_h)


if __name__ == "__main__":
    main()
    # Example for scenario 2:
    # detect_document_edges("receipt.jpg", resize_to=1000)
