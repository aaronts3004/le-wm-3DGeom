
import torchvision.transforms as T
import torch
import torch.nn as nn
from utils import get_img_preprocessor

# OGBench point-map statistics (computed on the training split)
POINT_MEAN = torch.tensor([
    0.21534102,   # X
   -0.02312508,   # Y
    0.04208415,   # Z
], dtype=torch.float32)

POINT_STD = torch.tensor([
    0.43103240,   # X
    0.26829550,   # Y
    0.10406391,   # Z
], dtype=torch.float32)



def depth_img_preprocessor(img_size, repeat_channels=True):

    resize = T.Resize(
        (img_size, img_size),
        interpolation=T.InterpolationMode.BILINEAR,
    )

    def transform(steps):
        # [T, 1, H, W]
        depth = steps["pixels"].float()

        # Remove NaNs/Infs
        depth = torch.nan_to_num(depth, nan=0.5, posinf=3.0, neginf=0.5)

        # Clip to rendering range
        depth = depth.clamp(0.5, 3.0)

        # Resize
        depth = resize(depth)

        # Normalize to [-1, 1]
        depth = (depth - 0.5) / (3.0 - 0.5)
        depth = depth * 2.0 - 1.0

        # Repeat to RGB channels
        if repeat_channels:
            depth = depth.repeat(1, 3, 1, 1)

        steps["pixels"] = depth
        
        return steps

    return transform

def normal_img_preprocessor(img_size):

    resize = T.Resize(
        (img_size, img_size),
        interpolation=T.InterpolationMode.BILINEAR,
    )

    def transform(steps):

        normals = steps["pixels"].float()

        # uint8 -> [-1,1]
        normals = normals / 127.5 - 1.0
        normals = resize(normals)

        # Optional but recommended
        normals = torch.nn.functional.normalize(
            normals,
            dim=1,
            eps=1e-6,
        )

        steps["pixels"] = normals
        return steps

    return transform



def rgbd_img_preprocessor(img_size):

    rgb_transform = get_img_preprocessor(
        source="pixels",
        target="pixels",
        img_size=img_size,
    )
    depth_transform = depth_img_preprocessor(img_size, repeat_channels=False)

    def transform(steps):

        rgb_steps = {"pixels": steps["pixels"][:, :3]}
        rgb_steps = rgb_transform(rgb_steps)

        depth_steps = {"pixels": steps["pixels"][:, 3:]}
        depth_steps = depth_transform(depth_steps)

        steps["pixels"] = torch.cat(
            [rgb_steps["pixels"], depth_steps["pixels"]],
            dim=1,
        )

        return steps

    return transform

def da3_img_preprocessor(img_size, repeat_channels=True):

    DA3_DEPTH_MIN = 0.5755
    DA3_DEPTH_MAX = 1.5811

    resize = T.Resize(
        (img_size, img_size),
        interpolation=T.InterpolationMode.BILINEAR,
    )

    def transform(steps):
        # [T, 1, H, W]
        depth = steps["pixels"].float()
        depth = torch.clamp(depth, DA3_DEPTH_MIN, DA3_DEPTH_MAX)
        depth = (depth - DA3_DEPTH_MIN) / (DA3_DEPTH_MAX - DA3_DEPTH_MIN)

        # Resize
        depth = resize(depth)

        # Repeat to RGB channels
        if repeat_channels:
            depth = depth.repeat(1, 3, 1, 1)

        steps["pixels"] = depth
        return steps

    return transform


def point_map_preprocessor(img_size):
    resize = T.Resize(
        (img_size, img_size),
        interpolation=T.InterpolationMode.BILINEAR,
    )

    def transform(steps):

        # (B,3,H,W)
        points = steps["pixels"].float()

        # print("points.shape=", points.shape)

        # -> (B,3,H,W)
        # points = points.permute(0, 3, 1, 2)

        mean = POINT_MEAN.to(points.device).view(1, 3, 1, 1)
        std = POINT_STD.to(points.device).view(1, 3, 1, 1)

        points = (points - mean) / std
        points = resize(points)
        steps["pixels"] = points

        return steps

    return transform
