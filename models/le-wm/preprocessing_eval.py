import torch
import torch.nn.functional as F
from torchvision import transforms



##### ---------------------------------     NORMALS    -----------------------------------

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2


class NormalMapTransform:
    """
    Convert uint8 normal maps to normalized unit vectors.

    Input:
        uint8 image (C,H,W) in [0,255]

    Output:
        float32 image (C,H,W) with unit normals in [-1,1]
    """

    def __init__(self, img_size, debug=False):
        self.resize = v2.Resize(
            (img_size, img_size),
            interpolation=InterpolationMode.BILINEAR,
        )
        self.debug = debug
        self._printed = False

    def __call__(self, img):

        if self.debug and not self._printed:
            print("\n===== BEFORE =====")
            print(img.shape, img.dtype)
            print("range:", img.min().item(), img.max().item())

        # uint8 -> float
        img = img.float()

        # [0,255] -> [-1,1]
        img = img / 127.5 - 1.0

        # resize
        img = self.resize(img)

        # restore unit-length normals
        img = F.normalize(img, dim=0, eps=1e-6)

        if self.debug and not self._printed:
            lengths = torch.linalg.norm(img, dim=0)

            print("\n===== AFTER =====")
            print(img.shape, img.dtype)
            print("range:", img.min().item(), img.max().item())
            print(
                f"normal lengths: "
                f"mean={lengths.mean():.4f} "
                f"std={lengths.std():.4f} "
                f"min={lengths.min():.4f} "
                f"max={lengths.max():.4f}"
            )
            print("=================\n")

            self._printed = True

        return img


def normal_transform(cfg):
    return v2.Compose(
        [
            v2.ToImage(),
            NormalMapTransform(
                img_size=cfg.eval.img_size,
                debug=True,   # disable after checking
            ),
        ]
    )