from PIL import Image

import torch.nn.functional as F
import torchvision.transforms as T


def base_transformer(
        img: Image
    ):
    tnsr = T.ToTensor()(img).unsqueeze(0)
    tnsr = F.interpolate(
        tnsr,
        size=(299, 299),
        mode='bilinear',
        align_corners=False
    )
    return tnsr.squeeze()


def clean_fid_transformer(
        img: Image
    ):
    img = img.resize((299, 299), resample=Image.BICUBIC)
    img = T.PILToTensor()(img)
    return img.float() / 255
