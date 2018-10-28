# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch
import torch.nn


class DownscaleBuilder:

    def __init__(self, factor, channels=3):
        self.downscaler = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, groups=channels,
                            kernel_size=(factor, factor), stride=(factor, factor))
        )

        self.downscaler[0].bias[:] = 0.0
        self.downscaler[0].weight[:,0,:,:] = 1.0 / (factor ** 2)

    def build(self, image):
        return self.downscaler(image)


class UpscaleBuilder:

    def __init__(self, factor, channels=3, mode='nearest'):
        self.upscaler = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=(factor, factor), mode=mode) # , align_corners=True)
        )

    def build(self, image):
        return self.upscaler(image)
