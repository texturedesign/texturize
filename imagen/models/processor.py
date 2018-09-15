# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch.nn


class Processor(torch.nn.Module):
    """Manipulate and transform images in an intermediate representation, for example
    upscaling or downscaling operations.
    """

    def __init__(self, pretrained=True):
        super(Processor, self).__init__()

        self.downscale = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(64, 48, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.SELU(),

            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(48, 48, kernel_size=(4, 4), stride=(2, 2)),
            torch.nn.SELU(),

            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.SELU(),
        )

        self.upscale = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(64, 48, kernel_size=(3, 3), stride=(1, 1)),

            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(48, 48*4, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.SELU(),

            torch.nn.PixelShuffle(2),

            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.SELU(),
        )

        if pretrained is True:
            self.load_state_dict(torch.load('data/processor.model'))
