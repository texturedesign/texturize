# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch.nn
import torchvision


class Translator(torch.nn.Module):
    """Convert RGB images into an interediate representation suitable for further processing.
    """

    def __init__(self, pretrained=True):
        super(Translator, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.SELU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.SELU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(64, 48, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.SELU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(48, 32, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.SELU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1)),
            torch.nn.SELU(),
        )

        if pretrained:
            self.load_state_dict(torch.load("data/translator.model"))

    def encode(self, image):
        """Translate an input RGB image into an intermediate representation of the same size,
        but different number of channels.
        """

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]
                ),
            ]
        )

        data = transform(image)[None]
        return self.encoder(data)

    def decode(self, latent):
        """Translate an intermediate representation of an image back into RGB.
        """

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Lambda(lambda y: y.clamp_(-2.0, +2.0)),
                torchvision.transforms.Normalize(
                    mean=[-2.0, -2.0, -2.0], std=[4.0, 4.0, 4.0]
                ),
                torchvision.transforms.ToPILImage(),
            ]
        )

        data = self.decoder(latent)
        return transform(data[0])
