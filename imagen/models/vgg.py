# Neural Imagen — Copyright (c) 2019, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch.nn


class ConvLayer(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, scale=None, activation="ReLU"):
        activation = getattr(torch.nn, activation)
        layers = [
            torch.nn.ReflectionPad2d(padding=1),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3)),
            activation(inplace=True),
        ]

        if scale == "average":
            layers.insert(0, torch.nn.AvgPool2d(kernel_size=(2, 2)))
        if scale == "maximum":
            layers.insert(0, torch.nn.MaxPool2d(kernel_size=(2, 2)))
        if scale == "reshuffle":
            layers.append(torch.nn.PixelShuffle(2))

        super(ConvLayer, self).__init__(*layers)


class NormLayer(torch.nn.Module):
    def __init__(self, size, direction="encode"):
        super(NormLayer, self).__init__()
        self.direction = direction
        self.mean = torch.nn.Parameter(torch.zeros(size), requires_grad=False)
        self.std = torch.nn.Parameter(torch.ones(size), requires_grad=False)

    def forward(self, data):
        if self.direction == "encode":
            return ((data * 0.25 + 0.5) - self.mean) / self.std
        if self.direction == "decode":
            return ((data * self.std + self.mean) - 0.5) / 0.25
        assert False, "Unknown direction specified."


class VGGEncoder(torch.nn.Module):
    def extract(self, image, layers: set, start="0_0"):
        """Preprocess an image to be compatible with pre-trained model, and return required features.
        """
        if len(layers) == 0:
            return

        names = list(self.features._modules.keys())
        indices = [names.index(l) for l in layers]

        for i in range(names.index(start), max(indices) + 1):
            image = self.features[i].forward(image)
            if i in indices:
                yield names[i], image
