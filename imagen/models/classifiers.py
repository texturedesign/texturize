# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import collections

import torch.nn

from . import download_to_file
from .vgg import ConvLayer, NormLayer, VGGEncoder


class VGG19Encoder(VGGEncoder):
    def __init__(self, pooling="average"):
        """Loads the pre-trained VGG19 convolution layers from the PyTorch vision module.
        """
        super(VGG19Encoder, self).__init__()

        self.features = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("0_0", NormLayer((1, 3, 1, 1), direction="encode")),
                    ("1_1", ConvLayer(3, 64)),
                    ("1_2", ConvLayer(64, 64)),
                    ("2_1", ConvLayer(64, 128, scale="average")),
                    ("2_2", ConvLayer(128, 128)),
                    ("3_1", ConvLayer(128, 256, scale="average")),
                    ("3_2", ConvLayer(256, 256)),
                    ("3_3", ConvLayer(256, 256)),
                    ("3_4", ConvLayer(256, 256)),
                    ("4_1", ConvLayer(256, 512, scale="average")),
                    ("4_2", ConvLayer(512, 512)),
                    ("4_3", ConvLayer(512, 512)),
                    ("4_4", ConvLayer(512, 512)),
                    ("5_1", ConvLayer(512, 512, scale="average")),
                    ("5_2", ConvLayer(512, 512)),
                    ("5_3", ConvLayer(512, 512)),
                    ("5_4", ConvLayer(512, 512)),
                ]
            )
        )

        filename = download_to_file("vgg19_enc", "6cbccfc92ca1be3c4ac96d7da2df3dcf")
        self.load_state_dict(torch.load(filename))


class VGG19Decoder(torch.nn.Module):
    def __init__(self):
        """Loads the pre-trained VGG19 convolution layers from the PyTorch vision module.
        """
        super(VGG19Decoder, self).__init__()

        self.features = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("5_3", ConvLayer(512, 512, activation="ReLU")),
                    ("5_2", ConvLayer(512, 512, activation="ReLU")),
                    ("5_1", ConvLayer(512, 512, activation="ReLU")),
                    ("4_4", ConvLayer(512, 512, scale="reshuffle", activation="ReLU")),
                    ("4_3", ConvLayer(512, 512, activation="ReLU")),
                    ("4_2", ConvLayer(512, 512, activation="ReLU")),
                    ("4_1", ConvLayer(512, 512, activation="ReLU")),
                    ("3_4", ConvLayer(512, 256, scale="reshuffle", activation="ReLU")),
                    ("3_3", ConvLayer(256, 256, activation="ReLU")),
                    ("3_2", ConvLayer(256, 256, activation="ReLU")),
                    ("3_1", ConvLayer(256, 256, activation="ReLU")),
                    ("2_2", ConvLayer(256, 128, scale="reshuffle", activation="ReLU")),
                    ("2_1", ConvLayer(128, 128, activation="ReLU")),
                    ("1_2", ConvLayer(128, 64, scale="reshuffle", activation="ReLU")),
                    ("1_1", ConvLayer(64, 64, activation="ReLU")),
                    ("1_0", ConvLayer(64, 3, activation=None)),
                    ("0_0", NormLayer((1, 3, 1, 1), direction="decode")),
                ]
            )
        )

    def rebuild(self, data, layers: set, start: str):
        """Convert features extracted from the encoder and turn them into an image.
        """
        if len(layers) == 0:
            return

        names = list(self.features._modules.keys())
        indices = [names.index(l) for l in layers]

        for i in range(names.index(start), max(indices) + 1):
            data = self.features[i].forward(data)
            if i in indices:
                yield names[i], data
