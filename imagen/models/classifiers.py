# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch.nn
import torchvision


class VGG19(torch.nn.Module):

    def __init__(self):
        """Loads the pre-trained VGG19 convolution layers from the PyTorch vision module.
        """
        super(VGG19, self).__init__()

        vgg19 = torchvision.models.vgg19(pretrained=False)
        self.features = vgg19.features

        self.load_state_dict(torch.load('data/vgg19_torch.pth'))

        for i, f in enumerate(self.features):
            if isinstance(f, torch.nn.MaxPool2d):
                self.features[i] = torch.nn.AvgPool2d(f.kernel_size, f.stride)

        self.stmean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.stddev = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def extract(self, image, layers : set): # {1, 6, 11, 20, 29}:
        """Preprocess an image to be compatible with pre-trained model, and return required features.
        """
        if len(layers) == 0:
            return

        image = ((image * 0.25 + 0.5) - self.stmean) / self.stddev
        for i in range(max(layers)+1):
            image = self.features[i].forward(image)
            if i in layers:
                yield i, image
