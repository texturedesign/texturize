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

        self.stmean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.stddev = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def extract(self, image, layers : set): # {1, 6, 11, 20, 29}:
        """Preprocess an image to be compatible with pre-trained model, and return required features.
        """
        image = ((image * 0.25 + 0.5) - self.stmean) / self.stddev
        for i in range(max(layers)+1):
            image = self.features[i].forward(image)
            if i in layers:
                yield i, image

    def gram_matrix(self, features):
        (b, ch, h, w) = features.size()
        f_i = (features - 1.0).view(b, ch, w * h)
        f_t = f_i.transpose(1, 2)
        return f_i.bmm(f_t) / (ch * h * w)
