# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch.nn


class OctaveBuilder:
    def __init__(self, levels, kernel=5, sigma=2.0):
        self.gaussian_blur = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(kernel // 2),
            torch.nn.Conv2d(3, 3, groups=3, kernel_size=(1, kernel), stride=(1, 1)),
            torch.nn.Conv2d(3, 3, groups=3, kernel_size=(kernel, 1), stride=(1, 1)),
        )

        self.gaussian_blur[1].bias[:] = 0.0
        self.gaussian_blur[2].bias[:] = 0.0

        self.gaussian_blur[1].weight[:, 0, 0, :] = self.make_kernel(kernel, sigma=2.0)
        self.gaussian_blur[2].weight[:, 0, :, 0] = self.make_kernel(kernel, sigma=2.0)

        self.levels = levels

    def make_kernel(self, size, sigma):
        x = torch.arange(-size // 2 + 1.0, size // 2 + 1.0)
        kernel = torch.exp(-x ** 2 / (2.0 * sigma ** 2))
        return kernel / torch.sum(kernel)

    def build(self, image):
        result = [image]
        for i in range(self.levels - 1):
            with torch.no_grad():
                for j in range(2 ** (i + 1)):
                    image = self.gaussian_blur(image)
            result.append(image)
        return torch.cat(result, dim=1)
