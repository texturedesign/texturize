# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch.optim
import torch.nn.functional as F


class ImageOptimizer:

    def __init__(self, image, evaluations):
        self.evaluations = evaluations
        self.counter = 0
        self.image = image.detach().requires_grad_()
        self.lbfgs = torch.optim.LBFGS([self.image])

    def run(self):
        while self.counter <= self.evaluations:
            self.lbfgs.step(self.tick)

        self.image.data.clamp_(-2.0, +2.0)
        return self.image

    def tick(self):
        self.lbfgs.zero_grad()
        
        image = self.image.clamp(-2.0, +2.0)
        loss = self.evaluate(image)
        loss.backward()

        out_of_range_loss = F.mse_loss(image, self.image)
        self.image.data.clamp_(-2.0, +2.0)

        self.counter += 1
        return loss + out_of_range_loss

    def evaluate(self, image):
        raise NotImplementedError
