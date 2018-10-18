# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch.optim


class ImageOptimizer:

    def __init__(self, image, evaluations):
        self.evaluations = evaluations
        self.counter = 0
        self.image = image.detach_().requires_grad_()
        self.lbfgs = torch.optim.LBFGS([self.image])

    def run(self):
        while self.counter <= self.evaluations:
            self.lbfgs.step(self.tick)

        self.image.data.clamp_(-2.0, +2.0)
        return self.image

    def tick(self):
        self.lbfgs.zero_grad()
        self.image.data.clamp_(-2.0, +2.0)
        
        loss = self.evaluate(self.image)
        loss.backward()

        self.counter += 1
        return loss

    def evaluate(self, image):
        raise NotImplementedError
