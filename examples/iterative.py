# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import argparse

import torch
import torch.nn.functional as F

from imagen import optim
from imagen.data import images
from imagen.models import classifiers


class StyleTransfer(optim.ImageOptimizer):

    def __init__(self, args):
        self.device = torch.device(args.device)
        self.model = classifiers.VGG19().to(self.device)

        content_img = images.load_from_file(args.content, self.device)
        style_img = images.load_from_file(args.style, self.device)
        seed_img = torch.empty_like(content_img).normal_().mul_(0.5).clamp_(-2.0, +2.0)

        super().__init__(seed_img, evaluations=args.iterations)

        self.config = args
        self.all_layers = self.config.content_layers | self.config.style_layers

        self.content_feat = {k: (v - 1.0).detach() for k, v in self.model.extract(content_img, layers=self.config.content_layers)}
        self.style_feat = {k: self.model.gram_matrix(v - 1.0).detach() for k, v in self.model.extract(style_img, layers=self.config.style_layers)}

    def evaluate(self, image):
        style_score = 0.0
        content_score = 0.0

        for i, l in self.model.extract(image, layers=self.all_layers):
            if i in self.config.content_layers:
                content_score += F.mse_loss(self.content_feat[i], l - 1.0)
            if i in self.config.style_layers:
                style_score += F.mse_loss(self.style_feat[i], self.model.gram_matrix(l - 1.0))

        style_score *= self.config.style_weight / len(self.style_feat)
        content_score *= self.config.content_weight

        if self.should_do(self.config.save_every) == 0:
            images.save_to_file(self.image, 'output/test%03i.png' % self.counter)
        if self.should_do(self.config.print_every) == 0:
            print('Iteration: {}    Style Loss: {:4f}     Content Loss: {:4f}'.format(
                self.counter, style_score.item(), content_score.item()))

        return style_score + content_score

    def should_do(self, every):
        return (every != 0) and (self.counter % every == 0)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(prog='imagen')
    add_arg = parser.add_argument
    add_arg('content', type=str, help='Image to use as reference.')
    add_arg('--with', dest='style', type=str, required=True, help='Image for inspiration.')
    add_arg('--iterations', type=int, default=250, help='Number of iterations.')
    add_arg('--device', type=str, default=device, help='Where to perform the computation.')
    add_arg('--style-layers', type=set, default={1, 6, 11, 20})
    add_arg('--style-weight', type=float, default=100000.0)
    add_arg('--content-layers', type=set, default={20})
    add_arg('--content-weight', type=float, default=1.0)
    add_arg('--save-every', type=int, default=0)
    add_arg('--print-every', type=int, default=10)
    args = parser.parse_args()

    optimizer = StyleTransfer(args)
    output = optimizer.run()


if __name__ == '__main__':
    import sys
    main(sys.argv)
