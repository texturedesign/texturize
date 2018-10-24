# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import argparse

import torch
import torch.nn.functional as F

from imagen import optim
from imagen.data import images, resize, histogram
from imagen.models import classifiers


class StyleTransfer(optim.ImageOptimizer):

    def __init__(self, args):
        super(StyleTransfer, self).__init__()

        if args.random_seed is not None:
            torch.manual_seed(args.random_seed)

        self.device = torch.device(args.device)
        self.model = classifiers.VGG19().to(self.device)

        if args.content is not None:
            self.content_img = images.load_from_file(args.content, self.device)
        else:
            h, w = map(int, args.content_size.split('x'))
            self.content_img = torch.empty((1, 3, h, w), device=self.device)

        self.style_img = images.load_from_file(args.style, self.device)
        self.style_feat = {}
        self.style_hist = {}
        self.seed_img = None

        self.config = args
        self.config.style_weights = [w * self.config.style_multiplier for w in self.config.style_weights]
        self.all_layers = set(self.config.content_layers) | set(self.config.style_layers) | set(self.config.histogram_layers)

    def evaluate(self, image):
        style_score = torch.tensor(0.0)
        hist_score = torch.tensor(0.0)
        content_score = torch.tensor(0.0)

        cw = iter(self.config.content_weights)
        sw = iter(self.config.style_weights)
        hw = iter(self.config.histogram_weights)

        for i, l in self.model.extract(image, layers=self.all_layers):
            if i in self.config.content_layers:
                content_score += F.mse_loss(self.content_feat[i], l - 1.0) * next(cw)
            if i in self.config.style_layers:
                style_score += F.mse_loss(self.style_feat[i], self.model.gram_matrix(l - 1.0)) * next(sw)
            if i in self.config.histogram_layers:
                tl = histogram.match_histograms(l, self.style_hist[i], same_range=True)
                hist_score += F.mse_loss(tl, l) * next(hw)

        if self.should_do(self.config.save_every):
            images.save_to_file(self.image.clone().detach().cpu(), 'output/test%04i.png' % (self.scale * 1000 + self.counter))
        if self.should_do(self.config.print_every):
            print('Iteration: {}    Style Loss: {:4f}     Content Loss: {:4f}    Histogram Loss: {:4f}'.format(
                self.counter, style_score.item(), content_score.item(), hist_score.item()))

        return content_score + hist_score + style_score

    def should_do(self, every):
        return (every != 0) and (self.counter % every == 0)

    def run(self):
        for self.scale in range(0, self.config.scales):
            factor = 2 ** (self.config.scales - self.scale - 1)
            content_img = resize.DownscaleBuilder(factor).build(self.content_img)
            style_img = resize.DownscaleBuilder(factor).build(self.style_img)

            if self.seed_img is None:
                seed_img = torch.empty_like(content_img).normal_(std=0.5).clamp_(-2.0, +2.0)
            else:
                seed_img = (resize.DownscaleBuilder(factor).build(self.seed_img)
                           + torch.empty_like(content_img).normal_(std=0.1)).clamp_(-2.0, +2.0)

            self.content_feat = {k: (v - 1.0).detach() for k, v in self.model.extract(content_img, layers=self.config.content_layers)}
            self.style_feat, self.style_hist = {}, {}
            for k, v in self.model.extract(style_img, layers=self.config.style_layers):
                self.style_feat[k] = self.model.gram_matrix(v - 1.0).detach()
            for k, v in self.model.extract(style_img, layers=self.config.histogram_layers):
                self.style_hist[k] = histogram.extract_histograms(v, bins=5, min=torch.tensor(-1.0+1e-3), max=torch.tensor(+4.0))

            output = self.optimize(seed_img, self.config.iterations, lr=1.0 if self.scale == 0 else 0.2)

            self.seed_img = resize.UpscaleBuilder(factor, mode='bilinear').build(output)

        return output


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(prog='imagen')
    add_arg = parser.add_argument
    add_arg('--content', type=str, default=None, help='Image to use as reference.')
    add_arg('--content-size', type=str, default=None)
    add_arg('--with', dest='style', type=str, required=True, help='Image for inspiration.')
    add_arg('--iterations', type=int, default=250, help='Number of iterations.')
    add_arg('--scales', type=int, default=3, help='Total number of scales.')
    add_arg('--random-seed', type=int, default=None, help='Seed for random numbers.')
    add_arg('--device', type=str, default=device, help='Where to perform the computation.')
    add_arg('--style-layers', type=int, nargs='*', default=[1, 6, 11])
    add_arg('--style-weights', type=float, nargs='*', default=[1.0, 1.0, 1.0])
    add_arg('--style-multiplier', type=float, default=1e+6)
    add_arg('--histogram-layers', type=int, nargs='*', default=[0, 5])
    add_arg('--histogram-weights', type=float, nargs='*', default=[1.0, 1.0])
    add_arg('--content-layers', type=int, nargs='*', default=[11])
    add_arg('--content-weights', type=float, nargs='*', default=[1.0])
    add_arg('--save-every', type=int, default=0)
    add_arg('--print-every', type=int, default=10)
    args = parser.parse_args()

    optimizer = StyleTransfer(args)
    output = optimizer.run()

if __name__ == '__main__':
    import sys
    main(sys.argv)
