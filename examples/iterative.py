# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import os
import argparse

import torch
import torch.nn.functional as F

from imagen import optim
from imagen.data import images, resize, histogram
from imagen.models import classifiers


class StyleTransfer(optim.ImageOptimizer):

    def __init__(self, args):
        """Constructor prepares the model and loads the original images. 
        """
        super(StyleTransfer, self).__init__()

        # Setup state for random number generator for deterministic results.
        if args.seed_random is not None:
            torch.manual_seed(args.seed_random)

        # Load the convolution network from pre-trained parameters.
        self.device = torch.device(args.device)
        self.model = classifiers.VGG19().to(self.device)

        # Load the content image from disk or create an empty tensor.
        if args.content is not None:
            self.content_img = images.load_from_file(args.content, self.device)
        else:
            args.content_weights, args.content_layers = [], []
            h, w = reversed(list(map(int, args.output_size.split('x'))))
            self.content_img = torch.zeros((1, 3, h, w), device=self.device)

        # Load the style image from disk to be processed during optimization.
        if args.style is not None:
            self.style_img = images.load_from_file(args.style, self.device)
        else:
            args.style_weights, args.style_layers = [], []
            self.style_img = None

        self.seed_img = None

        # Preprocess the various loss weights and decide which layers need to be computed.
        self.args = args
        self.args.style_weights = [w * self.args.style_multiplier for w in self.args.style_weights]
        self.all_layers = set(self.args.content_layers) | set(self.args.style_layers) | set(self.args.histogram_layers)

    def evaluate(self, image):
        """Compute the style and content loss for the image specified, used at each step of the optimization.
        """

        # Default scores start at zero, stored as tensors so it's possible to compute gradients
        # even if there are no layers enabled below.
        style_score = torch.tensor(0.0).to(self.device)
        hist_score = torch.tensor(0.0).to(self.device)
        content_score = torch.tensor(0.0).to(self.device)

        # Each layer can have custom weights for style and content loss, stored as Python iterators.
        cw = iter(self.args.content_weights)
        sw = iter(self.args.style_weights)
        hw = iter(self.args.histogram_weights)

        # Ask the model to prepare each layer one by one, then decide which losses to calculate.
        for i, f in self.model.extract(image, layers=self.all_layers):

            # The content loss is a mean squared error directly on the activation features.
            if i in self.args.content_layers:
                content_score += F.mse_loss(self.content_feat[i], f) * next(cw)

            # The style loss is mean squared error on cross-correlation statistics (aka. gram matrix).
            if i in self.args.style_layers:
                gram = histogram.square_matrix(f - 1.0)
                style_score += F.mse_loss(self.style_gram[i], gram) * next(sw)

            # Histogram loss is computed like a content loss, but only after the values have been
            # adjusted to match the target histogram.
            if i in self.args.histogram_layers:
                tl = histogram.match_histograms(f, self.style_hist[i], same_range=True)
                hist_score += F.mse_loss(tl, f) * next(hw)

        # Store the image to disk at the specified intervals.
        if self.should_do(self.args.save_every):
            images.save_to_file(self.image.clone().detach().cpu(), 'output/test%04i.png' % (self.scale * 1000 + self.counter))

        # Print optimization statistics at regular intervals.
        if self.should_do(self.args.print_every):
            print('Iteration: {}    Style Loss: {:4f}     Content Loss: {:4f}    Histogram Loss: {:4f}'.format(
                self.counter, style_score.item(), content_score.item(), hist_score.item()))

        # Total loss is passed back to the optimizer.
        return content_score + hist_score + style_score

    def should_do(self, every):
        return (every != 0) and (self.counter % every == 0)

    def run(self):
        """Main entry point for style transfer, operates coarse-to-fine as specified by the number of scales.
        """

        for self.scale in range(0, self.args.scales):
            # Pre-process the input images so they have the expected size.
            factor = 2 ** (self.args.scales - self.scale - 1)
            content_img = resize.DownscaleBuilder(factor).build(self.content_img)
            style_img = resize.DownscaleBuilder(factor).build(self.style_img)

            # Determine the stating point for the optimizer, was there an output of previous scale?
            if self.seed_img is None:
                # a) Load an image from disk, this needs to be the exact right size.
                if self.args.seed is not None:
                    seed_img = images.load_from_file(self.args.seed, self.device)
                    assert seed_img.shape == content_img.shape

                # b) Use completely random buffer from a normal distribution.
                else:
                    seed_img = torch.empty_like(content_img).normal_(std=0.5).clamp_(-2.0, +2.0)
            else:
                # c) There was a previous scale, so resize and add noise from normal distribution. 
                seed_img = (resize.DownscaleBuilder(factor).build(self.seed_img)
                           + torch.empty_like(content_img).normal_(std=0.1)).clamp_(-2.0, +2.0)

            # Pre-compute the cross-correlation statistics for the style image layers (aka. gram matrices).
            self.style_gram = {}
            for i, f in self.model.extract(style_img, layers=self.args.style_layers):
                self.style_gram[i] = histogram.square_matrix(f - 1.0).detach()

            # Pre-compute feature histograms for the style image layers specified.
            self.style_hist = {}
            for k, v in self.model.extract(style_img, layers=self.args.histogram_layers):
                self.style_hist[k] = histogram.extract_histograms(v, bins=5, min=torch.tensor(-1.0), max=torch.tensor(+4.0))

            # Prepare and store the content image activations for image layers too.
            self.content_feat = {}
            for i, f in self.model.extract(content_img, layers=self.args.content_layers):
                self.content_feat[i] = f.detach()

            # Now run the optimization using L-BFGS starting from the seed image.
            output = self.optimize(seed_img, self.args.iterations, lr=0.2)

            # For the next scale, we'll reuse a biliniear interpolated version of this output.
            self.seed_img = resize.UpscaleBuilder(factor, mode='bilinear').build(output).detach()

        # Save the final image at the finest scale to disk.
        basename = os.path.splitext(os.path.basename(self.args.content or self.args.style))[0]
        images.save_to_file(self.image.clone().detach().cpu(), self.args.output or ('output/%s_final.png' % basename))


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(prog='imagen')
    add_arg = parser.add_argument
    add_arg('--scales', type=int, default=3, help='Total number of scales.')
    add_arg('--iterations', type=int, default=250, help='Number of iterations each scale.')
    add_arg('--device', type=str, default=device, help='Where to perform the computation.')
    add_arg('--content', type=str, default=None, help='Image to use as reference.')
    add_arg('--content-layers', type=int, nargs='*', default=[20])
    add_arg('--content-weights', type=float, nargs='*', default=[1.0])
    add_arg('--output', type=str, default=None, help='Filename for output image.')
    add_arg('--output-size', type=str, default=None)
    add_arg('--seed', type=str, default=None, help='Initial image to use.')
    add_arg('--seed-random', type=int, default=None, help='Seed for random numbers.')
    add_arg('--style', type=str, default=None, help='Image for inspiration.')
    add_arg('--style-layers', type=int, nargs='*', default=[1, 6, 11, 20, 29])
    add_arg('--style-weights', type=float, nargs='*', default=[1.0, 1.0, 1.0, 1.0, 1.0])
    add_arg('--style-multiplier', type=float, default=1e+6)
    add_arg('--histogram-layers', type=int, nargs='*', default=[])
    add_arg('--histogram-weights', type=float, nargs='*', default=[])
    add_arg('--save-every', type=int, default=0)
    add_arg('--print-every', type=int, default=10)
    args = parser.parse_args()

    optimizer = StyleTransfer(args)
    optimizer.run()


if __name__ == '__main__':
    import sys
    import imagen.__main__

    main(sys.argv)
