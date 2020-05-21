#!/usr/bin/env python3
r"""                         _   _            _              _         
  _ __   ___ _   _ _ __ __ _| | | |_ _____  _| |_ _   _ _ __(_)_______ 
 | '_ \ / _ \ | | | '__/ _` | | | __/ _ \ \/ / __| | | | '__| |_  / _ \
 | | | |  __/ |_| | | | (_| | | | ||  __/>  <| |_| |_| | |  | |/ /  __/
 |_| |_|\___|\__,_|_|  \__,_|_|  \__\___/_/\_\\__|\__,_|_|  |_/___\___|

Usage:
    texturize SOURCE [-s WxH] [-o FILE]
                     [--scales=S] [--precision=P] [--iterations=I]
    texturize --help

Examples:
    texturize samples/grass.webp --size=1440x960 --output=result.png
    texturize samples/gravel.png --iterations=200 --precision=1e-6
    texturize samples/sand.tiff  --output=tmp/{source}-{scale}.webp

Options:
    SOURCE                  Path to source image to use as texture.
    -s WxH, --size=WxH      Output resolution as WIDTHxHEIGHT. [default: 640x480]
    --scales=S              Number of scales to process. [default: 5]
    --precision=P           Set the quality for the optimization. [default: 1e-5]
    --iterations=I          Maximum number of iterations each octave. [default: 99]
    -o FILE, --output=FILE  Filename for saving the result. [default: {source}_gen.png]
    -h --help               Show this message.

"""
#
# Copyright (c) 2020, Novelty Factory KG.
#
# neural-texturize is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License version 3. This program is distributed
# in the hope that it will be useful but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#

import os
import itertools

import docopt
import progressbar

import torch
import torch.nn.functional as F

from encoders import models

from . import __version__
from . import io


class GramMatrixCritic:
    """A `Critic` evaluates the features of an image to determine how it scores.

    This is a gram-based critic that computes a 2D histogram of feature cross-correlations for
    a specific layer, and compares it to the target gram matrix.
    """

    def __init__(self, layer, offset: float = -1.0):
        self.layer = layer
        self.offset = offset
        self.gram = None

    def evaluate(self, features):
        current = self._prepare_gram(features)
        yield 1e4 * F.mse_loss(current, self.gram, reduction="mean")

    def from_features(self, features):
        self.gram = self._prepare_gram(features)

    def get_layers(self):
        return {self.layer}

    def _gram_matrix(self, column, row):
        (b, ch, h, w) = column.size()
        f_c = column.view(b, ch, w * h)
        (b, ch, h, w) = row.size()
        f_r = row.view(b, ch, w * h)

        gram = (f_c / w).bmm((f_r / h).transpose(1, 2)) / ch
        assert not torch.isnan(gram).any()

        return gram

    def _prepare_gram(self, features):
        f = features[self.layer] + self.offset
        return self._gram_matrix(f, f)


def get_all_layers(critics):
    """Determine the minimal list of layer features that needs to be extracted from the image.
    """
    layers = set(itertools.chain.from_iterable(c.get_layers() for c in critics))
    return sorted(list(layers))


class SolverLBFGS:
    """Encapsulate the L-BFGS optimizer from PyTorch with a standard interface.
    """

    def __init__(self, objective, image, lr=1.0):
        self.objective = objective
        self.image = image
        self.optimizer = torch.optim.LBFGS(
            [image], lr=lr, max_iter=2, max_eval=4, history_size=10
        )
        self.scores = []

    def step(self):
        def _wrap():
            self.optimizer.zero_grad()
            return self.objective(self.image)

        return self.optimizer.step(_wrap)


class MultiCriticObjective:
    """An `Objective` that defines a problem by evaluating possible solutions (i.e. images).

    This objective evaluates a list of critics to produce a final "loss" that's the sum of all the
    scores returned by the critics.  It's also responsible for computing the gradients.
    """

    def __init__(self, encoder, critics):
        self.encoder = encoder
        self.critics = critics

    def __call__(self, image):
        """Main evaluation function that's called by the solver.  Processes the image, computes the
        gradients, and returns the loss.
        """

        image.data.clamp_(0.0, 1.0)

        # Extract features from image.
        feats = dict(self.encoder.extract(image, get_all_layers(self.critics)))

        # Apply all the critics one by one.
        scores = []
        for critic in self.critics:
            total = 0.0
            for loss in critic.evaluate(feats):
                total += loss
            scores.append(total)

        # Calculate the final loss and compute the gradients.
        loss = sum(scores) / len(scores)
        loss.backward()

        return loss


class TextureSynthesizer:
    def __init__(self, encoder, lr, precision, max_iter):
        self.encoder = encoder
        self.lr = lr
        self.precision = precision
        self.max_iter = max_iter

    def prepare(self, critics, image):
        """Extract the features from the source texture and initialize the critics.
        """
        feats = dict(self.encoder.extract(image, get_all_layers(critics)))
        for critic in critics:
            critic.from_features(feats)

    def run(self, seed_img, critics):
        """Run the optimizer on the image according to the loss returned by the critics.
        """
        image = seed_img.to("cuda").requires_grad_(True)

        obj = MultiCriticObjective(self.encoder, critics)
        opt = SolverLBFGS(obj, image, lr=self.lr)

        widgets = [
            progressbar.SimpleProgress(),
            " | ",
            progressbar.Variable("loss", format="{name}: {value:0.3e}"),
            " ",
            progressbar.Bar(marker="■", fill="·"),
            " ",
            progressbar.ETA(),
        ]
        progress = progressbar.ProgressBar(
            max_value=self.max_iter, widgets=widgets, variables={"loss": float("+inf")}
        )

        previous = None
        for i in range(self.max_iter):
            # Perform one step of the optimization.
            loss = opt.step()
            # Update the progress bar with the result!
            progress.update(i, loss=loss)

            with torch.no_grad():
                # Constrain the image to the valid color range.
                image.data.clamp_(0.0, 1.0)

                # Return this iteration to the caller...
                yield loss.item(), image

                # See if we can terminate the optimization early.
                if previous is not None and abs(loss - previous) < self.precision:
                    progress.max_value = i
                    break
                previous = loss

        progress.finish()


class ansi:
    WHITE = "\033[1;97m"
    BLACK = "\033[0;30m\033[47m"
    PINK = "\033[1;35m"
    ENDC = "\033[0m\033[49m"


def run(config):
    # Load the original image.
    texture_img = io.load_image_from_file(config["SOURCE"], device="cuda")

    # Configure the critics.
    critics = [GramMatrixCritic(layer=l) for l in ("1_1", "2_1", "3_1")]

    # Encoder used by all the critics.
    encoder = models.VGG11(pretrained=True, pool_type=torch.nn.AvgPool2d)
    encoder = encoder.to("cuda", dtype=torch.float32)

    # Generate the starting image for the optimization.
    scales = int(config["--scales"])
    result_sz = list(map(int, config["--size"].split("x")))[::-1]
    result_img = torch.empty(
        (1, 3, result_sz[0] // 2 ** (scales + 1), result_sz[1] // 2 ** (scales + 1)),
        device="cuda",
        dtype=torch.float32,
    ).uniform_(0.4, 0.6)

    for i, scale in enumerate(2 ** s for s in range(scales, -1, -1)):
        # Each octave we start a new optimization process.
        synth = TextureSynthesizer(
            encoder,
            lr=0.5,
            precision=float(config["--precision"]),
            max_iter=int(config["--iterations"]),
        )
        print(ansi.BLACK + "\n OCTAVE", f"#{i} " + ansi.ENDC)
        print("<- scale:", f"1/{scale}")

        # Create downscaled version of original texture to match this octave.
        texture_cur = F.interpolate(
            texture_img,
            scale_factor=1.0 / scale,
            mode="area",
            recompute_scale_factor=False,
        )
        synth.prepare(critics, texture_cur)
        print("<- texture:", tuple(texture_cur.shape[2:]))

        # Compute the seed image for this octave, sprinkling a bit of gaussian noise.
        size = result_sz[0] // scale, result_sz[1] // scale
        seed_img = F.interpolate(result_img, size, mode="bicubic", align_corners=False)
        seed_img += torch.empty_like(seed_img, dtype=torch.float32).normal_(std=0.1)
        print("<- seed:", tuple(seed_img.shape[2:]), end="\n\n")
        del result_img

        # Now we can enable the automatic gradient computation to run the optimization.
        with torch.enable_grad():
            for _, result_img in synth.run(seed_img, critics):
                pass
        del synth

        # Save the files for each octave to disk.
        result_img = result_img.cpu()
        filename = config["--output"].format(
            scale=scale, source=os.path.splitext(os.path.basename(config["SOURCE"]))[0]
        )
        io.save_image_to_file(
            F.interpolate(result_img, size=result_sz, mode="nearest"), filename
        )
        print("\n=> output:", filename)


def main():
    print(ansi.PINK + "    " + __doc__[:356] + ansi.ENDC)

    config = docopt.docopt(__doc__[356:], version=__version__)
    with torch.no_grad():
        run(config)


if __name__ == "__main__":
    main()
