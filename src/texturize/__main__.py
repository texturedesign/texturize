#!/usr/bin/env python3
r"""                         _   _            _              _         
  _ __   ___ _   _ _ __ __ _| | | |_ _____  _| |_ _   _ _ __(_)_______ 
 | '_ \ / _ \ | | | '__/ _` | | | __/ _ \ \/ / __| | | | '__| |_  / _ \
 | | | |  __/ |_| | | | (_| | | | ||  __/>  <| |_| |_| | |  | |/ /  __/
 |_| |_|\___|\__,_|_|  \__,_|_|  \__\___/_/\_\\__|\__,_|_|  |_/___\___|

Usage:
    texturize SOURCE... [--size=WxH] [--output=FILE] [--variations=V] [--seed=SEED]
                        [--mode=MODE] [--octaves=O] [--precision=P] [--iterations=I]
                        [--device=DEVICE] [--quiet] [--verbose]
    texturize --help

Examples:
    texturize samples/grass.webp --size=1440x960 --output=result.png
    texturize samples/gravel.png --iterations=200 --precision=1e-5
    texturize samples/sand.tiff  --output=tmp/{source}-{octave}.webp
    texturize samples/brick.jpg  --device=cpu

Options:
    SOURCE                  Path to source image to use as texture.
    -s WxH, --size=WxH      Output resolution as WIDTHxHEIGHT. [default: 640x480]
    -o FILE, --output=FILE  Filename for saving the result, includes format variables.
                            [default: {source}_gen{variation}.png]
    --variations=V          Number of images to generate at same time. [default: 1]
    --seed=SEED             Configure the random number generation.
    --mode=MODE             Either "patch" or "gram" to specify critics. [default: gram]
    --octaves=O             Number of octaves to process. [default: 5]
    --precision=P           Set the quality for the optimization. [default: 1e-4]
    --iterations=I          Maximum number of iterations each octave. [default: 99]
    --device=DEVICE         Hardware to use, either "cpu" or "cuda".
    --quiet                 Suppress any messages going to stdout.
    --verbose               Display more information on stdout.
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
import glob
import itertools

import docopt
import progressbar

import torch
import torch.nn.functional as F

from creativeai.image.encoders import models

from .critics import GramMatrixCritic, PatchCritic
from .solvers import SolverLBFGS, MultiCriticObjective
from . import __version__
from . import io


class OutputLog:
    def __init__(self, config):
        self.quiet = config["--quiet"]
        self.verbose = config["--verbose"]

    def create_progress_bar(self, iterations):
        widgets = [
            progressbar.SimpleProgress(),
            " | ",
            progressbar.Variable("loss", format="{name}: {value:0.3e}"),
            " ",
            progressbar.Bar(marker="■", fill="·"),
            " ",
            progressbar.ETA(),
        ]
        ProgressBar = progressbar.NullBar if self.quiet else progressbar.ProgressBar
        return ProgressBar(
            max_value=iterations, widgets=widgets, variables={"loss": float("+inf")}
        )

    def debug(self, *args):
        if self.verbose:
            print(*args)

    def info(self, *args):
        if not self.quiet:
            print(*args)


class TextureSynthesizer:
    def __init__(self, device, encoder, lr, precision, max_iter):
        self.device = device
        self.encoder = encoder
        self.lr = lr
        self.precision = precision
        self.max_iter = max_iter

    def prepare(self, critics, image):
        """Extract the features from the source texture and initialize the critics.
        """
        feats = dict(self.encoder.extract(image, [c.get_layers() for c in critics]))
        for critic in critics:
            critic.from_features(feats)

    def run(self, logger, seed_img, critics):
        """Run the optimizer on the image according to the loss returned by the critics.
        """
        image = seed_img.to(self.device).requires_grad_(True)

        obj = MultiCriticObjective(self.encoder, critics)
        opt = SolverLBFGS(obj, image, lr=self.lr)

        progress = logger.create_progress_bar(self.max_iter)

        try:
            for i, loss in self._iterate(opt):
                # Update the progress bar with the result!
                progress.update(i, loss=loss)
                # Constrain the image to the valid color range.
                image.data.clamp_(0.0, 1.0)
                # Return back to the user...
                yield loss, image

            progress.max_value = i + 1
        finally:
            progress.finish()

    def _iterate(self, opt):
        previous = None
        for i in range(self.max_iter):
            # Perform one step of the optimization.
            loss = opt.step()

            # Return this iteration to the caller...
            yield i, loss

            # See if we can terminate the optimization early.
            if i > 1 and abs(loss - previous) < self.precision:
                assert i > 10, f"Optimization stalled at iteration {i}."
                break

            previous = loss


class ansi:
    WHITE = "\033[1;97m"
    BLACK = "\033[0;30m\033[47m"
    PINK = "\033[1;35m"
    ENDC = "\033[0m\033[49m"


def process_file(config, source):
    log = config["--logger"]
    for octave, result_img in process_image(config, io.load_image_from_file(source)):
        filenames = []
        for i, result in enumerate(result_img):
            # Save the files for each octave to disk.
            filename = config["--output"].format(
                octave=octave,
                source=os.path.splitext(os.path.basename(source))[0],
                variation=i,
            )
            result.save(filename)
            log.debug("\n=> output:", filename)
            filenames.append(filename)

    return filenames


@torch.no_grad()
def process_image(config, source):
    # Load the original image.
    texture_img = io.load_tensor_from_image(source, device="cpu")

    # Configure the critics.
    if config["--mode"] == "patch":
        critics = [PatchCritic(layer=l) for l in ("1_1", "2_1", "3_1")]
        config["--noise"] = 0.0
    else:
        critics = [
            GramMatrixCritic(layer=l)
            for l in ("1_1", "1_1:2_1", "2_1", "2_1:3_1", "3_1")
        ]
        config["--noise"] = 0.2

    # Encoder used by all the critics.
    encoder = models.VGG11(pretrained=True, pool_type=torch.nn.AvgPool2d)
    encoder = encoder.to(config["--device"], dtype=torch.float32)

    # Generate the starting image for the optimization.
    octaves = int(config["--octaves"])
    result_size = list(map(int, config["--size"].split("x")))[::-1]
    result_img = torch.empty(
        (
            int(config["--variations"]),
            3,
            result_size[0] // 2 ** (octaves + 1),
            result_size[1] // 2 ** (octaves + 1),
        ),
        device=config["--device"],
        dtype=torch.float32,
    ).uniform_(0.4, 0.6)

    # Coarse-to-fine rendering, number of octaves specified by user.
    log = config["--logger"]
    for i, octave in enumerate(2 ** s for s in range(octaves - 1, -1, -1)):
        # Each octave we start a new optimization process.
        synth = TextureSynthesizer(
            config["--device"],
            encoder,
            lr=1.0,
            precision=float(config["--precision"]),
            max_iter=int(config["--iterations"]),
        )
        log.info(ansi.BLACK + "\n OCTAVE", f"#{i} " + ansi.ENDC)
        log.debug("<- scale:", f"1/{octave}")

        # Create downscaled version of original texture to match this octave.
        texture_cur = F.interpolate(
            texture_img,
            scale_factor=1.0 / octave,
            mode="area",
            recompute_scale_factor=False,
        ).to(config["--device"])
        synth.prepare(critics, texture_cur)
        log.debug("<- texture:", tuple(texture_cur.shape[2:]))
        del texture_cur

        # Compute the seed image for this octave, sprinkling a bit of gaussian noise.
        size = result_size[0] // octave, result_size[1] // octave
        seed_img = F.interpolate(result_img, size, mode="bicubic", align_corners=False)
        if config['--noise'] > 0.0:
            seed_img += torch.empty_like(seed_img).normal_(std=config['--noise'])
        log.debug("<- seed:", tuple(seed_img.shape[2:]), "\n")
        del result_img

        # Now we can enable the automatic gradient computation to run the optimization.
        with torch.enable_grad():
            for _, result_img in synth.run(log, seed_img, critics):
                pass
        del synth

        output_img = F.interpolate(result_img, size=result_size, mode="nearest").cpu()
        yield octave, [
            io.save_tensor_to_image(output_img[j : j + 1])
            for j in range(output_img.shape[0])
        ]
        del output_img


def main():
    # Parse the command-line options based on the script's documentation.
    config = docopt.docopt(__doc__[356:], version=__version__)
    config["--logger"] = log = OutputLog(config)
    log.info(ansi.PINK + "    " + __doc__[:356] + ansi.ENDC)

    # Determine which device to use by default, then set it up.
    if config["--device"] is None:
        config["--device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["--device"] = torch.device(config["--device"])

    # Scan all the files based on the patterns specified.
    files = itertools.chain.from_iterable(glob.glob(s) for s in config["SOURCE"])
    for filename in files:
        # If there's a random seed, use it for all images.
        if config["--seed"] is not None:
            seed = int(config["--seed"])
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        # By default, disable autograd until the core optimization loop.
        with torch.no_grad():
            try:
                result = process_file(config, filename)
                log.info(ansi.PINK + "\n=> result:", result, ansi.ENDC)
            except KeyboardInterrupt:
                print(ansi.PINK + "\nCTRL+C detected, interrupting..." + ansi.ENDC)
                break


if __name__ == "__main__":
    main()
