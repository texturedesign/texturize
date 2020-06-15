#!/usr/bin/env python3
r"""                         _   _            _              _         
  _ __   ___ _   _ _ __ __ _| | | |_ _____  _| |_ _   _ _ __(_)_______ 
 | '_ \ / _ \ | | | '__/ _` | | | __/ _ \ \/ / __| | | | '__| |_  / _ \
 | | | |  __/ |_| | | | (_| | | | ||  __/>  <| |_| |_| | |  | |/ /  __/
 |_| |_|\___|\__,_|_|  \__,_|_|  \__\___/_/\_\\__|\__,_|_|  |_/___\___|

Usage:
    texturize SOURCE... [--size=WxH] [--output=FILE] [--variations=V] [--seed=SEED]
                        [--mode=MODE] [--octaves=O] [--threshold=H] [--iterations=I]
                        [--device=DEVICE] [--precision=PRECISION] [--quiet] [--verbose]
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
    --threshold=T           Quality for optimization, lower is better. [default: 1e-4]
    --iterations=I          Maximum number of iterations each octave. [default: 99]
    --device=DEVICE         Hardware to use, either "cpu" or "cuda".
    --precision=PRECISION   Floating-point format to use, "float16" or "float32".
    --quiet                 Suppress any messages going to stdout.
    --verbose               Display more information on stdout.
    -h, --help              Show this message.
"""
#
# Copyright (c) 2020, Novelty Factory KG.
#
# neural-texturize is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License version 3. This program is distributed
# in the hope that it will be useful but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#

import glob
import itertools

import docopt
import progressbar
from schema import Schema, Use, And, Or

import torch

from . import __version__
from .api import process_single_file, ansi, OutputLog


def validate(config):
    # Determine the shape of output tensor (H, W) from specified resolution.
    def split_size(size: str):
        return tuple(map(int, size.split("x")))

    sch = Schema(
        {
            "SOURCE": [str],
            "size": And(Use(split_size), tuple),
            "output": str,
            "variations": Use(int),
            "seed": Or(None, Use(int)),
            "mode": Or("patch", "gram"),
            "octaves": Use(int),
            "threshold": Use(float),
            "iterations": Use(int),
            "device": Or(None, "cpu", "cuda"),
            "precision": Or(None, "float16", "float32"),
            "help": Use(bool),
            "quiet": Use(bool),
            "verbose": Use(bool),
        },
        ignore_extra_keys=True,
    )
    return sch.validate({k.replace("--", ""): v for k, v in config.items()})


def main():
    # Parse the command-line options based on the script's documentation.
    config = docopt.docopt(__doc__[356:], version=__version__, help=False)
    # Ensure the user-specified values are correct.
    config = validate(config)
    filenames, seed, quiet, verbose, help = [
        config.pop(k) for k in ("SOURCE", "seed", "quiet", "verbose", "help")
    ]

    # Setup the output logging and display the logo!
    log = OutputLog(quiet, verbose)
    log.notice(ansi.PINK + "    " + __doc__[:356] + ansi.ENDC)
    if help is True:
        log.notice(__doc__[356:])
        return

    # Scan all the files based on the patterns specified.
    files = itertools.chain.from_iterable(glob.glob(s) for s in filenames)
    for filename in files:
        # If there's a random seed, use the same for all images.
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        # Process the files one by one, each may have multiple variations.
        try:
            result = process_single_file(filename, log, **config)
            log.notice(ansi.PINK + "\n=> result:", result, ansi.ENDC)
        except KeyboardInterrupt:
            print(ansi.PINK + "\nCTRL+C detected, interrupting..." + ansi.ENDC)
            break


if __name__ == "__main__":
    main()
