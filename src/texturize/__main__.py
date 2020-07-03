#!/usr/bin/env python3
r"""                         _   _            _              _         
  _ __   ___ _   _ _ __ __ _| | | |_ _____  _| |_ _   _ _ __(_)_______ 
 | '_ \ / _ \ | | | '__/ _` | | | __/ _ \ \/ / __| | | | '__| |_  / _ \
 | | | |  __/ |_| | | | (_| | | | ||  __/>  <| |_| |_| | |  | |/ /  __/
 |_| |_|\___|\__,_|_|  \__,_|_|  \__\___/_/\_\\__|\__,_|_|  |_/___\___|

Usage:
    texturize remix SOURCE... [options]
    texturize remake SOURCE... as TARGET [options]
    texturize blend SOURCE... TARGET [options]
    texturize --help

Examples:
    texturize remix samples/grass.webp --size=1440x960 --output=result.png
    texturize remix samples/gravel.png --iterations=200 --precision=1e-5
    texturize remix samples/sand.tiff  --output=tmp/{source}-{octave}.webp
    texturize remix samples/brick.jpg  --device=cpu

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

import os
import glob
import itertools

import docopt
from schema import Schema, Use, And, Or

import torch

from . import __version__
from . import commands
from .api import process_single_command
from .logger import ansi, ConsoleLog


def validate(config):
    # Determine the shape of output tensor (H, W) from specified resolution.
    def split_size(size: str):
        return tuple(map(int, size.split("x")))

    sch = Schema(
        {
            "SOURCE": [str],
            "TARGET": Or(None, str),
            "size": And(Use(split_size), tuple),
            "output": str,
            "variations": Use(int),
            "seed": Or(None, Use(int)),
            "mode": Or("patch", "gram", "hist"),
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
    command = [cmd for cmd in ("remix", "remake", "blend") if config[cmd]][0]

    # Ensure the user-specified values are correct.
    config = validate(config)
    sources, target, output, seed, quiet, verbose, help = [
        config.pop(k) for k in ("SOURCE", "TARGET", "output", "seed", "quiet", "verbose", "help")
    ]

    # Setup the output logging and display the logo!
    log = ConsoleLog(quiet, verbose)
    log.notice(ansi.PINK + "    " + __doc__[:356] + ansi.ENDC)
    if help is True:
        log.notice(__doc__[356:])
        return

    # Scan all the files based on the patterns specified.
    files = itertools.chain.from_iterable(glob.glob(s) for s in sources)
    for filename in files:
        # If there's a random seed, use the same for all images.
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        if command == "remix":
            cmd = commands.Remix(filename)
        if command == "remake":
            cmd = commands.Remix(filename, target)
        if command == "blend":
            cmd = commands.Blend(filename, target)

        # Process the files one by one, each may have multiple variations.
        try:
            config["output"] = output
            config["output"] = config["output"].replace(
                "{source}", os.path.splitext(os.path.basename(filename))[0]
            )
            if target:
                config["output"] = config["output"].replace(
                    "{target}", os.path.splitext(os.path.basename(target))[0]
                )

            result = process_single_command(cmd, log, **config)
            log.notice(ansi.PINK + "\n=> result:", result, ansi.ENDC)
        except KeyboardInterrupt:
            print(ansi.PINK + "\nCTRL+C detected, interrupting..." + ansi.ENDC)
            break


if __name__ == "__main__":
    main()
