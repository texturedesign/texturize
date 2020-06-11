# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import os
import progressbar

import torch
import torch.nn.functional as F
from creativeai.image.encoders import models

from .critics import GramMatrixCritic, PatchCritic
from .app import TextureSynthesizer
from .io import *


class ansi:
    WHITE = "\033[1;97m"
    BLACK = "\033[0;30m\033[47m"
    PINK = "\033[1;35m"
    ENDC = "\033[0m\033[49m"


class EmptyLog:
    def notice(self, *args):
        pass

    def info(self, *args):
        pass

    def debug(self, *args):
        pass

    def create_progress_bar(self, iterations):
        import progressbar

        return progressbar.NullBar(max_value=iterations)


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

    def notice(self, *args):
        if not self.quiet:
            print(*args)

    def info(self, *args):
        if not self.quiet:
            print(ansi.BLACK + "".join(args) + ansi.ENDC)


@torch.no_grad()
def process_octaves(
    source,
    log: object = EmptyLog(),
    size: tuple = None,
    octaves: int = -1,
    mode: str = "gram",
    variations: int = 1,
    iterations: int = 99,
    precision: float = 1e-5,
    device: str = None,
):
    # Determine which device to use by default, then set it up.
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load the original image, always on the host device to save memory.
    texture_img = load_tensor_from_image(source, device="cpu")

    # Configure the critics.
    if mode == "patch":
        critics = [PatchCritic(layer=l) for l in ("1_1", "2_1", "3_1")]
        noise = 0.0
    else:  # mode == "gram"
        critics = [
            GramMatrixCritic(layer=l)
            for l in ("1_1", "1_1:2_1", "2_1", "2_1:3_1", "3_1")
        ]
        noise = 0.1

    # Encoder used by all the critics.
    encoder = models.VGG11(pretrained=True, pool_type=torch.nn.AvgPool2d)
    encoder = encoder.to(device, dtype=torch.float32)

    # Generate the starting image for the optimization.
    result_img = torch.empty(
        (variations, 1, size[1] // 2 ** (octaves + 1), size[0] // 2 ** (octaves + 1)),
        device=device,
        dtype=torch.float32,
    ).normal_(std=0.05) + texture_img.mean(dim=(2, 3), keepdim=True).to(device)

    # Coarse-to-fine rendering, number of octaves specified by user.
    for i, octave in enumerate(2 ** s for s in range(octaves - 1, -1, -1)):
        # Each octave we start a new optimization process.
        synth = TextureSynthesizer(
            device, encoder, lr=1.0, precision=precision, max_iter=iterations,
        )
        log.info(f"\n OCTAVE #{i} ")
        log.debug("<- scale:", f"1/{octave}")

        # Create downscaled version of original texture to match this octave.
        texture_cur = F.interpolate(
            texture_img,
            scale_factor=1.0 / octave,
            mode="area",
            recompute_scale_factor=False,
        ).to(device)
        synth.prepare(critics, texture_cur)
        log.debug("<- texture:", tuple(texture_cur.shape[2:]))
        del texture_cur

        # Compute the seed image for this octave, sprinkling a bit of gaussian noise.
        result_size = size[1] // octave, size[0] // octave
        seed_img = F.interpolate(
            result_img, result_size, mode="bicubic", align_corners=False
        )
        if noise > 0.0:
            b, _, h, w = seed_img.shape
            seed_img += seed_img.new_empty(size=(b, 1, h, w)).normal_(std=noise)
        log.debug("<- seed:", tuple(seed_img.shape[2:]), "\n")
        del result_img

        # Now we can enable the automatic gradient computation to run the optimization.
        with torch.enable_grad():
            for loss, result_img in synth.run(log, seed_img, critics):
                pass
        del synth

        output_img = F.interpolate(
            result_img, size=(size[1], size[0]), mode="nearest"
        ).cpu()
        yield octave, loss, [
            save_tensor_to_image(output_img[j : j + 1])
            for j in range(output_img.shape[0])
        ]
        del output_img


def process_single_file(source, log: object, output: str = None, **config: dict):
    for octave, _, result_img in process_octaves(
        load_image_from_file(source), log, **config
    ):
        filenames = []
        for i, result in enumerate(result_img):
            # Save the files for each octave to disk.
            filename = output.format(
                octave=octave,
                source=os.path.splitext(os.path.basename(source))[0],
                variation=i,
            )
            result.save(filename)
            log.debug("\n=> output:", filename)
            filenames.append(filename)

    return filenames
