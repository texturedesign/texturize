# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import os
import collections
import progressbar

import torch
import torch.nn.functional as F
from creativeai.image.encoders import models

from .critics import GramMatrixCritic, PatchCritic, HistogramCritic
from .app import TextureSynthesizer
from .io import *


class ansi:
    WHITE = "\033[1;97m"
    BLACK = "\033[0;30m\033[47m"
    YELLOW = "\033[1;33m"
    PINK = "\033[1;35m"
    ENDC = "\033[0m\033[49m"


class EmptyLog:
    def notice(self, *args):
        pass

    def info(self, *args):
        pass

    def debug(self, *args):
        pass

    def warn(self, *args):
        pass

    def create_progress_bar(self, iterations):
        return progressbar.NullBar(max_value=iterations)


class ConsoleLog:
    def __init__(self, quiet, verbose):
        self.quiet = quiet
        self.verbose = verbose

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

    def warn(self, *args):
        print(ansi.YELLOW + "".join(args) + ansi.ENDC)


class NotebookLog:
    class ProgressBar:
        def __init__(self, max_iter):
            import ipywidgets

            self.bar = ipywidgets.IntProgress(
                value=0,
                min=0,
                max=max_iter,
                step=1,
                description="",
                bar_style="",
                orientation="horizontal",
                layout=ipywidgets.Layout(width="100%", margin="0"),
            )

            from IPython.display import display

            display(self.bar)

        def update(self, value, **keywords):
            self.bar.value = value

        def finish(self):
            self.bar.close()

    def create_progress_bar(self, iterations):
        return NotebookLog.ProgressBar(iterations)

    def debug(self, *args):
        pass

    def notice(self, *args):
        pass

    def info(self, *args):
        pass

    def warn(self, *args):
        pass


def get_default_log():
    try:
        get_ipython
        return NotebookLog()
    except NameError:
        return EmptyLog()


Result = collections.namedtuple(
    "Result", ["images", "octave", "scale", "iteration", "loss", "rate", "retries"]
)


@torch.no_grad()
def process_octaves(sources, **kwargs):
    """Synthesize a new texture from sources and return a PyTorch tensor at each octave.
    """

    for r in process_iterations(sources, **kwargs):
        if r.iteration >= 0:
            continue

        yield Result(
            r.images, r.octave, r.scale, -r.iteration, r.loss, r.rate, r.retries
        )


@torch.no_grad()
def process_iterations(
    sources,
    log: object = None,
    size: tuple = None,
    octaves: int = -1,
    mode: str = "gram",
    variations: int = 1,
    iterations: int = 200,
    threshold: float = 1e-5,
    device: str = None,
    precision: str = None,
):
    """Synthesize a new texture and return a PyTorch tensor at each iteration.
    """

    # Setup the output and logging to use throughout the synthesis.
    log = log or get_default_log()

    # Determine which device and dtype to use by default, then set it up.
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    precision = getattr(torch, precision or "float32")

    # Load the original images, always on the host device to save memory.
    texture_critics = []
    for source in sources:
        texture_img = load_tensor_from_image(source, device="cpu").to(dtype=precision)

        if mode == "patch":
            critics = [PatchCritic(layer=l) for l in ("3_1", "2_1", "1_1")]
        elif mode == "gram":
            critics = [
                GramMatrixCritic(layer=l)
                for l in ("1_1", "1_1:2_1", "2_1", "2_1:3_1", "3_1")
            ]
        elif mode == "hist":
            critics = [HistogramCritic(layer=l) for l in ("1_1", "2_1", "3_1")]

        texture_critics.append((texture_img, critics))

    # Encoder used by all the critics.
    encoder = models.VGG11(pretrained=True, pool_type=torch.nn.AvgPool2d)
    encoder = encoder.to(device=device, dtype=precision)

    # Generate the starting image for the optimization.
    result_img = (
        torch.empty(
            (variations, 1, size[1] // 2 ** octaves, size[0] // 2 ** octaves),
            device=device,
            dtype=torch.float32,
        ).normal_(std=0.1)
        + texture_img.mean(dim=(2, 3), keepdim=True).to(device)
    ).to(dtype=precision)

    # Coarse-to-fine rendering, number of octaves specified by user.
    for octave, scale in enumerate(2 ** s for s in range(octaves - 1, -1, -1)):
        # Each octave we start a new optimization process.
        synth = TextureSynthesizer(
            device, encoder, lr=1.0, threshold=threshold, max_iter=iterations,
        )
        log.info(f"\n OCTAVE #{octave} ")
        log.debug("<- scale:", f"1/{scale}")

        # Create downscaled version of original texture to match this octave.
        all_critics = []
        for (texture_img, critics) in texture_critics:
            texture_cur = F.interpolate(
                texture_img,
                scale_factor=1.0 / scale,
                mode="area",
                recompute_scale_factor=False,
            ).to(device=device, dtype=precision)
            synth.prepare(critics, texture_cur)
            all_critics.extend(critics)
            log.debug("<- texture:", tuple(texture_cur.shape[2:]))
            del texture_cur

        # Compute the seed image for this octave, sprinkling a bit of gaussian noise.
        result_size = size[1] // scale, size[0] // scale
        seed_img = F.interpolate(
            result_img, result_size, mode="bicubic", align_corners=False
        ).clamp_(0.0, 1.0)
        log.debug("<- seed:", tuple(seed_img.shape[2:]), "\n")
        del result_img

        # Now we can enable the automatic gradient computation to run the optimization.
        with torch.enable_grad():
            # The first iteration contains the rescaled image with noise.
            yield Result(seed_img, octave, scale, 0, float("+inf"), 1.0, 0)

            for iteration, (loss, result_img, lr, retries) in enumerate(
                synth.run(log, seed_img.to(dtype=precision), all_critics), start=1
            ):
                yield Result(result_img, octave, scale, iteration, loss, lr, retries)

            # The last iteration is repeated to indicate completion.
            yield Result(result_img, octave, scale, -iteration, loss, lr, retries)
        del synth


def process_single_file(source, log: object, output: str = None, **config: dict):
    for result in process_octaves([load_image_from_file(source)], log=log, **config):
        images = save_tensor_to_images(result.images)
        filenames = []
        for i, image in enumerate(images):
            # Save the files for each octave to disk.
            filename = output.format(
                octave=result.octave,
                source=os.path.splitext(os.path.basename(source))[0],
                variation=i,
            )
            image.save(filename)
            log.debug("\n=> output:", filename)
            filenames.append(filename)

    return filenames
