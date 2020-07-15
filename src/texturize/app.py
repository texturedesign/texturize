# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import os
import math
import itertools
import collections

import torch
import torch.nn.functional as F

from .logger import get_default_log
from .critics import GramMatrixCritic, PatchCritic
from .solvers import SolverLBFGS, MultiCriticObjective
from .io import *


__all__ = ["Application", "Result", "TextureSynthesizer"]


class TextureSynthesizer:
    def __init__(self, device, encoder, lr, quality):
        self.device = device
        self.encoder = encoder
        self.quality = quality
        self.learning_rate = lr

    def run(self, log, seed_img, critics):
        """Run the optimizer on the image according to the loss returned by the critics.
        """
        critics = list(itertools.chain.from_iterable(critics))
        image = seed_img.to(self.device).requires_grad_(True)

        obj = MultiCriticObjective(self.encoder, critics)
        opt = SolverLBFGS(obj, image, lr=self.learning_rate)

        progress = log.create_progress_bar(100)

        try:
            for i, loss, converge, lr, retries in self._iterate(opt):
                # Constrain the image to the valid color range.
                image.data.clamp_(0.0, 1.0)

                # Update the progress bar with the result!
                p = min(max(converge * 100.0, 0.0), 100.0)
                progress.update(p, loss=loss, iter=i)

                # Return back to the user...
                yield loss, image, lr, retries

            progress.max_value = i + 1
        finally:
            progress.finish()

    def _iterate(self, opt):
        threshold = math.pow(0.1, 1 + math.log(1 + self.quality))
        converge = 0.0
        previous, plateau = float("+inf"), 0

        for i in itertools.count():
            # Perform one step of the optimization.
            loss, scores, progress = opt.step()

            # Progress metric loosely based on convergence and time.
            current = (previous - loss) / loss
            c = math.exp(-max(current - threshold, 0.0) / (math.log(1+i) * 0.05))
            converge = converge * 0.8 + 0.2 * c

            # Return this iteration to the caller...
            yield i, loss, converge, opt.lr, opt.retries

            # See if we can terminate the optimization early.
            if i > 20 and progress and current <= threshold:
                plateau += 1
                if plateau > 2:
                    break
            else:
                plateau = 0

            previous = min(loss, previous)


Result = collections.namedtuple(
    "Result", ["images", "octave", "scale", "iteration", "loss", "rate", "retries"]
)


class Application:
    def __init__(self, log=None, device=None, precision=None):
        # Setup the output and logging to use throughout the synthesis.
        self.log = log or get_default_log()
        # Determine which device use based on what's available.
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # The floating point format is 32-bit by default, 16-bit supported.
        self.precision = getattr(torch, precision or "float32")

    def process_octave(self, result_img, encoder, critics, octave, scale, quality):
        # Each octave we start a new optimization process.
        synth = TextureSynthesizer(self.device, encoder, lr=1.0, quality=quality)
        result_img = result_img.to(dtype=self.precision)

        # Now we can enable the automatic gradient computation to run the optimization.
        with torch.enable_grad():
            # The first iteration contains the rescaled image with noise.
            yield Result(result_img, octave, scale, 0, float("+inf"), 1.0, 0)

            for iteration, (loss, result_img, lr, retries) in enumerate(
                synth.run(self.log, result_img, critics), start=1
            ):
                yield Result(result_img, octave, scale, iteration, loss, lr, retries)

            # The last iteration is repeated to indicate completion.
            yield Result(result_img, octave, scale, -iteration, loss, lr, retries)
