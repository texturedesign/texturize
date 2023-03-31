# texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import os
import math
import itertools
import collections

import torch
import torch.nn.functional as F

from .logger import get_default_log
from .critics import GramMatrixCritic, PatchCritic
from .solvers import (
    SolverSGD,
    SolverLBFGS,
    MultiCriticObjective,
    SequentialCriticObjective,
)
from .io import *


__all__ = ["Application", "Result", "TextureSynthesizer"]


class TextureSynthesizer:
    def __init__(self, device, encoder, lr, quality=None):
        self.device = device
        self.encoder = encoder
        self.quality = quality or 1
        self.learning_rate = lr

    def run(self, progress, seed_img, *args):
        for oc, sc in itertools.product(
            [MultiCriticObjective, SequentialCriticObjective], [SolverLBFGS, SolverSGD],
        ):
            if sc == SolverLBFGS and seed_img.dtype == torch.float16:
                continue

            try:
                yield from self._run(progress, seed_img, *args, objective_class=oc, solver_class=sc)
                progress.finish()
                return
            except RuntimeError as e:
                if "CUDA out of memory." not in str(e):
                    raise

                import gc; gc.collect
                torch.cuda.empty_cache()

        raise RuntimeError("CUDA out of memory.")

    def _run(
        self, progress, seed_img, critics, objective_class=None, solver_class=None
    ):
        """Run the optimizer on the image according to the loss returned by the critics.
        """
        critics = list(itertools.chain.from_iterable(critics))
        image = seed_img.to(self.device)
        alpha = None if image.shape[1] == 3 else image[:, 3:4]
        image = image[:, 0:3].detach().requires_grad_(True)

        obj = objective_class(self.encoder, critics, alpha=alpha)
        opt = solver_class(obj, image, lr=self.learning_rate)

        for i, loss, converge, lr, retries in self._iterate(opt):
            # Constrain the image to the valid color range.
            image.data.clamp_(0.0, 1.0)

            # Update the progress bar with the result!
            p = min(max(converge * 100.0, 0.0), 100.0)
            progress.update(p, loss=loss, iter=i)

            # Return back to the user...
            yield loss, image, lr, retries

        progress.max_value = i + 1

    def _iterate(self, opt):
        threshold = math.pow(0.1, 1 + math.log(1 + self.quality))
        converge = 0.0
        previous, plateau = float("+inf"), 0

        for i in itertools.count():
            # Perform one step of the optimization.
            loss, scores = opt.step()

            # Progress metric loosely based on convergence and time.
            current = (previous - loss) / loss
            c = math.exp(-max(current - threshold, 0.0) / (math.log(2 + i) * 0.05))
            converge = converge * 0.8 + 0.2 * c

            # Return this iteration to the caller...
            yield i, loss, converge, opt.lr, opt.retries

            # See if we can terminate the optimization early.
            if i > 3 and current <= threshold:
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

        # The first iteration contains the rescaled image with noise.
        yield Result(result_img, octave, scale, 0, float("+inf"), 1.0, 0)

        for iteration, (loss, result_img, lr, retries) in enumerate(
            synth.run(self.progress, result_img, critics), start=1
        ):
            yield Result(result_img, octave, scale, iteration, loss, lr, retries)

        # The last iteration is repeated to indicate completion.
        yield Result(result_img, octave, scale, -iteration, loss, lr, retries)
