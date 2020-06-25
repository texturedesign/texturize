# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import os

import torch
import torch.nn.functional as F

from creativeai.image.encoders import models

from .critics import GramMatrixCritic, PatchCritic
from .solvers import SolverLBFGS, MultiCriticObjective
from .io import *


class TextureSynthesizer:
    def __init__(self, device, encoder, lr, threshold, max_iter):
        self.device = device
        self.encoder = encoder
        self.threshold = threshold
        self.max_iter = max_iter
        self.learning_rate = lr

    def prepare(self, critics, image):
        """Extract the features from the source texture and initialize the critics.
        """
        feats = dict(self.encoder.extract(image, [c.get_layers() for c in critics]))
        for critic in critics:
            critic.from_features(feats)

    def run(self, log, seed_img, critics):
        """Run the optimizer on the image according to the loss returned by the critics.
        """
        image = seed_img.to(self.device).requires_grad_(True)

        obj = MultiCriticObjective(self.encoder, critics)
        opt = SolverLBFGS(obj, image, lr=self.learning_rate)

        progress = log.create_progress_bar(self.max_iter)

        try:
            for i, loss, lr, retries in self._iterate(opt):
                # Constrain the image to the valid color range.
                image.data.clamp_(0.0, 1.0)

                # Update the progress bar with the result!
                progress.update(i, loss=loss)

                # Return back to the user...
                yield loss, image, lr, retries

            progress.max_value = i + 1
        finally:
            progress.finish()

    def _iterate(self, opt):
        previous, plateau = float("+inf"), 0
        for i in range(self.max_iter):
            # Perform one step of the optimization.
            loss, scores, progress = opt.step()

            if not progress:
                continue

            # Return this iteration to the caller...
            yield i, loss, opt.lr, opt.retries

            # See if we can terminate the optimization early.
            if i > 10 and abs(loss - previous) <= self.threshold:
                plateau += 1
                if plateau > 2:
                    break
            else:
                plateau = 0

            previous = min(loss, previous)
