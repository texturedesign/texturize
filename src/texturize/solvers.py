# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import torch.optim


class SolverLBFGS:
    """Encapsulate the L-BFGS optimizer from PyTorch with a standard interface.
    """

    def __init__(self, objective, image, lr=1.0):
        self.objective = objective
        self.image = image
        self.lr = lr
        self.retries = 0
        self.last_result = (float("+inf"), None)
        self.last_image = None

        self.reset_optimizer()

    def reset_optimizer(self):
        self.optimizer = torch.optim.LBFGS(
            [self.image], lr=self.lr, max_iter=2, max_eval=4, history_size=10
        )
        self.iteration = 0

    def update_lr(self, factor=None):
        if factor is not None:
            self.lr *= factor

        # The first forty iterations, we increase the learning rate slowly to full value.
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr * min(self.iteration / 40.0, 1.0) ** 2

    def call_objective(self):
        """This function wraps the main LBFGS optimizer from PyTorch and uses simple
        hard-coded heuristics to determine its stability, and how best to manage it.

        Informally, it acts like a look-ahead optimizer that rolls back the step if
        there was a divergence in the optimization.
        """
        # Update the learning-rate dynamically, prepare optimizer.
        self.iteration += 1
        self.update_lr()

        # Each iteration we reset the accumulated gradients.
        self.optimizer.zero_grad()

        # Prepare the image for evaluation, then run the objective.
        self.image.data.clamp_(0.0, 1.0)
        loss, scores = self.objective(self.image)

        # Did the optimizer make progress as expected?
        cur_result = self.image.grad.data.abs().mean()
        if cur_result < self.last_result[0] * 8.0:
            self.next_result = loss, scores

            if cur_result < self.last_result[0] * 2.0:
                self.last_image = self.image.data.cpu().clone()
                self.last_result = (cur_result.item(), loss)
            return loss * 1.0

        # Look-ahead failed, so restore the image from the backup.
        self.image.data[:] = self.last_image.to(self.image.device)
        self.image.data[:] += torch.empty_like(self.image.data).normal_(std=1e-3)

        # There was a small regression: dampen the gradients and reduce step size.
        if cur_result < self.last_result[0] * 24.0:
            self.image.grad.data.mul_(self.last_result[0] / cur_result)
            self.update_lr(factor=0.95)
            self.next_result = loss, scores
            return loss * 2.0

        self.update_lr(factor=0.8)
        raise ValueError

    def step(self):
        """Perform one iteration of the optimization process.  This function will catch
        the optimizer diverging, and reset the optimizer's internal state.
        """
        while True:
            try:
                # This optimizer decides when and how to call the objective.
                self.optimizer.step(self.call_objective)
                break
            except ValueError:
                # To restart the optimization, we create a new instance from same image.
                self.reset_optimizer()
                self.retries += 1

        # Report progress once the first few retries are done.
        loss, score = self.next_result
        return loss, score, self.iteration > 3


class SolverSGD:
    """Encapsulate the SGD or Adam optimizers from PyTorch with a standard interface.
    """

    def __init__(self, objective, image, opt_class="SGD", lr=1.0):
        self.objective = objective
        self.image = image
        self.lr = lr

        self.optimizer = getattr(torch.optim, opt_class)([image], lr=lr)
        self.iteration = 1

    def step(self):
        # The first 10 iterations, we increase the learning rate slowly to full value.
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr * min(self.iteration / 10.0, 1.0) ** 2

        # Each iteration we reset the accumulated gradients and compute the objective.
        self.iteration += 1
        self.optimizer.zero_grad()

        # Let the objective compute the loss and its gradients.
        loss = self.objective(self.image)
        assert not torch.isnan(self.image.grad).any(), f"Gradient is NaN, loss {loss}."

        # Now compute the updates to the image according to the gradients.
        self.optimizer.step()
        assert not torch.isnan(self.image).any(), f"Image is NaN for loss {loss}."

        return loss


class MultiCriticObjective:
    """An `Objective` that defines a problem to be solved by evaluating candidate
    solutions (i.e. images) and returning the computed error.

    This objective evaluates a list of critics to produce a final "loss" that's the sum
    of all the scores returned by the critics.  It's also responsible for computing the
    gradients.
    """

    def __init__(self, encoder, critics):
        self.encoder = encoder
        self.critics = critics

    def __call__(self, image):
        """Main evaluation function that's called by the solver.  Processes the image,
        computes the gradients, and returns the loss.
        """

        # Extract features from image.
        layers = [c.get_layers() for c in self.critics]
        feats = dict(self.encoder.extract(image, layers))

        # Apply all the critics one by one.
        scores = []
        for critic in self.critics:
            total = 0.0
            for loss in critic.evaluate(feats):
                total += loss
            scores.append(total)

        # Calculate the final loss and compute the gradients.
        loss = (sum(scores) / len(scores)).mean()
        loss.backward()

        return loss, scores
