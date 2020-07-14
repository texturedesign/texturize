# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import itertools

import torch
import torch.nn.functional as F


def torch_flatten_2d(a):
    a = a.permute(1, 0, 2, 3)
    return a.reshape(a.shape[:1] + (-1,))


def torch_gather_2d(array, indices):
    """Extract the content of an array using the 2D coordinates provided.
    """
    batch = torch.arange(
        0, array.shape[0], dtype=torch.long, device=indices.device
    ).view(-1, 1, 1)

    idx = batch * (array.shape[2] * array.shape[3]) + (
        indices[:, 0, :, :] * array.shape[3] + indices[:, 1, :, :]
    )
    flat_array = torch_flatten_2d(array)
    x = torch.index_select(flat_array, 1, idx.view(-1))

    result = x.view(array.shape[1:2] + indices.shape[:1] + indices.shape[2:])
    return result.permute(1, 0, 2, 3)


def torch_scatter_2d(output, indices, values):
    _, c, h, w = output.shape

    assert output.shape[0] == 1
    assert output.shape[1] == values.shape[1]

    chanidx = torch.arange(0, c, dtype=torch.long, device=indices.device).view(-1, 1, 1)

    idx = chanidx * (h * w) + (indices[:, 0] * w + indices[:, 1])
    output.flatten().scatter_(0, idx.flatten(), values.flatten())


def torch_pad_reflect(array, padding):
    return torch.nn.functional.pad(array, pad=padding, mode="reflect")


def iterate_range(size, split=2):
    assert split <= size
    for start, stop in zip(range(0, split), range(1, split + 1)):
        yield (
            max(0, (size * start) // split),
            min(size, (size * stop) // split),
        )


def cosine_similarity_matrix_1d(source, target, eps=1e-8):
    source = source / (torch.norm(source, dim=1, keepdim=True) + eps)
    target = target / (torch.norm(target, dim=1, keepdim=True) + eps)

    result = torch.bmm(source.permute(0, 2, 1), target)
    return torch.clamp(result, max=1.0 / eps)


def cosine_similarity_vector_1d(source, target, eps=1e-8):
    source = source / (torch.norm(source, dim=1, keepdim=True) + eps)
    target = target / (torch.norm(target, dim=1, keepdim=True) + eps)

    source = source.expand_as(target)

    result = torch.sum(source * target, dim=1)
    return torch.clamp(result, max=1.0 / eps)


class Mapping:
    def __init__(self, size, device="cpu"):
        b, _, h, w = size
        self.device = torch.device(device)
        self.indices = torch.empty((b, 2, h, w), dtype=torch.int64, device=device)
        self.scores = torch.full((b, 1, h, w), float("-inf"), device=device)
        self.biases = None
        self.target_size = None

    def clone(self):
        b, _, h, w = self.indices.shape
        clone = Mapping((b, -1, h, w), self.device)
        clone.indices[:] = self.indices
        clone.scores[:] = self.scores
        clone.biases = self.biases.copy()
        return clone

    def setup_biases(self, target_size):
        b, _, h, w = target_size
        self.biases = torch.full((b, 1, h, w), 0.0, device=self.device)

    def rescale(self, target_size):
        factor = torch.tensor(target_size, dtype=torch.float) / torch.tensor(
            self.target_size[2:], dtype=torch.float
        )
        self.indices = (
            self.indices.float().mul(factor.to(self.device).view(1, 2, 1, 1)).long()
        )
        self.indices[:, 0].clamp_(0, target_size[0] - 1)
        self.indices[:, 1].clamp_(0, target_size[1] - 1)
        self.target_size = self.target_size[:2] + target_size

        self.setup_biases(self.scores.shape[:2] + target_size)

    def resize(self, size):
        self.indices = F.interpolate(
            self.indices.float(), size=size, mode="nearest"
        ).long()
        self.scores = F.interpolate(self.scores, size=size, mode="nearest")

    def improve(self, candidate_scores, candidate_indices):
        candidate_indices = candidate_indices.view(self.indices.shape)
        candidate_scores = candidate_scores.view(self.scores.shape) + torch_gather_2d(
            self.biases, candidate_indices
        )

        cond = candidate_scores > self.scores
        self.indices[:] = torch.where(cond, candidate_indices, self.indices)
        self.scores[:] = torch.where(cond, candidate_scores, self.scores)

    def improve_window(self, this_window, other_window, candidates):
        assert candidates.indices.shape[0] == 1

        sy, dy, sx, dx = other_window
        grid = torch.empty((1, 2, dy, dx), dtype=torch.int64, device=self.device)
        self.meshgrid(grid, offset=(sy, sx), range=(dy, dx))

        chanidx = torch.arange(0, 2, dtype=torch.long, device=self.device).view(-1, 1)
        chanidx = chanidx * (dx * dy)
        indices_2d = torch.index_select(
            grid.flatten(),
            dim=0,
            index=(chanidx + candidates.indices.to(self.device).view(1, -1)).flatten(),
        )

        return self._improve_window(
            this_window, candidates.values.to(self.device), indices_2d.view(1, 2, -1)
        )

    def _improve_window(self, this_window, scores, indices):
        start_y, size_y, start_x, size_x = this_window
        assert indices.ndim == 3 and indices.shape[2] == size_y * size_x

        candidate_scores = scores.view(1, 1, size_y, size_x)
        candidate_indices = indices.view(1, 2, size_y, size_x)

        slice_y, slice_x = (
            slice(start_y, start_y + size_y),
            slice(start_x, start_x + size_x),
        )

        cond = candidate_scores > self.scores[:, :, slice_y, slice_x]
        self.indices[:, :, slice_y, slice_x] = torch.where(
            cond, candidate_indices, self.indices[:, :, slice_y, slice_x]
        )
        self.scores[:, :, slice_y, slice_x] = torch.where(
            cond, candidate_scores, self.scores[:, :, slice_y, slice_x]
        )
        return (cond != 0).sum().item()

    def _improve_scatter(self, this_indices, scores, other_window):
        sy, dy, sx, dx = other_window
        grid = torch.empty((1, 2, dy, dx), dtype=torch.int64, device=self.device)
        self.meshgrid(grid, offset=(sy, sx), range=(dy, dx))

        this_scores = (
            torch_gather_2d(self.scores, this_indices.view(1, 2, dy, dx))
            + self.biases[:, :, sy : sy + dy, sx : sx + dx]
        )
        cond = scores.flatten(2) > this_scores.flatten(2)

        better_indices = this_indices.flatten(2)[cond.expand(1, 2, -1)].view(1, 2, -1)
        if better_indices.shape[2] == 0:
            return 0

        better_scores = scores.flatten(2)[cond].view(1, 1, -1)
        window_indices = grid.flatten(2)[cond.expand(1, 2, -1)].view(1, 2, -1)

        torch_scatter_2d(self.scores, better_indices, better_scores)
        torch_scatter_2d(self.indices, better_indices, window_indices)

        return better_indices.shape[2]

    def from_random(self, target_size):
        assert target_size[0] == 1, "Only 1 feature map supported."
        self.target_size = target_size
        self.randgrid(self.indices, offset=(0, 0), range=target_size[2:])
        self.setup_biases(target_size)
        return self

    def randgrid(self, output, offset, range):
        torch.randint(
            low=offset[0],
            high=offset[0] + range[0],
            size=output[:, 0, :, :].shape,
            out=output[:, 0, :, :],
        )
        torch.randint(
            low=offset[1],
            high=offset[1] + range[1],
            size=output[:, 1, :, :].shape,
            out=output[:, 1, :, :],
        )

    def meshgrid(self, output, offset, range):
        b, _, h, w = output.shape
        output[:, 0, :, :] = (
            torch.arange(h, dtype=torch.float32)
            .mul(range[0] / h)
            .add(offset[0])
            .view((1, h, 1))
            .expand((b, h, 1))
            .long()
        )
        output[:, 1, :, :] = (
            torch.arange(w, dtype=torch.float32)
            .mul(range[1] / w)
            .add(offset[1])
            .view((1, 1, w))
            .expand((b, 1, w))
            .long()
        )

    def from_linear(self, target_size):
        assert target_size[0] == 1, "Only 1 feature map supported."
        self.target_size = target_size
        self.meshgrid(self.indices, offset=(0, 0), range=target_size[2:])
        self.setup_biases(target_size)
        return self


class FeatureMatcher:
    """Implementation of feature matching between two feature maps in 2D arrays, using
    normalized cross-correlation of features as similarity metric.
    """

    def __init__(self, target=None, sources=None, device="cpu", variety=0.0):
        self.device = torch.device(device)
        self.variety = variety

        self.target = None
        self.sources = None
        self.repro_target = None
        self.repro_sources = None

        if sources is not None:
            self.update_sources(sources)
        if target is not None:
            self.update_target(target)

    def clone(self):
        clone = FeatureMatcher(device=self.device)
        clone.sources = self.sources
        clone.target = self.target

        clone.repro_target = self.repro_target.clone()
        clone.repro_sources = self.repro_sources.clone()
        return clone

    def update_target(self, target):
        assert len(target.shape) == 4
        assert target.shape[0] == 1

        self.target = target

        if self.repro_target is None:
            self.repro_target = Mapping(self.target.shape, self.device)
            self.repro_target.from_random(self.sources.shape)
            self.repro_sources.from_random(self.target.shape)

        self.repro_target.scores.fill_(float("-inf"))
        self.repro_sources.scores.fill_(float("-inf"))

        if target.shape[2:] != self.repro_target.indices.shape[2:]:
            self.repro_sources.rescale(target.shape[2:])
            self.repro_target.resize(target.shape[2:])

    def update_sources(self, sources):
        assert len(sources.shape) == 4
        assert sources.shape[0] == 1

        self.sources = sources

        if self.repro_sources is None:
            self.repro_sources = Mapping(self.sources.shape, self.device)

        if sources.shape[2:] != self.repro_sources.indices.shape[2:]:
            self.repro_target.rescale(sources.shape[2:])
            self.repro_sources.resize(sources.shape[2:])

    def update_biases(self):
        sources_value = (
            self.repro_sources.scores
            - torch_gather_2d(self.repro_sources.biases, self.repro_sources.indices)
            - self.repro_target.biases
        )
        target_value = (
            self.repro_target.scores
            - torch_gather_2d(self.repro_target.biases, self.repro_target.indices)
            - self.repro_sources.biases
        )

        k = self.variety
        self.repro_target.biases[:] = -k * (sources_value - sources_value.mean())
        self.repro_sources.biases[:] = -k * (target_value - target_value.mean())

        self.repro_target.scores.fill_(float("-inf"))
        self.repro_sources.scores.fill_(float("-inf"))

    def reconstruct_target(self):
        return torch_gather_2d(
            self.sources, self.repro_target.indices.to(self.sources.device)
        )

    def reconstruct_source(self):
        return torch_gather_2d(
            self.target, self.repro_sources.indices.to(self.sources.device)
        )

    def compare_features_coarse(self, parent, radius=2, split=1):
        def _compare(a, b, repro_a, repro_b, parent_a):
            if parent_a.indices.shape[2] > repro_a.indices.shape[2]:
                return 0

            total = 0
            for (t1, t2) in iterate_range(a.shape[2], split):
                assert t2 >= t1

                indices = F.interpolate(
                    parent_a.indices.float() * 2.0,
                    size=repro_a.indices.shape[2:],
                    mode="nearest",
                )[:, :, t1:t2].long()
                indices += torch.empty_like(indices).random_(-radius, radius + 1)

                indices[:, 0, :, :].clamp_(min=0, max=b.shape[2] - 1)
                indices[:, 1, :, :].clamp_(min=0, max=b.shape[3] - 1)

                total += self._improve(
                    a, (t1, t2 - t1, 0, a.shape[3]), repro_a, b, indices, repro_b,
                )
            return total

        if parent is None:
            return 0

        ts = _compare(
            self.target,
            self.sources,
            self.repro_target,
            self.repro_sources,
            parent.repro_target,
        )
        st = _compare(
            self.sources,
            self.target,
            self.repro_sources,
            self.repro_target,
            parent.repro_sources,
        )
        return ts + st

    def compare_features_matrix(self, split=1):
        assert self.sources.shape[0] == 1, "Only 1 source supported."

        for (t1, t2), (s1, s2) in itertools.product(
            iterate_range(self.target.shape[2], split),
            iterate_range(self.sources.shape[2], split),
        ):
            assert t2 != t1 and s2 != s1

            target_window = self.target[:, :, t1:t2].flatten(2)
            source_window = self.sources[:, :, s1:s2].flatten(2)

            similarity = cosine_similarity_matrix_1d(target_window, source_window)
            similarity += (
                self.repro_target.biases[:, :, s1:s2]
                .to(similarity.device)
                .reshape(1, 1, -1)
            )
            similarity += (
                self.repro_sources.biases[:, :, t1:t2]
                .to(similarity.device)
                .reshape(1, -1, 1)
            )

            best_source = torch.max(similarity, dim=2)
            self.repro_target.improve_window(
                (t1, t2 - t1, 0, self.target.shape[3]),
                (s1, s2 - s1, 0, self.sources.shape[3]),
                best_source,
            )

            best_target = torch.max(similarity, dim=1)
            self.repro_sources.improve_window(
                (s1, s2 - s1, 0, self.sources.shape[3]),
                (t1, t2 - t1, 0, self.target.shape[3]),
                best_target,
            )

    def compare_features_random(self, radius=-1, split=1, times=4):
        """Generate random coordinates within a radius for each pixel, then compare the 
        features to see if the current selection can be improved.
        """

        def _compare(a, b, repro_a, repro_b):
            total = 0
            for (t1, t2) in iterate_range(a.shape[2], split):
                assert t2 >= t1

                if radius == -1:
                    # Generate random grid size (h, w) with indices in range of B.
                    h, w = t2 - t1, a.shape[3]
                    indices = torch.empty(
                        (times, 2, h, w), dtype=torch.int64, device=self.device
                    )
                    repro_a.randgrid(indices, offset=(0, 0), range=b.shape[2:])
                else:
                    indices = repro_a.indices[:, :, t1:t2].clone()
                    indices = indices + torch.empty_like(indices).random_(
                        -radius, radius + 1
                    )

                    indices[:, 0, :, :].clamp_(min=0, max=b.shape[2] - 1)
                    indices[:, 1, :, :].clamp_(min=0, max=b.shape[3] - 1)

                total += self._improve(
                    a, (t1, t2 - t1, 0, a.shape[3]), repro_a, b, indices, repro_b,
                )
            return total

        ts = _compare(self.target, self.sources, self.repro_target, self.repro_sources)
        st = _compare(self.sources, self.target, self.repro_sources, self.repro_target)
        return ts + st

    def compare_features_identity(self, split=1):
        def _compare(a, b, repro_a, repro_b):
            for (t1, t2) in iterate_range(a.shape[2], split):
                assert t2 >= t1

                indices = repro_a.indices[:, :, t1:t2]
                self._update(
                    a, (t1, t2 - t1, 0, a.shape[3]), repro_a, b, indices, repro_b,
                )

        _compare(self.target, self.sources, self.repro_target, self.repro_sources)
        _compare(self.sources, self.target, self.repro_sources, self.repro_target)

    def compare_features_inverse(self, split=1):
        def _compare(a, b, repro_a, repro_b, twice=False):
            total = 0
            for (t1, t2) in iterate_range(a.shape[2], split):
                assert t2 >= t1

                indices = repro_a.indices[:, :, t1:t2]
                total += self._improve(
                    a, (t1, t2 - t1, 0, a.shape[3]), repro_a, b, indices, repro_b,
                )
            return total

        ts = _compare(self.target, self.sources, self.repro_target, self.repro_sources)
        st = _compare(self.sources, self.target, self.repro_sources, self.repro_target)
        return ts + st

    def compare_features_nearby(self, radius, split=1):
        """Generate nearby coordinates for each pixel to see if offseting the neighboring
        pixel would provide better results.
        """
        assert isinstance(radius, int)
        padding = radius

        def _compare(a, b, repro_a, repro_b):
            # Compare all the neighbours from the original position.
            original = repro_a.indices.clone()
            padded_original = torch_pad_reflect(
                original.to(dtype=torch.float32).expand(4, -1, -1, -1),
                (padding, padding, padding, padding),
            ).long()

            total = 0
            for (t1, t2) in iterate_range(a.shape[2], split):
                h, w = (t2 - t1), a.shape[3]

                x = original.new_tensor([0, 0, -radius, +radius]).view(4, 1, 1)
                y = original.new_tensor([-radius, +radius, 0, 0]).view(4, 1, 1)

                # Create a lookup map with offset coordinates from each coordinate.
                lookup = original.new_empty((4, 2, h, w))
                lookup[:, 0, :, :] = torch.arange(
                    t1, t1 + lookup.shape[2], dtype=torch.long
                ).view((1, -1, 1))
                lookup[:, 1, :, :] = torch.arange(
                    0, lookup.shape[3], dtype=torch.long
                ).view((1, 1, -1))
                lookup[:, 0, :, :] += y + padding
                lookup[:, 1, :, :] += x + padding

                # Compute new padded buffer with the current best coordinates.
                indices = padded_original.clone()
                indices[:, 0, :, :] -= y
                indices[:, 1, :, :] -= x

                # Lookup the neighbor coordinates and clamp if the calculation overflows.
                candidates = torch_gather_2d(indices, lookup)

                # Handle `out_of_bounds` by clamping. Could be randomized?
                candidates[:, 0, :, :].clamp_(min=0, max=b.shape[2] - 1)
                candidates[:, 1, :, :].clamp_(min=0, max=b.shape[3] - 1)

                # Update the target window, and the scattered source pixels.
                total += self._improve(
                    a, (t1, t2 - t1, 0, w), repro_a, b, candidates, repro_b,
                )

            return total

        ts = _compare(self.target, self.sources, self.repro_target, self.repro_sources)
        st = _compare(self.sources, self.target, self.repro_sources, self.repro_target)
        return ts + st

    def _compute_similarity(
        self, a_full, window_a, repro_a, b_full, b_indices, repro_b
    ):
        y, dy, x, dx = window_a
        a = a_full[:, :, y : y + dy, x : x + dx]
        b = torch_gather_2d(b_full, b_indices.to(b_full.device))

        similarity = cosine_similarity_vector_1d(a.flatten(2), b.flatten(2))
        similarity += (
            torch_gather_2d(repro_a.biases, b_indices)
            .to(similarity.device)
            .view(similarity.shape[0], -1)
        )
        similarity += (
            repro_b.biases[:, :, y : y + dy, x : x + dx]
            .to(similarity.device)
            .view(1, -1)
        )
        return similarity

    def _update(self, a_full, window_a, repro_a, b_full, b_indices, repro_b):
        y, dy, x, dx = window_a
        similarity = self._compute_similarity(
            a_full, window_a, repro_a, b_full, b_indices, repro_b
        )
        assert similarity.shape[0] == 1
        repro_a.scores[:, :, y : y + dy, x : x + dx] = similarity.view(1, 1, dy, dx)

    def _improve(self, a_full, window_a, repro_a, b_full, b_indices, repro_b):
        similarity = self._compute_similarity(
            a_full, window_a, repro_a, b_full, b_indices, repro_b
        )

        best_candidates = similarity.max(dim=0)
        candidates = torch.gather(
            b_indices.flatten(2),
            dim=0,
            index=best_candidates.indices.to(self.device)
            .view(1, 1, -1)
            .expand(1, 2, -1),
        )

        scores = best_candidates.values.view(1, 1, -1).to(self.device)
        cha = repro_a._improve_window(window_a, scores, candidates.flatten(2))
        chb = repro_b._improve_scatter(candidates.flatten(2), scores, window_a)
        return cha + chb
