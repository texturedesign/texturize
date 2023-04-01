# texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import math

import torch
import torch.nn.functional as F

from .io import load_tensor_from_image
from .app import Result
from .critics import PatchCritic, GramMatrixCritic, HistogramCritic


__all__ = ["Remix", "Mashup", "Enhance", "Expand", "Remake", "Repair"]


def create_default_critics(mode, layers=None):
    if mode == "gram":
        layers = layers or ("1_1", "1_1:2_1", "2_1", "2_1:3_1", "3_1")
    else:
        layers = layers or ("3_1", "2_1", "1_1")

    if mode == "patch":
        return [PatchCritic(layer=l) for l in layers]
    elif mode == "gram":
        return [GramMatrixCritic(layer=l) for l in layers]
    elif mode == "hist":
        return [HistogramCritic(layer=l) for l in layers]


def prepare_default_critics(app, scale, texture, critics):
    texture_cur = F.interpolate(
        texture, scale_factor=1.0 / scale, mode="area", recompute_scale_factor=False,
    ).to(device=app.device, dtype=app.precision)

    layers = [c.get_layers() for c in critics]
    feats = dict(app.encoder.extract_all(texture_cur, layers))
    for critic in critics:
        critic.from_features(feats)
    app.log.debug("<- source:", tuple(texture_cur.shape[2:]), "\n")


class Command:
    def prepare_critics(self, app, scale):
        raise NotImplementedError

    def prepare_seed_tensor(self, size, previous=None):
        raise NotImplementedError

    def finalize_octave(self, result):
        return result


def renormalize(origin, target):
    target_mean = target.mean(dim=(2, 3), keepdim=True)
    target_std = target.std(dim=(2, 3), keepdim=True)
    origin_mean = origin.mean(dim=(2, 3), keepdim=True)
    origin_std = origin.std(dim=(2, 3), keepdim=True)

    result = target_mean + target_std * ((origin - origin_mean) / origin_std)
    return result.clamp(0.0, 1.0)


def upscale(features, size):
    features = F.pad(features, pad=(0, 1, 0, 1), mode='circular')
    features = F.interpolate(features, (size[0]+1, size[1]+1), mode='bilinear', align_corners=True)
    return features[:, :, 0:-1, 0:-1]


def downscale(image, size):
    return F.interpolate(image, size=size, mode="area").clamp(0.0, 1.0)


def random_normal(size, mean):
    b, _, h, w = size
    current = torch.empty((b, 1, h, w), device=mean.device, dtype=torch.float32)
    return (mean + current.normal_(std=0.1)).clamp(0.0, 1.0)


class Remix(Command):
    def __init__(self, source):
        self.source = load_tensor_from_image(source.convert("RGB"), device="cpu")

    def prepare_critics(self, app, scale):
        critics = create_default_critics(app.mode or "hist", app.layers)
        prepare_default_critics(app, scale, self.source, critics)
        return [critics]

    def prepare_seed_tensor(self, app, size, previous=None):
        if previous is None:
            b, _, h, w = size
            mean = self.source.mean(dim=(2, 3), keepdim=True).to(device=app.device)
            result = torch.empty((b, 1, h, w), device=app.device, dtype=torch.float32)
            return (
                (result.normal_(std=0.1) + mean).clamp(0.0, 1.0).to(dtype=app.precision)
            )

        return upscale(previous, size=size[2:])


class Enhance(Command):
    def __init__(self, target, source, zoom=1):
        self.octaves = int(math.log(zoom, 2) + 1.0)
        self.source = load_tensor_from_image(source.convert("RGB"), device="cpu")
        self.target = load_tensor_from_image(target.convert("RGB"), device="cpu")

    def prepare_critics(self, app, scale):
        critics = create_default_critics(app.mode or "hist", app.layers)
        prepare_default_critics(app, scale, self.source, critics)
        return [critics]

    def prepare_seed_tensor(self, app, size, previous=None):
        if previous is not None:
            return upscale(previous, size=size[2:])

        seed = downscale(self.target.to(device=app.device), size=size[2:])
        return renormalize(seed, self.source.to(app.device)).to(dtype=app.precision)


class Remake(Command):
    def __init__(self, target, source, weights=[1.0]):
        self.source = load_tensor_from_image(source.convert("RGB"), device="cpu")
        self.target = load_tensor_from_image(target.convert("RGB"), device="cpu")
        self.weights = torch.tensor(weights, dtype=torch.float32).view(-1, 1, 1, 1)

    def prepare_critics(self, app, scale):
        critics = create_default_critics(app.mode or "hist", app.layers)
        prepare_default_critics(app, scale, self.source, critics)
        return [critics]

    def prepare_seed_tensor(self, app, size, previous=None):
        seed = upscale(self.target.to(device=app.device), size=size[2:])
        return renormalize(seed, self.source.to(app.device)).to(dtype=app.precision)

    def finalize_octave(self, result):
        device = result.images.device
        weights = self.weights.to(device)
        images = result.images.expand(len(self.weights), -1, -1, -1)
        target = self.target.to(device).expand(len(self.weights), -1, -1, -1)
        return Result(images * (weights + 0.0) + (1.0 - weights) * target, *result[1:])


class Repair(Command):
    def __init__(self, target, source):
        assert target.mode == "RGBA"
        self.source = load_tensor_from_image(source.convert("RGB"), device="cpu")
        self.target = load_tensor_from_image(target.convert("RGBA"), device="cpu")

    def prepare_critics(self, app, scale):
        critics = create_default_critics(app.mode or "hist", app.layers)
        # source = renormalize(self.source, self.target[:, 0:3])
        prepare_default_critics(app, scale, self.source, critics)
        return [critics]

    def prepare_seed_tensor(self, app, size, previous=None):
        target = downscale(self.target.to(device=app.device), size=size[2:])
        if previous is None:
            mean = self.source.mean(dim=(2, 3), keepdim=True).to(device=app.device)
            current = random_normal(size, mean).to(
                device=app.device, dtype=app.precision
            )
        else:
            current = upscale(previous, size=size[2:])

        # Use the alpha-mask directly from user.  Could blur it here for better results!
        alpha = target[:, 3:4].detach()
        return torch.cat(
            [target[:, 0:3] * (alpha + 0.0) + (1.0 - alpha) * current, 1.0 - alpha],
            dim=1,
        )


class Expand(Command):
    def __init__(self, target, source, factor=None):
        self.factor = factor or (1.0, 1.0)
        self.source = load_tensor_from_image(source.convert("RGB"), device="cpu")
        self.target = load_tensor_from_image(target.convert("RGB"), device="cpu")

    def prepare_critics(self, app, scale):
        critics = create_default_critics(app.mode or "patch", app.layers)
        prepare_default_critics(app, scale, self.source, critics)
        return [critics]

    def prepare_seed_tensor(self, app, size, previous=None):
        target_size = (int(size[2] / self.factor[0]), int(size[3] / self.factor[1]))
        target = downscale(self.target.to(device=app.device), size=target_size)

        if previous is None:
            mean = self.source.mean(dim=(2, 3), keepdim=True).to(device=app.device)
            current = random_normal(size, mean).to(
                device=app.device, dtype=app.precision
            )
        else:
            current = upscale(previous, size=size[2:])

        start = (size[2] - target_size[0]) // 2, (size[3] - target_size[1]) // 2
        slice_y = slice(start[0], start[0] + target_size[0])
        slice_x = slice(start[1], start[1] + target_size[1])
        current[:, :, slice_y, slice_x,] = target

        # This currently uses a very crisp boolean mask, looks better when edges are
        # smoothed for `overlap` pixels.
        alpha = torch.ones_like(current[:, 0:1])
        alpha[:, :, slice_y, slice_x,] = 0.0
        return torch.cat([current, alpha], dim=1)


class Mashup(Command):
    def __init__(self, sources):
        self.sources = [
            load_tensor_from_image(s.convert("RGB"), device="cpu") for s in sources
        ]

    def prepare_critics(self, app, scale):
        critics = create_default_critics(app.mode or "patch", app.layers)
        all_layers = [c.get_layers() for c in critics]
        sources = [
            F.interpolate(
                img,
                scale_factor=1.0 / scale,
                mode="area",
                recompute_scale_factor=False,
            ).to(device=app.device, dtype=app.precision)
            for img in self.sources
        ]

        # Combine all features into a single dictionary.
        features = [dict(app.encoder.extract_all(f, all_layers)) for f in sources]
        features = dict(zip(features[0].keys(), zip(*[f.values() for f in features])))

        # Initialize the critics from the combined dictionary.
        for critic in critics:
            critic.from_features(features)

        return [critics]

    def prepare_seed_tensor(self, app, size, previous=None):
        if previous is None:
            means = [torch.mean(s, dim=(0, 2, 3), keepdim=True) for s in self.sources]
            mean = (sum(means) / len(means)).to(app.device)
            result = random_normal(size, mean).to(dtype=app.precision)
            return result.to(device=app.device, dtype=app.precision)

        return F.interpolate(
            previous, size=size[2:], mode="bicubic", align_corners=False
        ).clamp(0.0, 1.0)
