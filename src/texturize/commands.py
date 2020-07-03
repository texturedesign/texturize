# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import torch
import torch.nn.functional as F

from .io import load_tensor_from_file
from .app import Application
from .critics import PatchCritic, GramMatrixCritic, HistogramCritic


def create_default_critics(mode):
    if mode == "gram":
        layers = ("1_1", "1_1:2_1", "2_1", "2_1:3_1", "3_1")
    else:
        layers = ("3_1", "2_1", "1_1")

    if mode == "patch":
        return {l: PatchCritic(layer=l) for l in layers}
    elif mode == "gram":
        return {l: GramMatrixCritic(layer=l) for l in layers}
    elif mode == "hist":
        return {l: HistogramCritic(layer=l) for l in layers}


class Command:
    def prepare_critics(self, app, scale):
        raise NotImplementedError

    def _prepare_critics(self, app, scale, texture, critics):
        texture_cur = F.interpolate(
            texture,
            scale_factor=1.0 / scale,
            mode="area",
            recompute_scale_factor=False,
        ).to(device=app.device, dtype=app.precision)

        layers = [c.get_layers() for c in critics]
        feats = dict(app.encoder.extract(texture_cur, layers))
        for critic in critics:
            critic.from_features(feats)
        app.log.debug("<- texture:", tuple(texture_cur.shape[2:]))

    def prepare_seed_tensor(self, size, previous=None):
        raise NotImplementedError


class Remix(Command):
    def __init__(self, source, mode):
        self.critics = list(create_default_critics(mode).values())
        self.source = load_tensor_from_file(source, device="cpu")

    def prepare_critics(self, app, scale):
        self._prepare_critics(app, scale, self.source, self.critics)
        return [self.critics]

    def prepare_seed_tensor(self, app, size, previous=None):
        if previous is None:
            b, _, h, w = size
            mean = self.source.mean(dim=(2, 3), keepdim=True).to(device=app.device)
            result = torch.empty((b, 1, h, w), device=app.device, dtype=torch.float32)
            return (
                (result.normal_(std=0.1) + mean).clamp(0.0, 1.0).to(dtype=app.precision)
            )

        return F.interpolate(
            previous, size=size[2:], mode="bicubic", align_corners=False,
        ).clamp_(0.0, 1.0)


class Remake(Command):
    def __init__(self, target, source, mode):
        self.critics = list(create_default_critics(mode).values())
        self.source = load_tensor_from_file(source, device="cpu")
        self.target = load_tensor_from_file(target, device="cpu")

    def prepare_critics(self, app, scale):
        self._prepare_critics(app, scale, self.source, self.critics)
        return [self.critics]

    def prepare_seed_tensor(self, app, size, previous=None):
        seed = F.interpolate(
            self.target.to(device=app.device),
            size=size[2:],
            mode="bicubic",
            align_corners=False,
        )

        source_mean = self.source.mean(dim=(2, 3), keepdim=True)
        source_std = self.source.std(dim=(2, 3), keepdim=True)
        seed_mean = seed.mean(dim=(2, 3), keepdim=True)
        seed_std = seed.std(dim=(2, 3), keepdim=True)

        result = source_mean + source_std * ((seed - seed_mean) / seed_std)
        return result.clamp(0.0, 1.0).to(dtype=app.precision)

    def process(self, app, *args):
        return super(Remake, self).process(
            app, size=self.target.shape[2:][::-1], octaves=1
        )


class Blend(Command):
    def __init__(self, sources, mode):
        self.critics = create_default_critics(mode)
        self.sources = [load_tensor_from_file(s, device="cpu") for s in sources]

    def prepare_critics(self, app, scale):
        layers = [c.get_layers() for c in self.critics.values()]

        sources = [
            F.interpolate(
                img,
                scale_factor=1.0 / scale,
                mode="area",
                recompute_scale_factor=False,
            ).to(device=app.device, dtype=app.precision)
            for img in self.sources
        ]

        for features in zip(*[app.encoder.extract(f, layers) for f in sources]):
            assert features[0][0] == features[1][0]
            layer = features[1][0]

            mix = features[0][1] * 0.5 + 0.5 * features[1][1]
            self.critics[layer].from_features({layer: mix})

        return [list(self.critics.values())]

    def prepare_seed_tensor(self, app, size, previous=None):
        if previous is None:
            b, _, h, w = size
            mean = (
                torch.cat(self.sources, dim=0)
                .mean(dim=(0, 2, 3), keepdim=True)
                .to("cuda")
            )
            result = torch.empty((b, 1, h, w), device=app.device, dtype=torch.float32)
            result = (result.normal_(std=0.1) + mean).clamp(0.0, 1.0)
            return result.to(dtype=app.precision)

        return F.interpolate(
            previous, size=size[2:], mode="bicubic", align_corners=False
        ).clamp(0.0, 1.0)
