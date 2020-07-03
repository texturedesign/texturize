# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import torch
import torch.nn.functional as F

from .io import load_tensor_from_file
from .app import Application
from .critics import PatchCritic


from creativeai.image.encoders import models


class Command:
    def prepare_critics(self, app, scale):
        raise NotImplementedError

    def _prepare_critics(self, app, scale, all_textures, all_critics):
        for (texture_img, critics) in zip(all_textures, all_critics):
            texture_cur = F.interpolate(
                texture_img,
                scale_factor=1.0 / scale,
                mode="area",
                recompute_scale_factor=False,
            ).to(device=app.device, dtype=app.precision)

            feats = dict(
                app.encoder.extract(texture_cur, [c.get_layers() for c in critics])
            )
            for critic in critics:
                critic.from_features(feats)
            app.log.debug("<- texture:", tuple(texture_cur.shape[2:]))

    def prepare_seed_tensor(self, size, previous=None):
        raise NotImplementedError


class Remix(Command):
    def __init__(self, source):
        self.critics = [PatchCritic(layer=l) for l in ("3_1", "2_1", "1_1")]
        self.source = load_tensor_from_file(source, device="cpu")

    def prepare_critics(self, app, scale):
        self._prepare_critics(app, scale, [self.source], [self.critics])
        return [self.critics]

    def prepare_seed_tensor(self, size, previous=None):
        if previous is None:
            b, _, h, w = size
            # TODO: Make the device and precision accessible here.
            mean = self.source.mean(dim=(2, 3), keepdim=True).to("cuda")
            result = torch.empty((b, 1, h, w), device="cuda", dtype=torch.float32)
            result = (
                (result.normal_(std=0.1) + mean)
                .clamp_(0.0, 1.0)
                .to(dtype=torch.float32)
            )
            return result

        return F.interpolate(
            previous, size=size[2:], mode="bicubic", align_corners=False,
        ).clamp_(0.0, 1.0)


class Remake(Command):
    def __init__(self, target, source):
        self.critics = [PatchCritic(layer=l) for l in ("3_1", "2_1", "1_1")]
        self.source = load_tensor_from_file(source, device="cpu")
        self.target = load_tensor_from_file(target, device="cpu")

    def prepare_critics(self, app, scale):
        self._prepare_critics(app, scale, [self.source], [self.critics])
        return [self.critics]

    def prepare_seed_tensor(self, size, previous=None):
        # TODO: Create helper that perform this interpolation modularly, could be decoder.
        seed = F.interpolate(
            self.target, size=size[2:], mode="bicubic", align_corners=False
        )

        # TODO: Create helpers to match two feature maps, in this case by whitening/normalizing.
        source_mean = self.source.mean(dim=(2, 3), keepdim=True)
        source_std = self.source.std(dim=(2, 3), keepdim=True)

        seed_mean = seed.mean(dim=(2, 3), keepdim=True)
        seed_std = seed.std(dim=(2, 3), keepdim=True)

        return (source_mean + source_std * ((seed - seed_mean) / seed_std)).clamp(
            0.0, 1.0
        )

    def process(self, app, *args):
        return super(Remake, self).process(
            app, size=self.target.shape[2:][::-1], octaves=1
        )


class Blend(Command):
    def __init__(self, sources):
        self.critics = {l: PatchCritic(layer=l) for l in ("3_1", "2_1", "1_1")}
        self.sources = [load_tensor_from_file(s, device="cpu") for s in sources]

    def prepare_critics(self, app, scale):
        layers = [c.get_layers() for c in self.critics.values()]

        sources = [
            F.interpolate(
                img,
                scale_factor=1.0 / scale,
                mode="area",
                recompute_scale_factor=False,
            ).to(device="cuda", dtype=torch.float32)
            for img in self.sources
        ]

        for features in zip(*[app.encoder.extract(f, layers) for f in sources]):
            assert features[0][0] == features[1][0]
            layer = features[1][0]

            mix = features[0][1] * 0.5 + 0.5 * features[1][1]
            self.critics[layer].from_features({layer: mix})

        return [list(self.critics.values())]

    def prepare_seed_tensor(self, size, previous=None):
        if previous is None:
            b, _, h, w = size
            mean = (
                torch.cat(self.sources, dim=0)
                .mean(dim=(0, 2, 3), keepdim=True)
                .to("cuda")
            )
            result = torch.empty((b, 1, h, w), device="cuda", dtype=torch.float32)
            result = (
                (result.normal_(std=0.1) + mean)
                .clamp_(0.0, 1.0)
                .to(dtype=torch.float32)
            )
            return result

        return F.interpolate(
            previous, size=size[2:], mode="bicubic", align_corners=False,
        ).clamp_(0.0, 1.0)
