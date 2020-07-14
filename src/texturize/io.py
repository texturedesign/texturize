# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import time
import random
import urllib
import asyncio
from io import BytesIO

import PIL.Image
import torch
import torchvision.transforms.functional as V


def load_tensor_from_file(filename, device, mode="RGB"):
    image = load_image_from_file(filename, mode)
    return load_tensor_from_image(image, device)


def load_image_from_file(filename, mode="RGB"):
    return PIL.Image.open(filename).convert(mode)


def load_tensor_from_image(image, device, dtype=torch.float32):
    return V.to_tensor(image).unsqueeze(0).to(device, dtype)


def load_image_from_url(url, mode="RGB"):
    response = urllib.request.urlopen(url)
    buffer = BytesIO(response.read())
    return PIL.Image.open(buffer).convert(mode)


def random_crop(image, size):
    x = random.randint(0, image.size[0] - size[0])
    y = random.randint(0, image.size[1] - size[1])
    return image.crop((x, y, x + size[0], y + size[1]))


def save_tensor_to_file(tensor, filename, mode="RGB"):
    assert tensor.shape[0] == 1
    img = save_tensor_to_images(tensor)
    img[0].save(filename)


def save_tensor_to_images(tensor, mode="RGB"):
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0
    return [
        V.to_pil_image(tensor[j].detach().cpu().float(), mode)
        for j in range(tensor.shape[0])
    ]


try:
    import io
    from IPython.display import display, clear_output
    import ipywidgets
except ImportError:
    pass


def show_image_as_tiles(image, count, size):
    def make_crop():
        buffer = io.BytesIO()
        x = random.randint(0, image.size[0] - size[0])
        y = random.randint(0, image.size[1] - size[1])
        tile = image.crop((x, y, x + size[0], y + size[1]))
        tile.save(buffer, format="webp", quality=80)
        buffer.seek(0)
        return buffer.read()

    pct = 100.0 / count
    tiles = [
        ipywidgets.Image(
            value=make_crop(), format="webp", layout=ipywidgets.Layout(width=f"{pct}%")
        )
        for _ in range(count)
    ]
    box = ipywidgets.HBox(tiles, layout=ipywidgets.Layout(width="100%"))
    display(box)


def show_result_in_notebook(throttle=None, title=None):
    class ResultWidget:
        def __init__(self, throttle, title):
            self.title = f"<h3>{title}</h3>" if title is not None else ""
            self.style = """<style>
                    ul.statistics li { float: left; width: 48%; }
                    ul.statistics { font-size: 16px; }
                </style>"""
            self.html = ipywidgets.HTML(value="")
            self.img = ipywidgets.Image(
                value=b"",
                format="webp",
                layout=ipywidgets.Layout(width="100%", margin="0"),
            )
            self.box = ipywidgets.VBox(
                [self.html, self.img], layout=ipywidgets.Layout(display="none")
            )
            display(self.box)

            self.throttle = throttle
            self.start_time = time.time()
            self.total_sent = 0

        def update(self, result):
            assert len(result.images) == 1, "Only one image supported."

            for out in save_tensor_to_images(result.images):
                elapsed = time.time() - self.start_time
                last, first = bool(result.iteration < 0), bool(result.iteration == 0)
                self.html.set_trait(
                    "value",
                    f"""{self.title}
                        {self.style}
                    <ul class="statistics">
                        <li>scale: 1/{result.scale}</li>
                        <li>loss: {result.loss:0.3e}</li>
                        <li>size: {out.size}</li>
                        <li>rate: {result.rate:0.3e}</li>
                        <li>octave: {result.octave}</li>
                        <li>retries: {result.retries}</li>
                        <li>elapsed: {int(elapsed)}s</li>
                        <li>iteration: {abs(result.iteration)}</li>
                    </ul>""",
                )

                if not last and self.total_sent / elapsed > self.throttle:
                    break

                buffer = io.BytesIO()
                if throttle == float("+inf") or out.size[0] * out.size[1] < 192 * 192:
                    out.save(buffer, format="webp", method=6, lossless=True)
                else:
                    out.save(buffer, format="webp", quality=90 if last else 50)

                buffer.seek(0)
                self.img.set_trait("value", buffer.read())
                self.total_sent += buffer.tell()
                if first:
                    self.box.layout = ipywidgets.Layout(display="box")
                break

    try:
        # Notebooks running remotely on Google Colab require throttle to work reliably.
        import google.colab
        throttle = throttle or 16_384
    except ImportError:
        # When running Jupyter locally, you get the full experience by default!
        throttle = throttle or float('+inf')

    return ResultWidget(throttle, title)


def load_image_from_notebook():
    """Allow the user to upload an image directly into a Jupyter notebook, then provide
    a single-use iterator over the images that were collected.
    """

    class ImageUploadWidget(ipywidgets.FileUpload):
        def __init__(self):
            super(ImageUploadWidget, self).__init__(accept="image/*", multiple=True)

            self.observe(self.add_to_results, names="value")
            self.results = []

        def get(self, index):
            return self.results[index]

        def __iter__(self):
            while len(self.results) > 0:
                yield self.results.pop(0)

        def add_to_results(self, change):
            for filename, data in change["new"].items():
                buffer = BytesIO(data["content"])
                image = PIL.Image.open(buffer)
                self.results.append(image)
            self.set_trait("value", {})

    widget = ImageUploadWidget()
    display(widget)
    return widget
