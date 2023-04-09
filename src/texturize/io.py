# texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import glob
import time
import random
import urllib
import difflib
from io import BytesIO

import PIL.Image
import torch
import torchvision.transforms.functional as V


def load_tensor_from_files(glob_pattern, device='cpu', mode=None) -> tuple:
    arrays, props = [], []
    for filename in sorted(glob.glob(glob_pattern)):
        img = load_image_from_file(filename, mode)
        arr = load_tensor_from_image(img, device)
        arrays.append(arr)

        prop = ''.join([s[2] for s in difflib.ndiff(glob_pattern, filename) if s[0]=='+'])
        props.append(prop + ":" + str(arr.shape[1]))

    assert all(a.shape[2:] == arrays[0].shape[2:] for a in arrays[1:])
    return torch.cat(arrays, dim=1), props


def save_tensor_to_files(images, filename, props):
    c = 0
    for prop in props:
        suffix, i = prop.split(":")
        i = int(i)
        img = images[:, c:c+i]
        save_tensor_to_file(img, filename.replace('{prop}', suffix), mode="RGB" if i==3 else "L")
        c += i


def load_image_from_file(filename, mode=None):
    image = PIL.Image.open(filename)
    if mode is not None:
        return image.convert(mode)
    else:
        return image


def load_tensor_from_image(image, device, dtype=torch.float32):
    # NOTE: torchvision incorrectly assumes that I;16 means signed, but
    # Pillow docs say unsigned.
    if isinstance(image, PIL.Image.Image) and image.mode == "I;16":
        import numpy
        arr = numpy.frombuffer(image.tobytes(), dtype=numpy.uint16)
        arr = arr.reshape((image.height, image.width))
        assert arr.min() >= 0 and arr.max() < 65536
        image = arr.astype(numpy.float32) / 65536.0

    if not isinstance(image, torch.Tensor):
        return V.to_tensor(image).unsqueeze(0).to(device, dtype)
    return image


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
    img = save_tensor_to_images(tensor, mode=mode)
    img[0].save(filename)


def save_tensor_to_images(tensor, mode="RGB"):
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0
    return [
        V.to_pil_image(tensor[j].detach().cpu().float(), mode)
        for j in range(tensor.shape[0])
    ]


try:
    from IPython.display import display, clear_output
    import ipywidgets
except ImportError:
    pass


def show_image_as_tiles(image, count, size):
    def make_crop():
        buffer = BytesIO()
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
            assert len(result.tensor) == 1, "Only one image supported."

            for out in save_tensor_to_images(result.tensor[:, 0:3]):
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

                buffer = BytesIO()
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
