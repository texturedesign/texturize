# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import urllib
import asyncio
from io import BytesIO

import PIL.Image
import torchvision.transforms.functional as V


def load_tensor_from_file(filename, device, mode="RGB"):
    image = load_image_from_file(filename, mode)
    return load_tensor_from_image(image, device)


def load_image_from_file(filename, mode="RGB"):
    return PIL.Image.open(filename).convert(mode)


def load_tensor_from_image(image, device):
    return V.to_tensor(image).unsqueeze(0).to(device)


def load_image_from_url(url, mode="RGB"):
    response = urllib.request.urlopen(url)
    buffer = BytesIO(response.read())
    return PIL.Image.open(buffer).convert(mode)


def save_tensor_to_file(tensor, filename, mode="RGB"):
    img = save_tensor_to_image(tensor)
    img.save(filename)


def save_tensor_to_image(tensor, mode="RGB"):
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0
    return V.to_pil_image(tensor[0].detach().cpu().float(), mode)


try:
    import io
    from IPython.display import display, clear_output
    import ipywidgets
except ImportError:
    pass


def show_result_in_notebook(result, title="Generated Image"):
    clear_output()

    for out in result.images:
        html = ipywidgets.HTML(value=f"""
        <h3>{title}</h3>
        <ul style="font-size: 16px;">
            <li>octave: {result.octave}</li>
            <li>size: {out.size}</li>
            <li>scale: 1/{result.scale}</li>
            <li>loss: {result.loss:0.4f}</li>
        </ul>""")

        buffer = io.BytesIO()
        out.save(buffer, format="webp", quality=80)
        buffer.seek(0)

        img = ipywidgets.Image(
            value=buffer.read(),
            format="webp",
            layout=ipywidgets.Layout(width="100%", margin="0"),
        )
        box = ipywidgets.VBox([html, img])
        display(box)


def load_image_from_notebook():
    """Allow the user to upload an image directly into a Jupyter notebook, then provide
    a single-use iterator over the images that were collected.
    """

    class ImageUpload(ipywidgets.FileUpload):
        def __init__(self):
            super(ImageUpload, self).__init__(accept="image/*", multiple=True)

            self.observe(self.add_to_results, names='value')
            self.results = []

        def get(self):
            return self.results.pop(0)

        def __iter__(self):
            while len(self.results) > 0:
                yield self.get()

        def add_to_results(self, change):
            for filename, data in change['new'].items():
                buffer = BytesIO(data['content'])
                image = PIL.Image.open(buffer)
                self.results.append(image)
            self.set_trait('value', {})

    widget = ImageUpload()
    display(widget)
    return widget
