# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import PIL.Image
from torchvision import transforms


def load_image_from_file(filename, device, mode="RGB"):
    loader = transforms.Compose([transforms.ToTensor()])
    image = PIL.Image.open(filename).convert(mode)
    image = loader(image).unsqueeze(0)
    return image.to(device)


def save_image_to_file(image, filename, mode="RGB"):
    img = image.mul(255.0).detach().cpu()
    img = img.squeeze(0).permute(1, 2, 0).numpy().astype("uint8")
    img = PIL.Image.fromarray(img, mode).convert("RGB")
    img.save(filename)
