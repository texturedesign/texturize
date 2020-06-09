# neural-texturize — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import PIL.Image
import torchvision.transforms.functional as V


def load_tensor_from_file(filename, device, mode="RGB"):
    image = load_image_from_file(filename, mode)
    return load_tensor_from_image(image, device)


def load_image_from_file(filename, mode="RGB"):
    return PIL.Image.open(filename).convert(mode)


def load_tensor_from_image(image, device):
    return V.to_tensor(image).unsqueeze(0).to(device)


def save_tensor_to_file(tensor, filename, mode="RGB"):
    img = save_tensor_to_image(tensor)
    img.save(filename)


def save_tensor_to_image(tensor, mode="RGB"):
    return V.to_pil_image(tensor[0].detach().cpu().float(), mode)
