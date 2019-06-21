# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch
import torchvision.transforms as transforms
import PIL.Image


def load_from_file(filename, device, mode="RGB"):
    loader = transforms.Compose([transforms.ToTensor()])
    image = PIL.Image.open(filename).convert(mode)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float) * 4.0 - 2.0


def save_to_file(image, filename, mode="RGB"):
    img = (image * 0.25 + 0.5).mul(255.0).clamp(0, 255.0).detach().cpu().numpy()
    img = img[0].transpose(1, 2, 0).astype("uint8")
    img = PIL.Image.fromarray(img, mode).convert("RGB")
    img.save(filename)
