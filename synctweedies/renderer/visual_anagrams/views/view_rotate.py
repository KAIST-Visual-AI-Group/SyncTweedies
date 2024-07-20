from PIL import Image
import torch

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from .view_base import BaseView


class RotateCWView(BaseView):
    def __init__(self, angle):
        self.angle = angle

    def view(self, im, background=None, **kwargs):
        # TODO: Is nearest-exact better?
        rotated = TF.rotate(im, -self.angle, interpolation=InterpolationMode.BICUBIC)
        if background is not None:
            ones = torch.ones_like(rotated)
            mask =  TF.rotate(ones, -self.angle, interpolation=InterpolationMode.BICUBIC)
            rotated = mask * rotated + (1 - mask) * background
        return rotated

    def inverse_view(self, noise, background=None, **kwargs):
        rotated = TF.rotate(noise, self.angle, interpolation=InterpolationMode.BICUBIC)
        if background is not None:
            ones = torch.ones_like(rotated)
            mask = TF.rotate(ones, self.angle, interpolation=InterpolationMode.BICUBIC) 
            rotated = mask * rotated + (1 - mask) * background
        return rotated

    def make_frame(self, im, t):
        raise NotImplementedError


class ResizeView(BaseView):
    def __init__(self):
        pass
    def view(self, im, **kwargs):
        h, w = im.shape[-2], im.shape[-1]
        x = TF.resize(im, (h*2,w), InterpolationMode.NEAREST)
        x = x[...,:h,:w]
        return x

    def inverse_view(self, noise, **kwargs):
        h, w = noise.shape[-2], noise.shape[-1]
        x = TF.resize(noise, (h//2, w), InterpolationMode.NEAREST)
        zero = torch.zeros_like(x)
        x = torch.cat([x, zero], dim=-2)
        # x = TF.resize(x, (h, w), InterpolationMode.NEAREST)
        return x

class ZeroView(BaseView):
    def __init__(self):
        pass
    def view(self, im, **kwargs):
        return torch.zeros_like(im)
    def inverse_view(self, noise, **kwargs):
        return torch.zeros_like(noise)

###########


class Rotate90CWView(BaseView):
    def __init__(self):
        pass

    def view(self, im, background=None, **kwargs):
        # TODO: Is nearest-exact better?
        rotated = TF.rotate(im, -90, interpolation=InterpolationMode.NEAREST)
        if background is not None:
            ones = torch.ones_like(rotated)
            mask =  TF.rotate(ones, -90, interpolation=InterpolationMode.NEAREST)
            rotated = mask * rotated + (1 - mask) * background
        return rotated

    def inverse_view(self, noise, background=None, **kwargs):
        rotated = TF.rotate(noise, 90, interpolation=InterpolationMode.NEAREST)
        if background is not None:
            ones = torch.ones_like(rotated)
            mask = TF.rotate(ones, 90, interpolation=InterpolationMode.NEAREST) 
            rotated = mask * rotated + (1 - mask) * background
        return rotated

    def make_frame(self, im, t):
        im_size = im.size[0]
        frame_size = int(im_size * 1.5)
        theta = t * -90

        frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
        centered_loc = (frame_size - im_size) // 2
        frame.paste(im, (centered_loc, centered_loc))
        frame = frame.rotate(theta, 
                             resample=Image.Resampling.BILINEAR, 
                             expand=False, 
                             fillcolor=(255,255,255))

        return frame


class Rotate90CCWView(BaseView):
    def __init__(self):
        pass

    def view(self, im, **kwargs):
        # TODO: Is nearest-exact better?
        return TF.rotate(im, 90, interpolation=InterpolationMode.NEAREST)

    def inverse_view(self, noise, **kwargs):
        return TF.rotate(noise, -90, interpolation=InterpolationMode.NEAREST)

    def make_frame(self, im, t):
        im_size = im.size[0]
        frame_size = int(im_size * 1.5)
        theta = t * 90

        frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
        centered_loc = (frame_size - im_size) // 2
        frame.paste(im, (centered_loc, centered_loc))
        frame = frame.rotate(theta, 
                             resample=Image.Resampling.BILINEAR, 
                             expand=False, 
                             fillcolor=(255,255,255))

        return frame


class Rotate180View(BaseView):
    def __init__(self):
        pass

    def view(self, im, **kwargs):
        # TODO: Is nearest-exact better?
        return TF.rotate(im, 180, interpolation=InterpolationMode.NEAREST)

    def inverse_view(self, noise, **kwargs):
        return TF.rotate(noise, -180, interpolation=InterpolationMode.NEAREST)

    def make_frame(self, im, t):
        im_size = im.size[0]
        frame_size = int(im_size * 1.5)
        theta = t * 180

        frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
        centered_loc = (frame_size - im_size) // 2
        frame.paste(im, (centered_loc, centered_loc))
        frame = frame.rotate(theta, 
                             resample=Image.Resampling.BILINEAR, 
                             expand=False, 
                             fillcolor=(255,255,255))

        return frame
