import sys
from math import pi, tan, atan
from typing import Literal

import numpy as np
import torch


class CameraSet:
    def __init__(self, C2Ws=None, Ks=None, height=None, width=None):
        self.C2Ws = C2Ws
        self.Ks = Ks
        self.height = height
        self.width = width
        
        if self.C2Ws is not None and self.C2Ws.dim() == 2:
            self.C2Ws = self.C2Ws.unsqueeze(0)
        if self.Ks is not None and self.Ks.dim() == 2:
            self.Ks = self.Ks.unsqueeze(0)

    def is_empty(self):
        return self.C2Ws is None or self.Ks is None or self.height is None or self.width is None

    def __getitem__(self, key):
        if self.is_empty():
            return None
        
        sliced_data = {
            "C2Ws": self.C2Ws[key],
            "Ks": self.Ks[key],
            "height": self.height,
            "width": self.width,
        }
        return CameraSet(
            sliced_data["C2Ws"],
            sliced_data["Ks"],
            sliced_data["height"],
            sliced_data["width"],
        )
    
    def append(self, C2Ws, Ks, height=None, width=None):
        assert self.height == height or self.height is None or height is None, f"Height mismatch: {self.height} != {height}"
        assert self.width == width or self.width is None or width is None, f"Width mismatch: {self.width} != {width}"
        self.height = height or self.height
        self.width = width or self.width

        if self.C2Ws is None:
            self.C2Ws = C2Ws
            self.Ks = Ks
        else:
            device = self.C2Ws.device
            self.C2Ws = torch.cat([self.C2Ws, C2Ws.to(device)], dim=0)
            self.Ks = torch.cat([self.Ks, Ks.to(device)], dim=0)

    def insert(self, index, C2Ws, Ks):
        if self.C2Ws is None:
            self.C2Ws = C2Ws
            self.Ks = Ks
            device = self.C2Ws.device
            self.C2Ws = torch.cat(
                [self.C2Ws[:index], C2Ws.to(device), self.C2Ws[index:]],
                dim=0,
            )
            self.Ks = torch.cat(
                [self.Ks[:index], Ks.to(device), self.Ks[index:]],
                dim=0,
            )

    def __len__(self):
        return self.C2Ws.size(0) if self.C2Ws is not None else 0

    def __repr__(self):
        return f"CameraSet(C2Ws={self.C2Ws.size()}, Ks={self.Ks.size()}, height={self.height}, width={self.width})"
    
    def to(self, device):
        self.C2Ws = self.C2Ws.to(device)
        self.Ks = self.Ks.to(device)
        return self


def fov_to_focal_length(fov, hole_rad=0.5):
    return hole_rad / tan(fov / 2)


def focal_length_to_fov(focal_length, hole_rad=0.5):
    return 2 * atan(hole_rad / focal_length)


def dfov_to_focal_length(dfov):
    fov = dfov * pi / 180
    return 0.5 / tan(fov / 2)


def focal_length_to_dfov(focal_length):
    fov = 2 * atan(0.5 / focal_length)
    return fov * 180 / pi


def get_intrinsics(fov, height, width, invert_y=False):
    """
    Generates the camera intrinsic matrix based on field of view (fov), image height, and width.

    Args:
        fov (float): Field of view in radians.
        height (int): Height of the image in pixels.
        width (int): Width of the image in pixels.

    Returns:
        np.ndarray: The camera intrinsic matrix.
    """
    focal_length = fov_to_focal_length(fov)
    fx = focal_length * width
    fy = focal_length * height
    cx = width / 2.0
    cy = height / 2.0

    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    if invert_y:
        intrinsics[1, 1] *= -1

    return intrinsics


def generate_camera_params(
    dist,
    elev,
    azim,
    fov,
    height,
    width,
    up_vec: Literal["x", "y", "z", "-x", "-y", "-z"] = "z",
    convention: Literal[
        "LUF", "RDF", "RUB", "RUF", "Pytorch3D", "OpenCV", "OpenGL", "Unity"
    ] = "OpenCV",
):
    # If input is tensor, convert to float
    if isinstance(dist, torch.Tensor):
        dist = dist.item()
    if isinstance(elev, torch.Tensor):
        elev = elev.item()
    if isinstance(azim, torch.Tensor):
        azim = azim.item()

    elev = elev * np.pi / 180
    azim = azim * np.pi / 180
    fov = fov * np.pi / 180

    world_up, world_front, world_right = {
        "x": (np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0])),
        "y": (np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([0, 0, 1])),
        "z": (np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([1, 0, 0])),
        "-x": (np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])),
        "-y": (np.array([0, -1, 0]), np.array([0, 0, 1]), np.array([1, 0, 0])),
        "-z": (np.array([0, 0, -1]), np.array([1, 0, 0]), np.array([0, 1, 0])),
    }[up_vec]

    cam_pos = dist * (
        world_up * np.sin(elev)
        - world_front * np.cos(elev) * np.cos(azim)
        + world_right * np.cos(elev) * np.sin(azim)
    )

    lookat = -cam_pos / np.linalg.norm(cam_pos)
    fake_up = world_up

    right = np.cross(lookat, fake_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, lookat)

    invert_y = False
    if convention == "LUF" or convention == "Pytorch3D":
        R = np.stack([-right, up, lookat], axis=1)
        invert_y = True
    if convention == "RDF" or convention == "OpenCV":
        R = np.stack([right, -up, lookat], axis=1)
        invert_y = False
    elif convention == "RUB" or convention == "OpenGL":
        R = np.stack([right, up, -lookat], axis=1)
        invert_y = True
    elif convention == "RUF" or convention == "Unity":
        R = np.stack([right, up, lookat], axis=1)
        invert_y = True

    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = cam_pos

    focal_length = fov_to_focal_length(fov)
    fx = focal_length * width
    fy = focal_length * height
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    if invert_y:
        K[1, 1] *= -1
    
    return c2w, K

CAM_TRANSFORMATIONS = {
    "RUF": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    "Unity": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),

    "Pytorch3D": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    "LUF": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),

    "OpenGL": np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    "RUB": np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),

    "OpenCV": np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    "RDF": np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
}

def convert_camera_convention(C2Ws, from_convention, to_convention):
    # C2Ws: B x 4 x 4
    # from_convention, to_convention: str
    # returns: B x 4 x 4

    from_ruf = CAM_TRANSFORMATIONS[from_convention]
    ruf_to = CAM_TRANSFORMATIONS[to_convention]
    from_to = np.linalg.inv(from_ruf) @ ruf_to
    new_C2Ws = C2Ws.copy()
    new_C2Ws[:, :3, :3] = new_C2Ws[:, :3, :3] @ from_to
    return new_C2Ws


def generate_camera(
    dist,
    elev,
    azim,
    fov,
    height,
    width,
    up_vec: Literal["x", "y", "z", "-x", "-y", "-z"] = "z",
    convention: Literal[
        "LUF", "RDF", "RUB", "RUF", "Pytorch3D", "OpenCV", "OpenGL", "Unity"
    ] = "RDF",
    device="cpu",
):
    c2w, K = generate_camera_params(
        dist,
        elev,
        azim,
        fov,
        height,
        width,
        up_vec=up_vec,
        convention=convention,
    )
    c2w = torch.tensor(c2w, dtype=torch.float32).to(device)
    K = torch.tensor(K, dtype=torch.float32).to(device)

    return {
        "C2Ws": c2w.unsqueeze(0),
        "Ks": K.unsqueeze(0),
        "height": height,
        "width": width,
        #"azimuth": [azim],
        #"elevation": [elev],
    }