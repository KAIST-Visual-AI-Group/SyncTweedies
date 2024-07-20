from math import pi, tan, atan, sin, cos
import os
import sys
import json

import numpy as np
import torch

from .gaussian_utils import CameraSet, generate_camera, get_intrinsics
from .colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary
from .gaussian_utils import focal_length_to_fov, fov_to_focal_length, convert_camera_convention

def load_cameras(dataset_type, size=[512,512], source_path=None, up_vec='z', convention='OpenCV', dist=2.0, fov=60):
    if dataset_type == "dense":
        # gradually increase and decrease elevation 0 -> 60 -> 0
        # uniformly sample azimuth 0 -> 360
        camera_num = 50
        elevs = []
        azims = []
        for i in range(camera_num):
            elev = 30 * (1 - cos(i / camera_num * 2 * pi))
            azim = i * 2 * 360 / camera_num
            elevs.append(elev)
            azims.append(azim)

        cameraset = CameraSet()
        for elev, azim in zip(elevs, azims):
            cam = generate_camera(dist, elev, azim, fov, size[0], size[1], up_vec=up_vec, convention=convention)
            cameraset.append(**cam)
    elif dataset_type == "sparse":
        azims = [-180, -135, -90, -45, 0, 45, 90, 135, 0, 180]
        elevs = [0, 0, 0, 0, 0, 0, 0, 0, 30, 30]

        cameraset = CameraSet()
        for azim, elev in zip(azims, elevs):
            cam = generate_camera(dist, elev, azim, fov, size[0], size[1], up_vec=up_vec, convention=convention)
            cameraset.append(**cam)
    elif dataset_type == "rendering":
        camera_num = 50
        elevs = [30] * camera_num
        azims = [i * 360 / camera_num for i in range(camera_num)]

        cameraset = CameraSet()
        for elev, azim in zip(elevs, azims):
            cam = generate_camera(dist, elev, azim, fov, size[0], size[1], up_vec=up_vec, convention=convention)
            cameraset.append(**cam)
    elif dataset_type == "evaluation":
        rng = np.random.default_rng(0)
        azims = rng.uniform(-180, 180, 250)
        elevs = rng.uniform(20, 50, 250)

        cameraset = CameraSet()
        for elev, azim in zip(elevs, azims):
            cam = generate_camera(dist, elev, azim, fov, size[0], size[1], up_vec=up_vec, convention=convention)
            cameraset.append(**cam)
    elif dataset_type == "colmap":
        assert source_path is not None, "source_path must be provided for colmap dataset"
        C2Ws, Ks = load_colmap_cameras(source_path, size)
        cameraset = CameraSet(C2Ws, Ks, size[0], size[1])
    elif dataset_type == "blender":
        assert source_path is not None, "source_path must be provided for blender dataset"
        C2Ws, Ks = load_blender_cameras(source_path, size)

        cameraset = CameraSet(C2Ws, Ks, size[0], size[1])
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return cameraset
    
def load_colmap_cameras(path, target_size=[512, 512]):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    C2Ws = []
    Ks = []
    for idx, key in enumerate(cam_extrinsics):

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        target_height, target_width = target_size

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal_length_to_fov(focal_length_x, height/2)
        elif intr.model=="PINHOLE":
            focal_length_y = intr.params[1]
            FovY = focal_length_to_fov(focal_length_y, width/2)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        c2w = np.zeros((4, 4))
        c2w[:3, :3] = R
        c2w[:3, 3] = -R @ T
        c2w[3, 3] = 1.0

        k = get_intrinsics(FovY, target_height, target_width, invert_y=False)
        C2Ws.append(c2w)
        Ks.append(k)
    C2Ws = np.stack(C2Ws, axis=0)
    Ks = np.stack(Ks, axis=0)

    C2Ws = torch.from_numpy(C2Ws).float()
    Ks = torch.from_numpy(Ks).float()
    return C2Ws, Ks

def load_blender_cameras(path, target_size=[512, 512]):
    train_cam_path = os.path.join(path, "transforms_test.json")
    target_height, target_width = target_size

    C2Ws = []
    Ks = []
    with open(train_cam_path) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        frames = contents["frames"]

        for frame in frames:
            c2w = np.array(frame["transform_matrix"])
            FovY = focal_length_to_fov(fov_to_focal_length(fovx, target_width/2), target_height/2)
            k = get_intrinsics(FovY, target_height, target_width, invert_y=False)

            C2Ws.append(c2w)
            Ks.append(k)
    C2Ws = np.stack(C2Ws, axis=0)
    C2Ws = convert_camera_convention(C2Ws, "OpenGL", "OpenCV")
    Ks = np.stack(Ks, axis=0)

    C2Ws = torch.from_numpy(C2Ws).float()
    Ks = torch.from_numpy(Ks).float()
            
    return C2Ws, Ks