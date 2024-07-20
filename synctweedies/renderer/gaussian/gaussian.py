import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from .gsplat.gsplat.rendering import rasterization
from .gsplat.gsplat.rendering_pcd import rasterization as rasterization_pcd

from dataclasses import dataclass


MAX_BATCH_SIZE = 10
def render(gs, camera_set, rasterize_mode="classic", render_mode="RGB", override_color=None, backgrounds=None, sqrt_mode=False):
    """
    Render the splats to an image.
    """
    c2ws, Ks = camera_set.C2Ws, camera_set.Ks
    assert c2ws.shape[0] == Ks.shape[0], f"{c2ws.shape}, {Ks.shape}"
    assert c2ws.shape[1:] == (4, 4)
    assert Ks.shape[1:] == (3, 3)

    means = gs.xyz                           # [N, 3]
    quats = gs.quats                         # [N, 4]
    scales = torch.exp(gs.scales)            # [N, 3]
    opacities = torch.sigmoid(gs.opacities)  # [N,]
    if override_color is not None:
        colors = override_color
    else:
        colors = torch.sigmoid(gs.features)      # [N, D]

    # Split the splats into batches
    num_batches = len(c2ws)
    batches = []
    bg_batches = []
    for i in range(0, num_batches, MAX_BATCH_SIZE):
        start = i
        end = min(i + MAX_BATCH_SIZE, num_batches)
        batch = camera_set[start:end]
        batches.append(batch)
        if backgrounds is not None:
            bg_batches.append(backgrounds[start:end])
        else:
            bg_batches.append(None)
    
    results = []
    for batch, bg in zip(batches, bg_batches):
        if render_mode == "PCD":
            render_colors, render_alphas, _ = rasterization_pcd(
                means=means,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(batch.C2Ws),  # [B, 4, 4]
                Ks=batch.Ks,  # [B, 3, 3]
                width=batch.width,
                height=batch.height,
                packed=False,
                backgrounds=bg,
                sqrt_mode=sqrt_mode,
            )
        else:
            render_colors, render_alphas, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(batch.C2Ws),  # [B, 4, 4]
                Ks=batch.Ks,  # [B, 3, 3]
                width=batch.width,
                height=batch.height,
                packed=False,
                backgrounds=bg,
                absgrad=True,
                sparse_grad=False,
                rasterize_mode=rasterize_mode,
                render_mode=render_mode,
            )
        results.append((render_colors, render_alphas))
    
    render_colors = torch.cat([r[0] for r in results], dim=0)
    render_alphas = torch.cat([r[1] for r in results], dim=0)

    return {
        "image": render_colors.permute(0, 3, 1, 2),
        "alpha": render_alphas.permute(0, 3, 1, 2),
    }

class GSModel:
    """Model and Renderer for Gaussian splats."""

    @dataclass
    class Config:
        plyfile: Optional[str] = None
        device: str = "cuda"
        batch_size: int = 1

        xyz_lr: float = 3e-4
        scale_lr: float = 0.005
        quat_lr: float = 0.001
        opacity_lr: float = 0.05
        feature_lr: float = 0.025

    def __init__(self, args=None):
        super().__init__()
        self.cfg = self.Config(**args)
        self.optimizers = None

        self.xyz = torch.empty(0)
        self.scales = torch.empty(0)
        self.quats = torch.empty(0)
        self.opacities = torch.empty(0)
        self.features = torch.empty(0)

        if self.cfg.plyfile is not None:
            self.load(self.cfg.plyfile)

    @property
    def device(self):
        return self.cfg.device

    def __len__(self):
        return len(self.xyz)
    
    def clone(self):
        cfg = vars(self.cfg)
        cfg["plyfile"] = None
        new_model = GSModel(vars(self.cfg))
        new_model.xyz = self.xyz.clone().detach()
        new_model.scales = self.scales.clone().detach()
        new_model.quats = self.quats.clone().detach()
        new_model.opacities = self.opacities.clone().detach()
        new_model.features = self.features.clone().detach()
        return new_model

    def load(self, path):
        """
        Load splats from a PLY file.
        Properties:
        - xyz: [N, 3] tensor of point locations
        - scales: [N, 3] tensor of scale factors
        - quats: [N, 4] tensor of quaternions
        - opacities: [N,] tensor of opacities
        - features: [N, D] tensor of feature vectors
        """
        ply_data = PlyData.read(path)

        points = np.vstack(
            [ply_data["vertex"]["x"], ply_data["vertex"]["y"], ply_data["vertex"]["z"]]
        ).T
        scales = np.vstack(
            [
                ply_data["vertex"]["scale_0"],
                ply_data["vertex"]["scale_1"],
                ply_data["vertex"]["scale_2"],
            ]
        ).T
        quats = np.vstack(
            [
                ply_data["vertex"]["rot_0"],
                ply_data["vertex"]["rot_1"],
                ply_data["vertex"]["rot_2"],
                ply_data["vertex"]["rot_3"],
            ]
        ).T
        opacities = ply_data["vertex"]["opacity"]

        # Extract features
        feature_columns = [
            col
            for col in ply_data["vertex"].data.dtype.names
            if col.startswith("f_dc_")
        ]
        features = np.vstack([ply_data["vertex"][col] for col in feature_columns]).T

        self.xyz = torch.tensor(points, dtype=torch.float32).to(self.device)
        self.scales = torch.tensor(scales, dtype=torch.float32).to(self.device)
        self.quats = torch.tensor(quats, dtype=torch.float32).to(self.device)
        self.opacities = torch.tensor(opacities, dtype=torch.float32).to(self.device)
        self.features = torch.tensor(features, dtype=torch.float32).to(self.device)

        self.prepare_optimizer()

    def save(self, path):
        """
        Save splats to a PLY file.
        :param path: Path to the PLY file.
        """
        xyz = self.xyz.cpu().detach().numpy()
        scales = self.scales.cpu().detach().numpy()
        quats = self.quats.cpu().detach().numpy()
        opacities = self.opacities.cpu().detach().numpy()
        features = self.features.cpu().detach().numpy()

        # Create feature columns
        feature_columns = [f"f_dc_{i}" for i in range(features.shape[1])]

        vertex_data = np.array(
            [
                tuple(
                    np.concatenate(
                        (xyz[i], scales[i], quats[i], [opacities[i]], features[i])
                    )
                )
                for i in range(xyz.shape[0])
            ],
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("scale_0", "f4"),
                ("scale_1", "f4"),
                ("scale_2", "f4"),
                ("rot_0", "f4"),
                ("rot_1", "f4"),
                ("rot_2", "f4"),
                ("rot_3", "f4"),
                ("opacity", "f4"),
                *[(name, "f4") for name in feature_columns],
            ],
        )

        el = PlyElement.describe(vertex_data, "vertex")
        PlyData([el]).write(path)

    
    
    def prepare_optimizer(self, parameters=None):
        batch_size = self.cfg.batch_size

        lr_dict = [
            # name, lr
            ("xyz", self.cfg.xyz_lr),
            ("scales", self.cfg.scale_lr),
            ("quats", self.cfg.quat_lr),
            ("opacities", self.cfg.opacity_lr),
            ("features", self.cfg.feature_lr),
        ]
        splat_dict = {
            "xyz": self.xyz,
            "scales": self.scales,
            "quats": self.quats,
            "opacities": self.opacities,
            "features": self.features,
        }

        if parameters is not None:
            lr_dict = [(name, lr) for name, lr in lr_dict if name in parameters]

        self.optimizers = [
            torch.optim.Adam(
                [
                    {
                        "params": splat_dict[name],
                        "lr": lr * math.sqrt(batch_size),
                        "name": name,
                    }
                ],
                eps=1e-15 / math.sqrt(batch_size),
                betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
            )
            for name, lr in lr_dict
        ]

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()