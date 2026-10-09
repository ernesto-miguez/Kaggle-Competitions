"""
Carga y preparación de datos .zarr/.geff para training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import zarr
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset


def load_geff_nodes(geff_path):
    """
    Carga los nodos de un archivo .geff y los devuelve como DataFrame con columnas:
    node_id, t, z, y, x
    """
    g = zarr.open(geff_path, mode="r")
    return pd.DataFrame(
        {
            "node_id": g["nodes/ids"][:],
            "t": g["nodes/props/t/values"][:],
            "z": g["nodes/props/z/values"][:],
            "y": g["nodes/props/y/values"][:],
            "x": g["nodes/props/x/values"][:],
        }
    )


class TrainCellDatasetGEFF(Dataset):
    """
    Dataset para entrenamiento del 3D U-Net sobre volúmenes .zarr con GT sparse.
    
    Características:
    - Carga volúmenes 3D+tiempo en formato .zarr
    - Anotaciones sparsas (centroides) en formato .geff
    - Genera parches 3D aleatorios o centrados en células
    - Normalización robusta por percentiles
    - Mapas de calor gaussianos como targets
    """
    
    def __init__(
        self,
        zarr_path,
        geff_path,
        patch_size=(32, 64, 64),
        samples_per_epoch=35,
    ):
        """
        Args:
            zarr_path: Path al archivo .zarr con volúmenes 3D+tiempo
            geff_path: Path al archivo .geff con anotaciones de nodos
            patch_size: Tamaño del parche 3D (z, y, x)
            samples_per_epoch: Número de muestras por época
        """
        self.img_group = zarr.open(zarr_path, mode="r")["0"]
        self.nodes_df = load_geff_nodes(geff_path)
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.num_timesteps = self.img_group.shape[0]

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        pz, py, px = self.patch_size

        # 85% de los parches centrados en una célula, 15% aleatorios
        if np.random.rand() > 0.15 and len(self.nodes_df) > 0:
            sample_node = self.nodes_df.sample(1).iloc[0]
            t = int(sample_node["t"])
            nz, ny, nx = (
                int(sample_node["z"]),
                int(sample_node["y"]),
                int(sample_node["x"]),
            )

            z_max, y_max, x_max = self.img_group[t].shape
            z_s = max(0, min(nz - pz // 2, z_max - pz))
            y_s = max(0, min(ny - py // 2, y_max - py))
            x_s = max(0, min(nx - px // 2, x_max - px))
        else:
            t = np.random.randint(0, self.num_timesteps)
            z_max, y_max, x_max = self.img_group[t].shape
            z_s = np.random.randint(0, max(1, z_max - pz))
            y_s = np.random.randint(0, max(1, y_max - py))
            x_s = np.random.randint(0, max(1, x_max - px))

        vol = self.img_group[t][:]
        crop_vol = vol[z_s : z_s + pz, y_s : y_s + py, x_s : x_s + px].astype(
            np.float32
        )

        # Normalización robusta por percentiles (2% - 98%)
        p2, p98 = np.percentile(crop_vol, (2, 98))
        if p98 > p2:
            crop_vol = np.clip((crop_vol - p2) / (p98 - p2), 0.0, 1.0)
        else:
            crop_vol = np.zeros_like(crop_vol, dtype=np.float32)

        # Target en heatmap
        target = np.zeros(self.patch_size, dtype=np.float32)
        nodes_t = self.nodes_df[self.nodes_df["t"] == t]

        has_nodes = False
        for _, node in nodes_t.iterrows():
            cz, cy, cx = int(node["z"]), int(node["y"]), int(node["x"])
            if (
                z_s <= cz < z_s + pz
                and y_s <= cy < y_s + py
                and x_s <= cx < x_s + px
            ):
                target[cz - z_s, cy - y_s, cx - x_s] = 1.0
                has_nodes = True

        if has_nodes:
            target = gaussian_filter(target, sigma=1.8, mode="constant")
            if target.max() > 0:
                target /= target.max()

        crop_vol_tensor = torch.from_numpy(crop_vol).unsqueeze(0).float()
        target_tensor = torch.from_numpy(target).unsqueeze(0).float()

        return crop_vol_tensor, target_tensor
