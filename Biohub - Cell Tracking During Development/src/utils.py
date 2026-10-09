"""
Utilidades generales para detección de centroides, normalización y evaluación.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree


def normalize_volume_robust(volume_3d):
    """
    Normaliza un volumen 3D usando percentiles robustos (2% y 98%).
    
    Devuelve un volumen normalizado en rango [0, 1].
    
    Args:
        volume_3d: Array 3D (numpy)
    
    Returns:
        Volumen normalizado (float32)
    """
    vol_float = volume_3d.astype(np.float32)
    p2, p98 = np.percentile(vol_float, (2, 98))
    if p98 <= p2:
        return np.zeros_like(vol_float, dtype=np.float32)
    return np.clip((vol_float - p2) / (p98 - p2), 0.0, 1.0)


def suppress_duplicate_centroids(
    coords, min_dist_um=4.0, scale=(1.625, 0.40625, 0.40625)
):
    """
    Elimina centroides duplicados en función de la distancia física real (en micras).
    
    Usa KDTree para búsquedas rápidas de vecinos.
    
    Args:
        coords: Array de centroides en vóxeles (N, 3)
        min_dist_um: Distancia mínima en micras para considerar duplicado
        scale: Escala física (z, y, x) en micras/voxel
    
    Returns:
        Array de centroides filtrados
    """
    if len(coords) == 0:
        return coords

    coords_um = coords * np.array(scale)
    tree = KDTree(coords_um)

    filtered_coords = []
    visited = set()

    for idx, pt in enumerate(coords_um):
        if idx in visited:
            continue
        filtered_coords.append(coords[idx])
        neighbors = tree.query_ball_point(pt, r=min_dist_um)
        visited.update(neighbors)

    return np.array(filtered_coords, dtype=np.int64)


def compute_f1_frame(pred_coords, gt_coords, max_dist=4.0):
    """
    Calcula tp, fp, fn para una sola imagen/frame.
    
    Usa matching por distancia euclidiana y asignación lineal.
    
    Args:
        pred_coords: Centroides predichos (N, 3)
        gt_coords: Centroides ground truth (M, 3)
        max_dist: Distancia máxima para considerar match
    
    Returns:
        (tp, fp, fn): Verdaderos positivos, falsos positivos, falsos negativos
    """
    n_pred, n_gt = len(pred_coords), len(gt_coords)
    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0
    if n_pred == 0 or n_gt == 0:
        return 0, n_pred, n_gt

    dist_matrix = cdist(pred_coords, gt_coords)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    tp = sum(
        1 for r, c in zip(row_ind, col_ind) if dist_matrix[r, c] <= max_dist
    )
    fp = n_pred - tp
    fn = n_gt - tp
    return tp, fp, fn
