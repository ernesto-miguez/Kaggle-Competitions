"""
Funciones de inferencia, extracción de centroides y evaluación local.
"""

from __future__ import annotations

import numpy as np
import torch
import zarr
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max

from .data_loader import load_geff_nodes
from .utils import normalize_volume_robust, suppress_duplicate_centroids


def extract_centroids_physical(
    model,
    volume_3d,
    scale=(1.625, 0.40625, 0.40625),
    patch_size=(32, 128, 128),
    device="cuda",
):
    """
    Extrae centroides celulares a partir de un volumen 3D con inferencia por patches.
    
    Pipeline:
    1. Normalización robusta del volumen
    2. División en patches overlapping
    3. Inferencia mediante el modelo (con autocast)
    4. Concatenación de predicciones
    5. Smoothing Gaussiano
    6. Detección de picos locales
    7. Supresión de duplicados (NMS físico)
    
    Args:
        model: Modelo entrenado (UNet3D)
        volume_3d: Volumen 3D (Z, Y, X)
        scale: Escala física (z, y, x) en micras/voxel
        patch_size: Tamaño de patch para inferencia (pz, py, px)
        device: Dispositivo PyTorch ("cuda" o "cpu")
    
    Returns:
        Array de centroides (N, 3) en vóxeles
    """
    model.eval()
    z_max, y_max, x_max = volume_3d.shape

    norm_vol = normalize_volume_robust(volume_3d)
    pz, py, px = patch_size

    crops_list = []
    coords_list = []

    # Extrae parches del volumen normalizado
    for z in range(0, z_max, pz):
        for y in range(0, y_max, py):
            for x in range(0, x_max, px):
                z_e, y_e, x_e = (
                    min(z + pz, z_max),
                    min(y + py, y_max),
                    min(x + px, x_max),
                )
                crop = norm_vol[z:z_e, y:y_e, x:x_e]

                # Padding si el parche está en el borde
                pad_z, pad_y, pad_x = (
                    pz - crop.shape[0],
                    py - crop.shape[1],
                    px - crop.shape[2],
                )
                if pad_z > 0 or pad_y > 0 or pad_x > 0:
                    crop = np.pad(
                        crop,
                        ((0, pad_z), (0, pad_y), (0, pad_x)),
                    )

                crops_list.append(crop)
                coords_list.append((z, z_e, y, y_e, x, x_e))

    if len(crops_list) == 0:
        return np.empty((0, 3), dtype=np.int64)

    crops_tensor = torch.from_numpy(
        np.array(crops_list, dtype=np.float32)
    ).unsqueeze(1)
    preds = []

    # Inferencia por batch con autocast
    inference_batch = 4
    with torch.inference_mode():
        for i in range(0, len(crops_tensor), inference_batch):
            batch = crops_tensor[i : i + inference_batch].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                out = torch.sigmoid(model(batch))
            preds.append(out.squeeze(1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    heatmap = np.zeros_like(norm_vol, dtype=np.float32)

    # Reconstruye el heatmap desde los parches
    for idx, (z, z_e, y, y_e, x, x_e) in enumerate(coords_list):
        heatmap[z:z_e, y:y_e, x:x_e] = preds[idx][
            : z_e - z, : y_e - y, : x_e - x
        ]

    # Smoothing gaussiano para eliminar ruido del fondo
    heatmap_smooth = gaussian_filter(heatmap, sigma=0.8)

    max_p = heatmap_smooth.max()
    if max_p < 0.05:
        return np.empty((0, 3), dtype=np.int64)

    # Umbral relativo adaptativo (45% del pico máximo)
    rel_thresh = max(0.15, max_p * 0.45)

    # Detección de picos locales
    coords = peak_local_max(
        heatmap_smooth,
        min_distance=4,
        threshold_abs=rel_thresh,
        exclude_border=False,
        num_peaks=60,  # Límite para evitar explosión de falsos positivos
    )

    # NMS por distancia física en micras
    coords = suppress_duplicate_centroids(coords, min_dist_um=3.0, scale=scale)
    return coords.astype(np.int64)


def link_nodes_between_frames(
    nodes_t0, nodes_t1, max_dist_um=7.0, scale=(1.625, 0.40625, 0.40625)
):
    """
    Crea enlaces de tracking entre centroides de frames consecutivos.
    
    Usa asignación lineal (Algoritmo Húngaro) con distancia física real en micras.
    
    Args:
        nodes_t0: Centroides en frame t (N, 3) en vóxeles
        nodes_t1: Centroides en frame t+1 (M, 3) en vóxeles
        max_dist_um: Distancia máxima en micras para considerar link
        scale: Escala física (z, y, x)
    
    Returns:
        List de tuplas (idx_t0, idx_t1) representando aristas
    """
    if len(nodes_t0) == 0 or len(nodes_t1) == 0:
        return []

    # Convertir vóxeles a micras
    t0_um = nodes_t0 * np.array(scale)
    t1_um = nodes_t1 * np.array(scale)

    # Matriz de distancias físicas
    diff = t0_um[:, np.newaxis, :] - t1_um[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    # Asignación lineal
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    edges = []
    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] <= max_dist_um:
            edges.append((r, c))
    return edges


def evaluate_validation_f1(
    model,
    val_pairs,
    match_radius_um=7.0,
    scale=(1.625, 0.40625, 0.40625),
    device="cuda",
):
    """
    Calcula F1-Score local en validación iterando sobre pares (zarr, geff).
    
    Itera sobre todas las secuencias y timesteps de validación, extrae centroides
    predichos, y los compara con GT usando distancia física en micras.
    
    Args:
        model: Modelo entrenado
        val_pairs: Lista de tuplas (zarr_path, geff_path)
        match_radius_um: Radio de matching en micras
        scale: Escala física
        device: Dispositivo PyTorch
    
    Returns:
        (f1, precision, recall): Métricas globales de validación
    """
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0

    for z_path, g_path in val_pairs:
        img_group = zarr.open(z_path, mode="r")["0"]
        gt_nodes_df = load_geff_nodes(g_path)

        # Itera sobre todos los timesteps
        for t in range(img_group.shape[0]):
            vol_3d = img_group[t][:]
            pred_coords = extract_centroids_physical(
                model,
                vol_3d,
                scale=scale,
                patch_size=(32, 128, 128),
                device=device,
            )

            # Ground truth para este timestep
            gt_coords = (
                gt_nodes_df[gt_nodes_df["t"] == t][["z", "y", "x"]]
                .values.astype(np.int64)
            )

            if len(pred_coords) == 0:
                total_fn += len(gt_coords)
                continue
            if len(gt_coords) == 0:
                total_fp += len(pred_coords)
                continue

            # Matriz de distancias físicas en micras
            pred_um = pred_coords * np.array(scale)
            gt_um = gt_coords * np.array(scale)

            dist_matrix = np.linalg.norm(
                pred_um[:, None, :] - gt_um[None, :, :], axis=-1
            )

            # Matching voraz por umbral físico
            matched_gt = set()
            tp = 0

            for pred_idx in range(len(pred_coords)):
                min_dist_idx = np.argmin(dist_matrix[pred_idx])
                if dist_matrix[pred_idx, min_dist_idx] <= match_radius_um:
                    if min_dist_idx not in matched_gt:
                        tp += 1
                        matched_gt.add(min_dist_idx)

            fp = len(pred_coords) - tp
            fn = len(gt_coords) - len(matched_gt)

            total_tp += tp
            total_fp += fp
            total_fn += fn

    # Cálculo de métricas globales
    prec = total_tp / (total_tp + total_fp + 1e-8)
    rec = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)

    return f1, prec, rec
