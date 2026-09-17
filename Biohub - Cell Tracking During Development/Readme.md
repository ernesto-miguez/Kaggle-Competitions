# 🧫 Biohub - Cell Tracking During Development

Este directorio contiene la solución y los experimentos desarrollados para la competición de Kaggle **Biohub - Cell Tracking During Development**. El objetivo es automatizar la detección, el seguimiento temporal y la reconstrucción de linajes celulares a partir de volúmenes de microscopía 3D en embriones de pez cebra.

---

## 📌 Resumen de la Competición

* **Dominio:** Visión por Computador / Bioinformática / Datos 3D+Tiempo.
* **Problema:** El análisis manual de imágenes de microscopía en 3D es un cuello de botella en biología del desarrollo. Las células se mueven, cambian de forma y se dividen (mitosis) en entornos de alta densidad celular con ruido de imagen.
* **Métrica Oficial:** Métrica combinada basada en grafos:
  1. **Edge Jaccard:** Precisión en la conexión de centroides entre el tiempo $t$ y $t+1$ (evaluado según la distancia física en $\mu m$).
  2. **Division Jaccard:** Precisión al identificar los eventos de división celular (mitosis) y sus células hijas.

---

## 📊 Estructura y Formato de Datos

* **Imágenes 3D+Tiempo (`.zarr`):** Volúmenes $T \times Z \times Y \times X$ (típicamente $100 \times 64 \times 256 \times 256$) comprimidos mediante Blosc/Zstd.
* **Resolución Física:** $z = 1.625 \, \mu m$, $y = x = 0.40625 \, \mu m$ por voxel.
* **Ground Truth (`.geff`):** Anotaciones esparsas de grafos que contienen las coordenadas de los nodos (centroides) y las aristas (enlaces temporales).

---

## 🛠️ Roadmap de Desarrollo

- **Fase 1: Baseline Heurístico**
  - Detección de centroides mediante filtrado clásico de Laplaciano de Gaussiana (`scikit-image`).
  - Asignación de aristas temporales por proximidad euclídea (Algoritmo Húngaro).
  - Pipeline de inferencia offline funcional para generar un `submission.csv` válido.
- **Fase 2: Módulo de Detección 3D (Deep Learning)**
  - Entrenamiento de una red de segmentación/mapas de calor 3D (3D U-Net / MONAI) para la localización precisa de centroides.
- **Fase 3: Módulo de Seguimiento y Afinidad**
  - Modelos de coincidencia basados en *features* de apariencia, velocidad y dirección celular.
- **Fase 4: Optimización Global del Grafo**
  - Ajuste de linajes y detección de división celular mediante optimización basada en flujos o ILP.

---

## 📂 Estructura del Proyecto

```text
Biohub-Cell-Tracking/
│
├── 📂 notebooks/             # Exploración de datos (EDA), visualización MIP y animaciones GIF
├── 📂 src/                   # Módulos de procesamiento .zarr/.geff y algoritmos de tracking
├── 📂 submissions/           # Historial de archivos de entrega generados
└── 📄 README.md              # Documentación del proyecto
