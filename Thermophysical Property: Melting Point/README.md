# 🧪 Thermophysical Property: Melting Point

## 🎯 Objetivo del Proyecto

Desarrollar modelos de Machine Learning capaces de predecir el punto de fusión de compuestos orgánicos a partir de su información molecular.

El punto de fusión es una propiedad fisicoquímica relevante en ámbitos como el diseño de fármacos, la selección de materiales y la seguridad de los procesos químicos. Su predicción mediante modelos computacionales puede ayudar a estimar propiedades de sustancias cuando las mediciones experimentales son costosas, requieren tiempo o no están disponibles.

El reto consiste en identificar patrones entre las características moleculares de los compuestos y sus puntos de fusión, construyendo modelos que generalicen correctamente a moléculas no vistas durante el entrenamiento.


## 📌 Resumen de la Competición

- **Dominio:** Química física / Machine Learning / Quimioinformática.
- **Competición:** Thermophysical Property: Melting Point Prediction Challenge (Kaggle).
- **Problema:** Regresión para predecir el punto de fusión de compuestos orgánicos a partir de sus características moleculares.
- **Datos:** 3.328 compuestos químicos, distribuidos en 2.662 muestras de entrenamiento y 666 de test.
- **Variables de entrada:** Identificador del compuesto (`id`), representación molecular SMILES (`SMILES`) y descriptores de grupos funcionales (`Group 1..N`).
- **Variable objetivo:** Punto de fusión en Kelvin (`Tm`).
- **Métrica de evaluación:** Mean Absolute Error (MAE). Mide el error absoluto medio entre las predicciones y los valores reales; cuanto menor sea, mejor será el modelo.


## 📊 Dataset

El conjunto de datos proporcionado por la competición contiene información sobre compuestos químicos y sus propiedades.

| Archivo | Descripción |
|---|---|
| `train.csv` | Datos de entrenamiento, con las variables predictoras y el punto de fusión (`Tm`). |
| `test.csv` | Datos de test, con las variables predictoras pero sin el valor objetivo. |
| `sample_submission.csv` | Plantilla para generar las predicciones en el formato requerido. |

### Variables principales

- **`id`:** Identificador único de cada compuesto.
- **`SMILES`:** Representación textual de la estructura molecular.
- **`Group 1..N`:** Descriptores numéricos que representan recuentos de grupos funcionales presentes en las moléculas.
- **`Tm`:** Punto de fusión expresado en Kelvin, disponible únicamente en los datos de entrenamiento.

El conjunto de datos procede de los materiales del curso *Machine Learning for Engineers*, en el apartado dedicado a propiedades termofísicas.

## 🧠 Enfoques y experimentos

Para abordar la competición, se han explorado diferentes estrategias de modelado y representación de las moléculas:

- **Modelos de referencia:** Regresión lineal y Random Forest para establecer una primera referencia de rendimiento.
- **Descriptores moleculares:** Uso de RDKit para obtener características químicas adicionales a partir de las estructuras moleculares.
- **Modelos de boosting:** Experimentación con LightGBM, CatBoost y XGBoost.
- **Ingeniería de características:** Selección de variables, interacciones, características polinómicas y técnicas de reducción de dimensionalidad.
- **Representaciones moleculares:** Exploración de fingerprints moleculares y embeddings químicos.
- **Redes neuronales:** Evaluación de redes neuronales convencionales y redes neuronales gráficas (GNN).
- **Ensembles y stacking:** Combinación de distintos modelos para aprovechar sus capacidades complementarias.
- **Optimización de hiperparámetros:** Uso de Optuna para buscar configuraciones de los modelos.

Estos experimentos permiten comparar distintas formas de representar la información química y estudiar su impacto en la predicción del punto de fusión.

## 📈 Evaluación

La competición utiliza el **Mean Absolute Error (MAE)** para evaluar las predicciones sobre un conjunto de test reservado.

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

Donde:

- $y_i$ representa el punto de fusión real.
- $\hat{y}_i$ representa el punto de fusión predicho.
- $n$ es el número de muestras evaluadas.

El objetivo es minimizar este error, obteniendo predicciones lo más próximas posible a los valores experimentales.

Durante el desarrollo, los modelos se comparan mediante validación local antes de generar el archivo final de predicciones.

## 📤 Formato de entrega

La competición requiere un archivo CSV con una predicción del punto de fusión para cada identificador del conjunto de test.

El archivo debe incluir las columnas `id` y `Tm`, siguiendo el formato de `sample_submission.csv`.

```csv
id,Tm
210,350.1
211,583.8
```

Los valores anteriores son únicamente ilustrativos.

## 🛠️ Tecnologías utilizadas

- **Python:** Lenguaje principal.
- **Pandas y NumPy:** Manipulación y procesamiento de datos.
- **Scikit-learn:** Modelado, validación y evaluación.
- **RDKit:** Procesamiento y cálculo de descriptores químicos.
- **LightGBM, CatBoost y XGBoost:** Modelos de boosting.
- **PyTorch y PyTorch Geometric:** Desarrollo de redes neuronales y modelos basados en grafos.
- **Optuna:** Optimización de hiperparámetros.
- **Matplotlib y Seaborn:** Visualización y análisis exploratorio.
- **UMAP:** Reducción de dimensionalidad y exploración de representaciones moleculares.


## 📁 Estructura del proyecto

```text
Thermophysical Property - Melting Point/
│
├── 📂 notebooks/             # EDA, preprocesamiento, modelos.
├── 📂 src/                   # Módulos reutilizables
├── 📂 submissions/           # Ficheros finales
└── 📄 README.md              # Documentación del proyecto

---

