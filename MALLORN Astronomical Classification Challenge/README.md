# 🔭 MALLORN - Astronomical Classification Challenge

Este directorio contiene la solución y los experimentos desarrollados para la competición de Kaggle **MALLORN Astronomical Classification Challenge**. El objetivo es detectar *Tidal Disruption Events* (TDEs) a partir de curvas de luz astronómicas multibanda, utilizando técnicas de Machine Learning y *feature engineering* basado en el comportamiento físico de los transitorios.

El principal avance del proyecto fue incorporar características derivadas de la **función de Bazin**, que permitieron mejorar significativamente la capacidad de discriminación del modelo entre TDEs y otros fenómenos astronómicos, especialmente los núcleos galácticos activos (AGN).

---

## 📌 Resumen de la Competición

* **Dominio:** Machine Learning / Astrofísica / Series temporales.
* **Competición:** MALLORN Astronomical Classification Challenge (Kaggle).
* **Problema:** Clasificación binaria de eventos astronómicos para identificar TDEs, fenómenos que ocurren cuando una estrella se aproxima demasiado a un agujero negro supermasivo y es desgarrada por sus fuerzas de marea.
* **Datos:** Curvas de luz multibanda simuladas a partir de observaciones astronómicas reales del *Zwicky Transient Facility* (ZTF).
* **Clase positiva:** TDE (`target = 1`).
* **Métrica de evaluación:** F1 Score, que combina precisión (*precision*) y exhaustividad (*recall*) y resulta útil para evaluar problemas de clasificación con clases desbalanceadas.

### Objetivo científico

Identificar patrones temporales y fotométricos que permitan distinguir los TDEs de otros transitorios, como los AGN, cuyas curvas de luz pueden presentar comportamientos similares.

---

## 📊 Estructura y Formato de Datos

Los datos contienen observaciones fotométricas de múltiples eventos astronómicos. Cada evento está formado por una secuencia de medidas de flujo tomadas en diferentes instantes y bandas fotométricas.

### Variables principales

* **`object_id`:** Identificador único del objeto astronómico.
* **`Time (MJD)`:** Fecha de observación en *Modified Julian Date*.
* **`Flux`:** Flujo observado, expresado en microjanskys (μJy).
* **`Flux_err`:** Incertidumbre asociada a la medida del flujo.
* **`Filter`:** Banda fotométrica utilizada.
* **`Redshift (z)`:** Desplazamiento al rojo del objeto.
* **Extinción por polvo:** Variable relacionada con la atenuación de la luz.
* **`SpecType`:** Tipo espectroscópico del objeto, utilizado para construir la etiqueta de entrenamiento.

### Bandas fotométricas

| Filtro | Rango aproximado |
| ------ | ---------------- |
| u      | 338–395 nm       |
| g      | 405–552 nm       |
| r      | 553–690 nm       |
| i      | 690–817 nm       |
| z      | 828–920 nm       |
| y      | 920–1010 nm      |

### Características de los datos

* Series temporales irregulares, con intervalos variables entre observaciones.
* Observaciones con diferentes niveles de incertidumbre.
* Múltiples bandas fotométricas por objeto.
* Desbalanceo entre TDEs y el resto de las clases.

---

## 🛠️ Roadmap de Desarrollo

### Fase 1: Exploratory Data Analysis (EDA)

* Análisis de la distribución de clases.
* Exploración de las curvas de luz de diferentes tipos de eventos.
* Visualización de observaciones por filtro.
* Análisis de las diferencias entre TDEs y AGN.
* Estudio de la distribución temporal, el flujo y las incertidumbres observacionales.

### Fase 2: Feature Engineering

Desarrollo de características agregadas para representar cada curva de luz mediante un vector tabular.

**Features estadísticas**

* Media, desviación estándar, mínimos y máximos.
* Amplitud, cuantiles, rango intercuartílico, *skewness* y curtosis.

**Features temporales**

* Duración total de la curva.
* Tiempo de subida y de decaimiento.
* Asimetría temporal.
* Pendientes de subida y decaimiento.

**Features fotométricas**

* Fluencia integrada.
* Relación señal-ruido (*Signal-to-Noise Ratio*, SNR).
* Número de picos.
* Estadísticas del flujo por filtro.
* Diferencias de flujo entre bandas, como g-r, r-i e i-z.

### Fase 3: Modelos de Machine Learning y ensembles

Se experimentó con diferentes algoritmos para mejorar la clasificación:

* **LightGBM:** Gradient Boosting basado en árboles de decisión.
* **CatBoost:** Gradient Boosting con una estrategia alternativa de entrenamiento.
* **Stacking:** combinación de predicciones de los modelos base mediante una regresión logística como *meta-model*.
* **LSTM:** modelos recurrentes para procesar secuencias temporales.
* **GRU + Dense:** arquitectura híbrida para combinar información secuencial y características tabulares.

Para tratar el desbalanceo de clases se probaron pesos de clase y técnicas de aumento de datos como SMOTE.

Los modelos más complejos y el aumento de datos sintéticos no proporcionaron mejoras consistentes en el rendimiento final.

### Fase 4: Feature Engineering basado en la función de Bazin

Esta fase supuso el principal avance del proyecto.

La función de Bazin es un modelo analítico utilizado para describir la evolución temporal del flujo de transitorios astronómicos. Permite representar patrones de subida y decaimiento mediante parámetros interpretables.

Se ajustó la función a las curvas de luz y se extrajeron características como:

* Parámetros del ajuste de Bazin.
* Escalas temporales de subida y decaimiento.
* Amplitud y nivel de fondo.
* Error del ajuste y chi-cuadrado reducido (χ²).

Estas características aportan información sobre la forma de la curva de luz y permiten evaluar hasta qué punto un evento sigue el patrón temporal representado por la función.

El chi-cuadrado reducido se utilizó como una medida adicional de la calidad del ajuste, contribuyendo a la discriminación entre TDEs y otros transitorios, especialmente AGN.

### Fase 5: Optimización y evaluación

* Validación mediante *Stratified K-Fold* y predicciones *Out-of-Fold* (OOF).
* Optimización de hiperparámetros con Optuna en las primeras versiones.
* Ajuste del umbral de clasificación para maximizar el F1 Score.
* Análisis de *feature importance* mediante LightGBM.
* Análisis de falsos positivos por tipo de evento.
* Evaluación final mediante el *Public Leaderboard* de Kaggle.

---

## 🏆 Resultados principales

| Enfoque                                                      | Resultado             |
| ------------------------------------------------------------ | --------------------- |
| Stacking inicial (LightGBM + CatBoost + Logistic Regression) | Public LB F1 ≈ 0.44   |
| LightGBM con features derivadas de Bazin                     | Public LB F1 = 0.6179 |

La incorporación de características derivadas de Bazin supuso una mejora relativa aproximada del **40 %** respecto al resultado inicial de F1 = 0.44.

El resultado puso de manifiesto que la calidad de la representación de los datos puede tener más impacto que incrementar la complejidad de los algoritmos.

*Nota: los resultados corresponden al Public Leaderboard de la competición y no deben interpretarse como una estimación de rendimiento obtenida exclusivamente mediante validación local.*


---

## 📂 Estructura del Proyecto

```text
MALLORN Astronomical Classification Challenge/
│
├── 📂 notebooks/             # Exploración de datos (EDA), visualización MIP y animaciones GIF
├── 📂 src/                   # Módulos de procesamiento, feature extraction...
├── 📂 submissions/           # Ficheros subidos a la competición
└── 📄 README.md              # Documentación del proyecto
```

---

## 📚 Referencias

* Bazin, G., Ruhlmann-Kleider, V., Palanque-Delabrouille, N., et al. (2011). *Photometric selection of type Ia supernovae in the Supernova Legacy Survey*. Astronomy & Astrophysics, 534, A43.
* Komossa, S. (2015). *Tidal disruption of stars by supermassive black holes: Status of observations*. Journal of High Energy Astrophysics, 7, 148–157.
* Gezari, S., et al. (2009). *Ultraviolet detection of the tidal disruption of a star by a supermassive black hole*. The Astrophysical Journal, 698, 1367–1379.

---
