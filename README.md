# Predicting Poverty in Colombia

**Problem Set 2: predicción de pobreza a nivel de hogar con modelos de clasificación**

**Equipo 03:** Julián Herrera, Andrés Silva, Valentina Vera
Big Data y Machine Learning para Economía Aplicada (MECA 4107), Universidad de los Andes, 2026-20

---

Este repositorio contiene el código para predecir si un hogar colombiano es pobre con datos de la encuesta de hogares del DANE (Misión de Empalme, MESE). Las predicciones se evalúan con el F1 en la competencia de Kaggle [uniandes-bdml-2026-20-ps-2](https://www.kaggle.com/competitions/uniandes-bdml-2026-20-ps-2).

La pregunta que guía el trabajo es: **¿cuál es el mejor modelo para predecir la pobreza y qué implica para la política pública?**

## Resultados principales

- **Mejor modelo:** LightGBM con 1.296 variables del hogar y de sus miembros (conjunto X_7). F1 de 0,77 en Kaggle y de 0,768 en validación cruzada.
- **Con las mismas 43 variables**, el boosting supera a los demás algoritmos (0,731 frente a 0,715 de Random Forest y 0,711 de Elastic Net).
- **Las variables pesan más que el ajuste:** ampliar la información sube el F1 3,7 puntos; los hiperparámetros y el rebalanceo de clases lo mueven menos de 1.
- Las presentaciones están en `04_slides/`: `compare_equipo_03.pdf` (comparación de algoritmos) y `best_equipo_03.pdf` (mejor modelo y política).

## Instrucciones de replicación

Desde la carpeta raíz del repositorio, en R:

```r
source("01_code/00_rundirectory.R")
```

El script maestro corre todo en orden: preparación de los datos (scripts 01 a 05), modelos (06 a 12 y 16), análisis (13 a 15) y compilación de las presentaciones.

**Datos.** No están en el repositorio. `01_get_data.R` los busca en `00_data/raw/` y, si no están, los descarga con la [CLI de Kaggle](https://github.com/Kaggle/kaggle-api) (requiere `~/.kaggle/kaggle.json` y aceptar las reglas de la competencia). También se puede descargar el zip desde la [página de datos](https://www.kaggle.com/competitions/uniandes-bdml-2026-20-ps-2/data) y guardarlo, sin descomprimir, en `00_data/raw/`.

**Presentaciones.** `04_slides/compile_slides.R` compila los dos decks con XeLaTeX (por ejemplo, con `tinytex::install_tinytex()`). Si no hay LaTeX, avisa y el resto del análisis corre igual.

## Estructura del repositorio

```
BDML-PS2/
├── 00_data/
│   ├── raw/                  # datos de Kaggle (no versionados)
│   └── kaggle_scores.csv     # puntaje de cada envío a Kaggle
├── 01_code/
│   ├── 00_rundirectory.R     # script maestro
│   ├── 01_ a 05_*.R          # datos: obtención, limpieza, predictores, descriptivas y folds
│   ├── 06_ a 12_model_*.R    # un script por algoritmo
│   ├── 13_ a 15_*.R          # comparación, mejor modelo y política
│   ├── 16_rebalance_*.R      # rebalanceo de clases
│   ├── 17_eda_reporte.R      # informe exploratorio
│   └── functions/            # funciones reutilizables
├── 02_outputs/
│   ├── figures/              # figuras de las presentaciones
│   ├── tables/               # tablas .tex y registro de modelos
│   ├── artifacts/            # importancia por permutación del mejor modelo
│   └── predictions/          # archivos enviados a Kaggle
├── 03_temp/                  # bases limpias y resultados intermedios (no versionados)
└── 04_slides/                # presentaciones (.tex y .pdf)
```

## Salidas

Todas se generan con el código, en `02_outputs/`:

- **Figuras** (`figures/`) y **tablas** (`tables/`, en `.tex`) que usan las presentaciones.
- **Predicciones** (`predictions/`): los archivos enviados a Kaggle, con columnas `id` y `pobre`. El nombre indica el algoritmo y sus hiperparámetros.
- **Registro de modelos** (`tables/registro_modelos.csv`): una fila por variante, con su configuración y su F1 en validación cruzada.

## Decisiones de diseño

- **Validación:** todos los modelos usan las mismas 5 particiones (`05_create_cv_folds.R`) y el umbral que maximiza el F1 fuera de muestra.
- **Igualdad de condiciones:** `03_build_features.R` define conjuntos anidados de predictores, de X_1 (5 variables de vivienda) a X_7 (1.296). Los algoritmos se comparan con el mismo conjunto (X_5, 43 variables).
- **Registro:** cada variante queda en `02_outputs/tables/registro_modelos.csv` con su F1 de validación cruzada, y su puntaje de Kaggle en `00_data/kaggle_scores.csv`.

## Software

- R 4.6.1. Los paquetes se instalan y cargan con `pacman::p_load()`.
- Opcional: la CLI de Kaggle, para descargar los datos, y LaTeX, para compilar las presentaciones.