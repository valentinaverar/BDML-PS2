##########################################################
# Script maestro - Problem Set 2: Predicting Poverty
#
# Correr este archivo reproduce todos los resultados del repositorio.
#
# Para reproducir todo, desde la carpeta raíz del repositorio:
# - En una sesión interactiva de R: source("01_code/00_rundirectory.R")
# - Desde la terminal:              R CMD BATCH 01_code/00_rundirectory.R
#
# Autores (Equipo 03):
# - Julián Herrera
# - Andrés Silva
# - Valentina Vera
#
# Big Data y Machine Learning para Economía Aplicada (MECA 4107)
# Universidad de los Andes - 2026-20
##########################################################
# NOTA: este script solo llama a los demás scripts, en orden.
##########################################################

# Verificación del directorio de trabajo ----------------------------------
# Todas las rutas del repositorio son relativas a la carpeta raíz. Si R no
# está parado ahí, ningún script encontrará sus archivos. Detenemos la
# ejecución con un mensaje claro en lugar de un error confuso más adelante.
if (!file.exists("01_code/00_rundirectory.R")) {
  stop(
    "\n\nEl directorio de trabajo no es la carpeta raíz del repositorio.\n",
    "Directorio actual: ", getwd(), "\n",
    "Solución: use setwd(\"ruta/a/BDML-PS2\") o abra el proyecto desde esa ",
    "carpeta y vuelva a correr source(\"01_code/00_rundirectory.R\").\n",
    call. = FALSE
  )
}

# Bloque A: preparación de los datos (se corre una vez) -------------------

# Obtiene los datos crudos de Kaggle (ver README, sección Datos).
source("01_code/01_get_data.R")

# Limpia personas y hogares, construye variables a nivel hogar y las une.
source("01_code/02_clean_merge_data.R")

# Define los conjuntos de predictores, anidados por bloques económicos.
source("01_code/03_build_features.R")

# Etapa 1: desbalance de clases y relación con los predictores.
source("01_code/04_descriptive_analysis.R")

# Informe exploratorio de la base (02_outputs/figures/analisis_exploratorio.pdf).
source("01_code/17_eda_reporte.R")

# Particiones de validación cruzada compartidas por todos los modelos.
source("01_code/05_create_cv_folds.R")

# Bloque B: modelos (cada script genera sus envíos a Kaggle) --------------
# Cada script puede correrse solo, una vez corridos los scripts 01 a 05.

source("01_code/06_model_lpm.R")
source("01_code/07_model_logit.R")
source("01_code/08_model_elastic_net.R")
source("01_code/09_model_cart.R")
source("01_code/10_model_random_forest.R")
source("01_code/10_model_random_forest_caret.R")  
source("01_code/11_model_boosting.R")
source("01_code/11_model_adaboost.R")            
source("01_code/11_model_lightgbm.R")
source("01_code/12_model_naive_bayes.R")

# Rebalanceo de clases: Random Forest, AdaBoost y Naive Bayes con cada hogar
# pobre repetido 4 veces en el entrenamiento (mismos hiperparámetros).
source("01_code/16_rebalance_x4.R")

# Bloque C: análisis para las presentaciones ------------------------------

# Etapa 2: comparación de algoritmos (deck compare_equipo_03).
source("01_code/13_compare_algorithms.R")

# Etapa 3: análisis del mejor modelo (deck best_equipo_03).
source("01_code/14_best_model_analysis.R")

# Etapa 4: implicaciones de política.
source("01_code/15_policy_analysis.R")

# Compilación de las slides.
source("04_slides/compile_slides.R")
