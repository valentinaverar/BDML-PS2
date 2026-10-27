##########################################################
# 11_model_lightgbm.R
#
# Objetivo: Validar con validación cruzada variantes del modelo de boosting
# LightGBM y generar sus predicciones para Kaggle.
#
# Etapa del Problem Set: Etapa 2: Training and Comparing Algorithms.
# Insumos:  03_temp/ (train_clean.rds, test_clean.rds, train_X7.rds,
#           test_X7.rds, predictores.rds y cv_folds.rds).
# Salidas:  02_outputs/predictions/ (un .csv por variante), una fila por
#           variante en 02_outputs/tables/registro_modelos.csv y los
#           resultados de cada variante en 03_temp/.
##########################################################
#
# Este script se puede correr solo, siempre que antes se hayan corrido los
# scripts 01 a 05. La variante con X_7 necesita unos 7 GB de memoria RAM y
# tarda entre 15 y 20 minutos.
#
# Estructura de cada variante (la misma de los demás modelos):
#   1. Validación cruzada con las particiones de 03_temp/cv_folds.rds.
#   2. Elegir el umbral que maximiza el F1 con las predicciones fuera de
#      muestra.
#   3. Reentrenar con todo train, predecir el test y guardar el envío.
##########################################################

# Preparación -------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(data.table, tidyverse, caret, pROC, lightgbm)

source("01_code/functions/fun_metricas.R")
source("01_code/functions/fun_save_submission.R")
source("01_code/functions/fun_variables_lgbm.R")

predictores <- readRDS("03_temp/predictores.rds")
cv_folds    <- readRDS("03_temp/cv_folds.rds")

# Resultado, de train_clean. Las particiones guardan posiciones de fila: las
# matrices de cada variante deben tener los hogares en este mismo orden.
obs <- readRDS("03_temp/train_clean.rds")$Pobre
y <- as.integer(obs == "Yes")
stopifnot(max(unlist(cv_folds)) == length(y),
          length(unique(unlist(cv_folds))) == length(y))

# Categóricas de ubicación de X_7: vienen como códigos enteros (02) y
# LightGBM las trata como categorías, no como números ordenados.
categorias_x7 <- c("hh_Dominio", "hh_Depto", "hh_Clase")

# Hiperparámetros --------------------------------------------------------------
# Árboles de hasta 31 hojas, tasa de aprendizaje de 0,04 y 748 rondas;
# penalidad L2 (lambda_l2 = 10) y L1 (lambda_l1 = 0,1) sobre los valores de
# las hojas; al menos 100 hogares por hoja; y en cada ronda, 85% de las
# variables y 85% de los hogares, para reducir la varianza. La semilla y
# deterministic = TRUE hacen que el resultado se repita en cada corrida. Se
# usa la misma configuración en todas las variantes, para que la diferencia
# entre ellas se deba solo a las variables.
semilla <- 20261022L
parametros <- list(objective = "binary", metric = "binary_logloss",
                   boosting = "gbdt", learning_rate = 0.04, num_leaves = 31L,
                   max_depth = -1L, min_data_in_leaf = 100L,
                   min_sum_hessian_in_leaf = 0.001, feature_fraction = 0.85,
                   bagging_fraction = 0.85, bagging_freq = 1L,
                   lambda_l1 = 0.1, lambda_l2 = 10, min_gain_to_split = 0,
                   max_bin = 127L, bin_construct_sample_cnt = 200000L,
                   seed = semilla, deterministic = TRUE,
                   force_col_wise = TRUE,
                   num_threads = max(1L, min(6L, parallel::detectCores())),
                   verbosity = -1L)
rondas <- 748L

# Matrices ---------------------------------------------------------------------
fun_matriz_lgbm <- function(.nombre_conjunto, .base) {
  #' Armar la matriz de un conjunto de predictores para LightGBM
  #'
  #' @description \code{fun_matriz_lgbm()} arma la matriz numérica de train
  #' o test. X_7 se lee de sus propias bases (03_temp/train_X7.rds y
  #' test_X7.rds); los demás conjuntos, de train_clean/test_clean, con sus
  #' factores pasados a variables dicotómicas con model.matrix(), como en
  #' XGBoost.
  #'
  #' @param .nombre_conjunto \code{character}. Nombre de un conjunto de
  #' \code{03_temp/predictores.rds}.
  #' @param .base \code{character}. "train" o "test".
  #'
  #' @returns \code{list} con la matriz \code{x} y, en test, los \code{id}.
  vars <- predictores[[.nombre_conjunto]]
  limpia <- readRDS(paste0("03_temp/", .base, "_clean.rds"))
  id <- limpia$id
  
  if (.nombre_conjunto == "X_7_ampliado") {
    rm(limpia)
    base_x7 <- readRDS(paste0("03_temp/", .base, "_X7.rds"))
    # Mismo orden de hogares: en test se comparan los id; en train,
    # train_clean no guarda el id y el orden se verificó al construir X_7
    # en 02.
    if (.base == "test") stopifnot(identical(base_x7$id, id))
    x <- fun_matriz_x7(base_x7, vars)
    rm(base_x7)
  } else {
    stopifnot(all(vars %in% names(limpia)))
    x <- model.matrix(~ ., data = limpia[, vars, drop = FALSE])[, -1, drop = FALSE]
    rm(limpia)
  }
  gc()
  list(x = x, id = id)
}

# Función de una variante -------------------------------------------------------
fun_lgbm_variante <- function(.nombre_conjunto) {
  #' Validar una variante de LightGBM y generar su envío a Kaggle
  #'
  #' @description \code{fun_lgbm_variante()} valida el modelo con las
  #' particiones comunes, elige el umbral que maximiza el F1 fuera de
  #' muestra, reentrena con todo train, guarda el envío y registra la
  #' variante. Usa los objetos predictores, cv_folds, obs, y, parametros,
  #' rondas y categorias_x7 creados en la preparación de este script.
  #'
  #' @param .nombre_conjunto \code{character}. Nombre de un conjunto de
  #' \code{03_temp/predictores.rds}, por ejemplo "X_7_ampliado".
  #'
  #' @returns \code{list} (invisible) con el umbral, la comparación de
  #' umbrales y el nombre del archivo.
  
  x <- fun_matriz_lgbm(.nombre_conjunto, "train")$x
  stopifnot(nrow(x) == length(y))
  categorias <- intersect(categorias_x7, colnames(x))
  message(.nombre_conjunto, ": ", ncol(x), " columnas.")
  
  # 1. Validación cruzada. En cada fold se entrena con las filas de
  # cv_folds[[k]] y se predicen las demás.
  prob_oof <- rep(NA_real_, nrow(x))
  for (k in seq_along(cv_folds)) {
    inicio <- Sys.time()
    idx_train <- cv_folds[[k]]
    idx_valid <- setdiff(seq_len(nrow(x)), idx_train)
    datos_fold <- lgb.Dataset(x[idx_train, , drop = FALSE], label = y[idx_train],
                              categorical_feature = categorias,
                              params = parametros, free_raw_data = TRUE)
    modelo_fold <- lgb.train(params = parametros, data = datos_fold,
                             nrounds = rondas, verbose = -1L)
    prob_oof[idx_valid] <- as.numeric(predict(modelo_fold,
                                              x[idx_valid, , drop = FALSE]))
    message("Fold ", k, " listo en ",
            round(as.numeric(difftime(Sys.time(), inicio, units = "mins")), 1),
            " minutos.")
    rm(datos_fold, modelo_fold); gc(verbose = FALSE)
  }
  stopifnot(!anyNA(prob_oof))
  
  # 2. Umbral que maximiza el F1 con las predicciones fuera de muestra.
  # Evidencia: umbral de 0,5 frente al umbral elegido, con el número de
  # falsos positivos (leakage) y falsos negativos (undercoverage).
  res_umbral <- fun_umbral_f1(obs, prob_oof)
  umbral <- res_umbral$umbral
  comparacion <- bind_rows(
    fun_metricas_umbral(obs, prob_oof, 0.5) |>
      mutate(estrategia = "Umbral 0,5"),
    fun_metricas_umbral(obs, prob_oof, umbral) |>
      mutate(estrategia = "Umbral que maximiza F1")
  ) |>
    relocate(estrategia)
  print(comparacion)
  
  # 3. Modelo final con todo train, con la misma configuración.
  datos_final <- lgb.Dataset(x, label = y, categorical_feature = categorias,
                             params = parametros, free_raw_data = TRUE)
  rm(x); gc()
  modelo <- lgb.train(params = parametros, data = datos_final,
                      nrounds = rondas, verbose = -1L)
  rm(datos_final); gc()
  
  test <- fun_matriz_lgbm(.nombre_conjunto, "test")
  prob_test <- as.numeric(predict(modelo, test$x))
  id_test <- test$id
  rm(test); gc()
  
  # Envío. El nombre indica el algoritmo, el conjunto, los hiperparámetros
  # principales y el umbral.
  archivo <- paste0("LGBM_", .nombre_conjunto, "_hojas_31_iter_", rondas,
                    "_umbral_", fun_hiperparam_str(umbral, .digits = 2))
  fun_save_submission(data.frame(id = id_test), prob_test, umbral, archivo)
  
  # Registro. El puntaje público de Kaggle se anota a mano en
  # 00_data/kaggle_scores.csv, con el mismo nombre de archivo.
  fun_registrar_oof(.archivo = archivo, .algoritmo = "Boosting (LightGBM)",
                    .conjunto = .nombre_conjunto,
                    .n_predictores = length(predictores[[.nombre_conjunto]]),
                    .hiperparametros = paste0("hojas = 31; learning_rate = 0.04; ",
                                              "lambda_l2 = 10; lambda_l1 = 0.1; ",
                                              "rondas = ", rondas),
                    .obs = obs, .prob = prob_oof, .umbral = umbral)
  
  # Se guardan las predicciones fuera de muestra, las del test y la
  # importancia de las variables (ganancia), para la comparación de
  # algoritmos y el análisis del mejor modelo.
  saveRDS(list(oof = data.frame(rowIndex = seq_along(prob_oof), obs = obs,
                                Yes = prob_oof),
               prob_test = prob_test, umbral = umbral,
               comparacion = comparacion,
               importancia = lgb.importance(modelo)),
          file.path("03_temp", paste0(archivo, ".rds")))
  
  cat("\n", .nombre_conjunto, "- umbral que maximiza F1 en CV:", round(umbral, 3),
      "- F1:", round(comparacion$F1[2], 4), "\n")
  
  invisible(list(umbral = umbral, comparacion = comparacion, archivo = archivo))
}

# Variantes ---------------------------------------------------------------------
# Mejor modelo: conjunto ampliado X_7.
lgbm_X7 <- fun_lgbm_variante("X_7_ampliado")
# Conjunto común X_5_extendido, con el que se comparan todos los algoritmos.
lgbm_X5 <- fun_lgbm_variante("X_5_extendido")
# Modelo con X_6 para comparación 
lgbm_X6 <- fun_lgbm_variante("X_6_detallado")
