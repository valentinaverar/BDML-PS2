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
pacman::p_load(data.table, tidyverse, caret, pROC, lightgbm, knitr)

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

# Búsqueda de hiperparámetros --------------------------------------------------
# Grilla con validación cruzada, sobre el conjunto común X_5_extendido
# (43 variables), donde cada configuración tarda menos de un minuto. Se varían
# los tres hiperparámetros que más controlan la complejidad del modelo:
#   - num_leaves: hojas por árbol (15, 31, 63 y 127). Más hojas permiten
#     interacciones más complejas, con más riesgo de sobreajuste.
#   - learning_rate: cuánto aporta cada árbol (0,02, 0,04 y 0,1). Con una tasa
#     menor se usan más rondas, para que el aprendizaje total sea parecido
#     (0,02 x 1.496, 0,04 x 748 y 0,1 x 300 rondas).
#   - min_data_in_leaf: hogares mínimos por hoja (20, 100 y 300). Más
#     hogares por hoja hacen el modelo más simple y estable.
# Son 4 x 3 x 3 = 36 combinaciones, entre ellas la configuración elegida.
# Los demás hiperparámetros quedan como en la configuración elegida. Para cada
# combinación se usan las mismas particiones de los demás modelos, se elige el
# umbral que maximiza el F1 con las predicciones fuera de muestra y se guarda
# el F1. La búsqueda no genera envíos a Kaggle ni entra al registro de
# modelos: documenta si la configuración elegida está cerca de la mejor.
# Con FALSE se lee la tabla guardada en vez de volver a correr la búsqueda.
BUSCAR_HIPERPARAMETROS <- TRUE
ruta_busqueda <- "02_outputs/tables/busqueda_lgbm.csv"

if (BUSCAR_HIPERPARAMETROS || !file.exists(ruta_busqueda)) {
  x_busqueda <- fun_matriz_lgbm("X_5_extendido", "train")$x
  stopifnot(nrow(x_busqueda) == length(y))
  grilla <- expand.grid(hojas = c(15L, 31L, 63L, 127L),
                        tasa = c(0.02, 0.04, 0.1),
                        min_hoja = c(20L, 100L, 300L)) |>
    mutate(rondas = case_when(tasa == 0.02 ~ 1496L,
                              tasa == 0.04 ~ 748L,
                              TRUE ~ 300L))
  
  busqueda <- map_dfr(seq_len(nrow(grilla)), function(i) {
    inicio <- Sys.time()
    params_i <- modifyList(parametros,
                           list(num_leaves = grilla$hojas[i],
                                learning_rate = grilla$tasa[i],
                                min_data_in_leaf = grilla$min_hoja[i]))
    prob_i <- rep(NA_real_, nrow(x_busqueda))
    for (k in seq_along(cv_folds)) {
      idx_train <- cv_folds[[k]]
      idx_valid <- setdiff(seq_len(nrow(x_busqueda)), idx_train)
      datos_k <- lgb.Dataset(x_busqueda[idx_train, , drop = FALSE],
                             label = y[idx_train], params = params_i,
                             free_raw_data = TRUE)
      modelo_k <- lgb.train(params = params_i, data = datos_k,
                            nrounds = grilla$rondas[i], verbose = -1L)
      prob_i[idx_valid] <- as.numeric(predict(modelo_k,
                                              x_busqueda[idx_valid, , drop = FALSE]))
      rm(datos_k, modelo_k); gc(verbose = FALSE)
    }
    res_i <- fun_umbral_f1(obs, prob_i)
    met_i <- fun_metricas_umbral(obs, prob_i, res_i$umbral)
    message(i, "/", nrow(grilla), ". Hojas ", grilla$hojas[i], ", tasa ",
            grilla$tasa[i], ", mínimo por hoja ", grilla$min_hoja[i], ": F1 ",
            round(met_i$F1, 4), " (",
            round(as.numeric(difftime(Sys.time(), inicio, units = "mins")), 1),
            " minutos).")
    tibble(hojas = grilla$hojas[i], tasa = grilla$tasa[i],
           min_hoja = grilla$min_hoja[i], rondas = grilla$rondas[i], umbral = res_i$umbral,
           precision = met_i$precision, recall = met_i$recall, f1 = met_i$F1)
  })
  rm(x_busqueda); gc()
  write.csv(busqueda, ruta_busqueda, row.names = FALSE)
} else {
  busqueda <- read.csv(ruta_busqueda)
}

# Tabla para las slides: las 8 mejores combinaciones y, si no está entre
# ellas, la elegida, con su posición entre las 36.
busqueda <- busqueda |>
  arrange(desc(f1)) |>
  mutate(posicion = row_number(),
         elegida = hojas == 31 & tasa == 0.04 & min_hoja == 100)
tabla_busqueda <- busqueda |>
  filter(posicion <= 8 | elegida) |>
  transmute(`Posición` = posicion,
            Hojas = hojas,
            `Tasa` = formatC(tasa, format = "f", digits = 2, decimal.mark = ","),
            `Mínimo por hoja` = min_hoja,
            Rondas = format(rondas, big.mark = "."),
            F1 = formatC(f1, format = "f", digits = 4, decimal.mark = ","),
            ` ` = if_else(elegida, "Elegida", ""))
writeLines(knitr::kable(tabla_busqueda, format = "latex", booktabs = TRUE,
                        linesep = "", align = c("r", "r", "r", "r", "r", "r", "l")),
           "02_outputs/tables/busqueda_lgbm.tex")
message("F1 de las 36 combinaciones: entre ",
        round(min(busqueda$f1), 4), " y ", round(max(busqueda$f1), 4), ".")
print(busqueda |> arrange(desc(f1)))

# Variantes ---------------------------------------------------------------------
# Mejor modelo: conjunto ampliado X_7.
lgbm_X7 <- fun_lgbm_variante("X_7_ampliado")
# Conjunto común X_5_extendido, con el que se comparan todos los algoritmos.
lgbm_X5 <- fun_lgbm_variante("X_5_extendido")
# Modelo con X_6 para comparación 
lgbm_X6 <- fun_lgbm_variante("X_6_detallado")
