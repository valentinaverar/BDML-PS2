##########################################################
# 11_model_boosting.R
#
# Objetivo: Entrenar variantes de Boosting y generar sus
# predicciones para Kaggle.
#
# Etapa del Problem Set: Etapa 2: Training and Comparing Algorithms.
# Insumos:  03_temp/ (bases de modelación y particiones de validación cruzada).
# Salidas:  02_outputs/predictions/ (un .csv por variante), una fila por variante en
#           02_outputs/tables/registro_modelos.csv y el modelo en 03_temp/.
# ESTADO: pendiente. Se escribe cuando el tema se vea en clase.
##########################################################
#
# Este script se puede correr solo (no requiere el script maestro), siempre
# que antes se hayan corrido los scripts 01 a 05. Así se puede iterar con
# Kaggle sin repetir la preparación de los datos.
#
# Estructura de cada variante (un envío a Kaggle por variante):
#   1. Entrenar con train() usando las particiones de 03_temp/cv_folds.rds.
#   2. Elegir el umbral que maximiza F1 en las predicciones fuera de muestra.
#   3. Predecir probabilidades en el test.
#   4. Guardar el envío con fun_save_submission().
##########################################################

# Preparación -------------------------------------------------------------
# Código pendiente: cargar paquetes, funciones de 01_code/functions/,
# datos de 03_temp/ y particiones de validación cruzada.


######### XGBOOST PILOTO

if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, caret, dplyr, xgboost)


##########################################################
# 08_model_xgboost.R
#
# XGBOOST - PILOTO Y SUBMISSION KAGGLE
#
# 1. Cargar y validar datos
# 2. Construir matrices
# 3. Entrenar con CV de 5 folds
# 4. Early stopping interno
# 5. Seleccionar umbral mediante F1 OOF
# 6. Entrenar modelo final con todo Train
# 7. Generar submission
#
# No utiliza Holdout.
# No guarda modelos ni tablas.
# Única salida: CSV de Kaggle.
##########################################################


# 0. PAQUETES Y FUNCIONES ----------------------------------

library(xgboost)
library(caret)
library(dplyr)

source("01_code/functions/fun_metricas.R")
source("01_code/functions/fun_save_submission.R")
source("01_code/functions/fun_variables_lgbm.R")


# Función de una variante ----------------------------------
# Todo el procedimiento del piloto queda dentro de una función que recibe el
# conjunto de predictores. Así se corre la misma configuración con distintos
# conjuntos sin duplicar el código, y cada envío anterior se sigue
# reproduciendo con su mismo nombre de archivo.
#   nombre_conjunto: conjunto de 03_temp/predictores.rds.
#   prefijo_nombre:  texto que se agrega al nombre del archivo después de
#                    "XGB_" ("" para conservar el nombre del primer envío).
fun_xgb_piloto <- function(nombre_conjunto, prefijo_nombre = "") {
  
  # 1. CARGAR DATOS ------------------------------------------
  
  train <- readRDS("03_temp/train_clean.rds")
  test <- readRDS("03_temp/test_clean.rds")
  
  predictores <- readRDS("03_temp/predictores.rds")
  cv_folds <- readRDS("03_temp/cv_folds.rds")
  
  
  # 2. VALIDAR VARIABLES -------------------------------------
  
  if (!is.list(predictores) ||
      !nombre_conjunto %in% names(predictores)) {
    
    cat("\nConjuntos disponibles en predictores.rds:\n")
    print(names(predictores))
    
    stop(
      "No existe X_5_ampliado en predictores.rds. ",
      "Revisa si cambió el nombre del conjunto.",
      call. = FALSE
    )
  }
  
  variables_xgb <- predictores[[nombre_conjunto]]
  
  if (!is.character(variables_xgb) ||
      length(variables_xgb) == 0 ||
      anyNA(variables_xgb) ||
      anyDuplicated(variables_xgb)) {
    
    stop(
      "X_5_ampliado debe contener un vector de ",
      "nombres de variables, sin NA ni duplicados.",
      call. = FALSE
    )
  }
  
  # El conjunto X_7 no está en train_clean/test_clean sino en sus propias
  # bases (03_temp/train_X7.rds y test_X7.rds, creadas en 02).
  
  es_x7 <- nombre_conjunto == "X_7_ampliado"
  
  if (es_x7) {
    base_train_x7 <- readRDS("03_temp/train_X7.rds")
    base_test_x7 <- readRDS("03_temp/test_X7.rds")
    nombres_train <- names(base_train_x7)
    nombres_test <- names(base_test_x7)
  } else {
    nombres_train <- names(train)
    nombres_test <- names(test)
  }
  
  # Verificar que todas las variables existan en Train y Test.
  
  faltantes_train <- setdiff(
    variables_xgb, nombres_train
  )
  
  faltantes_test <- setdiff(
    variables_xgb, nombres_test
  )
  
  if (length(faltantes_train) > 0 ||
      length(faltantes_test) > 0) {
    
    cat("\nVariables faltantes en Train:\n")
    print(faltantes_train)
    
    cat("\nVariables faltantes en Test:\n")
    print(faltantes_test)
    
    stop(
      "Hay variables faltantes en las bases. ",
      "Revisar preprocesamiento y predictores.rds.",
      call. = FALSE
    )
  }
  
  stopifnot(
    identical(levels(train$Pobre), c("Yes", "No")),
    "id" %in% names(test)
  )
  
  cat("\n=== DATOS ===\n")
  cat("Hogares Train:", nrow(train), "\n")
  cat("Hogares Test:", nrow(test), "\n")
  cat("Predictores:", length(variables_xgb), "\n")
  
  
  # 3. CONSTRUIR MATRICES ------------------------------------
  
  # Utilizamos el mismo procedimiento del piloto original.
  # Aquí ya sabemos que variables_xgb es válido y no está vacío.
  # X_7 ya es numérico: se pasa a matriz con fun_matriz_x7(). Sus hogares
  # están en el mismo orden que train_clean (verificado en 02) y que
  # test_clean (se verifica aquí con el id).
  
  if (es_x7) {
    
    stopifnot(
      nrow(base_train_x7) == nrow(train),
      identical(base_test_x7$id, test$id)
    )
    
    matriz_train <- fun_matriz_x7(base_train_x7, variables_xgb)
    matriz_test <- fun_matriz_x7(base_test_x7, variables_xgb)
    rm(base_train_x7, base_test_x7)
    gc()
    
  } else {
    
    matriz_train <- model.matrix(
      ~ .,
      data = train[, variables_xgb, drop = FALSE]
    )[, -1, drop = FALSE]
    
    matriz_test <- model.matrix(
      ~ .,
      data = test[, variables_xgb, drop = FALSE]
    )[, -1, drop = FALSE]
    
  }
  
  y <- as.integer(train$Pobre == "Yes")
  
  # X_6 y los conjuntos anteriores no tienen NA. X_7 sí (por ejemplo, la edad
  # del cónyuge en hogares sin cónyuge): XGBoost los maneja directamente,
  # aprendiendo en cada corte hacia qué lado enviar los faltantes.
  stopifnot(
    identical(
      colnames(matriz_train),
      colnames(matriz_test)
    ),
    nrow(matriz_train) == nrow(train),
    nrow(matriz_test) == nrow(test),
    es_x7 || !anyNA(matriz_train),
    es_x7 || !anyNA(matriz_test),
    !any(is.infinite(matriz_train)),
    !any(is.infinite(matriz_test))
  )
  
  cat("Columnas de matriz:", ncol(matriz_train), "\n")
  
  
  # 4. VERIFICAR LOS CINCO FOLDS ------------------------------
  
  stopifnot(length(cv_folds) == 5)
  
  # Cada elemento debe contener los índices de entrenamiento.
  
  stopifnot(all(vapply(
    cv_folds,
    function(idx) {
      is.numeric(idx) &&
        length(idx) > 0 &&
        !anyNA(idx) &&
        !anyDuplicated(idx) &&
        all(idx >= 1 & idx <= nrow(train))
    },
    logical(1)
  )))
  
  indices_validacion <- unlist(lapply(
    cv_folds,
    function(idx) {
      setdiff(seq_len(nrow(train)), idx)
    }
  ))
  
  cat("\n=== VERIFICACIÓN DE FOLDS ===\n")
  cat("Hogares Train:", nrow(train), "\n")
  cat("Validaciones totales:", length(indices_validacion), "\n")
  cat("Hogares únicos:", length(unique(indices_validacion)), "\n")
  
  if (length(indices_validacion) != nrow(train) ||
      anyDuplicated(indices_validacion)) {
    
    stop(
      "Los folds NO corresponden a Train completo. ",
      "Vuelve a ejecutar 05_create_cv_folds.R.",
      call. = FALSE
    )
  }
  
  cat("Folds correctos.\n")
  
  
  # 5. HIPERPARÁMETROS ---------------------------------------
  
  # Exactamente la configuración original del piloto.
  
  parametros <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    tree_method = "hist",
    max_depth = 5,
    eta = 0.10,
    min_child_weight = 5,
    subsample = 0.8,
    colsample_bytree = 0.8,
    nthread = 8
  )
  
  nrounds_max <- 600
  early_stopping <- 30
  
  
  # 6. VALIDACIÓN CRUZADA ------------------------------------
  
  prob_oof <- rep(NA_real_, nrow(train))
  
  tiempos <- numeric(length(cv_folds))
  mejores_rondas <- integer(length(cv_folds))
  
  set.seed(2026)
  
  inicio_total <- Sys.time()
  
  for (k in seq_along(cv_folds)) {
    
    cat("\n================================\n")
    cat("XGBOOST - FOLD", k, "DE 5\n")
    cat("================================\n")
    
    inicio_fold <- Sys.time()
    
    
    # 6A. PARTICIÓN EXTERNA ---------------------------------
    
    idx_train <- cv_folds[[k]]
    
    idx_valid <- setdiff(
      seq_len(nrow(train)),
      idx_train
    )
    
    
    # 6B. PARTICIÓN INTERNA PARA EARLY STOPPING -------------
    
    idx_inner <- as.vector(
      caret::createDataPartition(
        train$Pobre[idx_train],
        p = 0.85,
        list = FALSE
      )
    )
    
    idx_ajuste <- idx_train[idx_inner]
    idx_parada <- idx_train[-idx_inner]
    
    
    # 6C. MATRICES XGBOOST ----------------------------------
    
    dtrain <- xgb.DMatrix(
      data = matriz_train[idx_ajuste, , drop = FALSE],
      label = y[idx_ajuste]
    )
    
    dparada <- xgb.DMatrix(
      data = matriz_train[idx_parada, , drop = FALSE],
      label = y[idx_parada]
    )
    
    dvalid <- xgb.DMatrix(
      data = matriz_train[idx_valid, , drop = FALSE]
    )
    
    
    # 6D. ENTRENAMIENTO -------------------------------------
    
    modelo <- xgb.train(
      params = parametros,
      data = dtrain,
      nrounds = nrounds_max,
      evals = list(validacion_interna = dparada),
      early_stopping_rounds = early_stopping,
      maximize = FALSE,
      verbose = 0
    )
    
    
    # 6E. MEJOR ITERACIÓN -----------------------------------
    
    mejor_ronda <- attr(modelo, "best_iteration")
    
    if (length(mejor_ronda) == 0) {
      mejor_ronda <- modelo$best_iteration
    }
    
    if (length(mejor_ronda) == 0) {
      
      valor <- xgboost::xgb.attr(
        modelo,
        "best_iteration"
      )
      
      if (length(valor) == 1) {
        mejor_ronda <- as.integer(valor) + 1L
      }
    }
    
    if (length(mejor_ronda) != 1 ||
        is.na(mejor_ronda)) {
      
      stop(
        "No se pudo recuperar la mejor ronda ",
        "del entrenamiento."
      )
    }
    
    mejores_rondas[k] <- as.integer(mejor_ronda)
    
    
    # 6F. PREDICCIONES OOF ----------------------------------
    
    prob_oof[idx_valid] <- as.numeric(
      predict(modelo, dvalid)
    )
    
    
    # 6G. TIEMPOS -------------------------------------------
    
    tiempos[k] <- as.numeric(
      difftime(
        Sys.time(),
        inicio_fold,
        units = "secs"
      )
    )
    
    cat("Mejor ronda:", mejores_rondas[k], "\n")
    cat("Tiempo:", round(tiempos[k], 1), "segundos\n")
    
    rm(modelo, dtrain, dparada, dvalid)
    
    gc(verbose = FALSE)
  }
  
  
  # 7. SELECCIÓN DEL UMBRAL -----------------------------------
  
  stopifnot(
    !anyNA(prob_oof),
    all(is.finite(prob_oof)),
    all(prob_oof >= 0 & prob_oof <= 1)
  )
  
  umbrales <- seq(0.15, 0.50, by = 0.01)
  
  resultados_f1 <- data.frame(
    Umbral = umbrales,
    F1 = vapply(
      umbrales,
      function(u) {
        
        fun_f1_score(
          .actual = train$Pobre,
          .predicted = ifelse(
            prob_oof >= u,
            "Yes",
            "No"
          ),
          .positive = "Yes"
        )
      },
      numeric(1)
    )
  ) %>%
    arrange(desc(F1))
  
  mejor <- resultados_f1[1, , drop = FALSE]
  
  umbral_final <- mejor$Umbral
  f1_oof <- mejor$F1
  
  
  # 8. RONDAS DEL MODELO DEFINITIVO ---------------------------
  
  nrounds_final <- as.integer(
    round(mean(mejores_rondas))
  )
  
  cat("\n=== RONDAS SELECCIONADAS ===\n")
  cat("Por fold:", mejores_rondas, "\n")
  cat("Promedio:", nrounds_final, "\n")
  
  
  # 9. ENTRENAMIENTO CON TODO TRAIN ---------------------------
  
  dtrain_full <- xgb.DMatrix(
    data = matriz_train,
    label = y
  )
  
  dtest <- xgb.DMatrix(
    data = matriz_test
  )
  
  set.seed(2026)
  
  modelo_final <- xgb.train(
    params = parametros,
    data = dtrain_full,
    nrounds = nrounds_final,
    verbose = 0
  )
  
  
  # 10. PREDICCIÓN SOBRE TEST ---------------------------------
  
  prob_test <- as.numeric(
    predict(modelo_final, dtest)
  )
  
  stopifnot(
    length(prob_test) == nrow(test),
    all(is.finite(prob_test)),
    all(prob_test >= 0 & prob_test <= 1)
  )
  
  
  # 11. GENERAR SUBMISSION ------------------------------------
  
  # Formato consistente con las convenciones del proyecto.
  
  nombre_submission <- paste0(
    "XGB_", prefijo_nombre, "depth_", parametros$max_depth,
    "_eta_010",
    "_mcw_", parametros$min_child_weight,
    "_rounds_", nrounds_final
  )
  
  submission_xgb <- fun_save_submission(
    .db_test = test,
    .prob = prob_test,
    .umbral = umbral_final,
    .nombre = nombre_submission
  )
  
  # Registro con las predicciones fuera de muestra, para la comparación de
  # algoritmos.
  fun_registrar_oof(
    .archivo = nombre_submission,
    .algoritmo = "Boosting (XGBoost)",
    .conjunto = nombre_conjunto,
    .n_predictores = length(variables_xgb),
    .hiperparametros = paste0("depth = ", parametros$max_depth, "; eta = ",
                              parametros$eta, "; mcw = ",
                              parametros$min_child_weight, "; rondas = ",
                              nrounds_final),
    .obs = train$Pobre,
    .prob = prob_oof,
    .umbral = umbral_final
  )
  
  
  # 12. RESUMEN FINAL -----------------------------------------
  
  tiempo_total <- as.numeric(
    difftime(
      Sys.time(),
      inicio_total,
      units = "mins"
    )
  )
  
  cat("\n========================================\n")
  cat("           RESUMEN XGBOOST\n")
  cat("========================================\n")
  
  cat("\nDATOS:\n")
  cat("Hogares Train:", nrow(train), "\n")
  cat("Hogares Test:", nrow(test), "\n")
  cat("Predictores:", length(variables_xgb), "\n")
  cat("Columnas:", ncol(matriz_train), "\n")
  
  cat("\nVALIDACIÓN CRUZADA:\n")
  cat("Mejores rondas:", mejores_rondas, "\n")
  cat("Rondas finales:", nrounds_final, "\n")
  cat("Segundos por fold:", round(tiempos, 1), "\n")
  
  cat("\nDESEMPEÑO:\n")
  cat("Umbral óptimo:", umbral_final, "\n")
  cat("F1 OOF:", round(f1_oof, 7), "\n")
  
  cat("\nTOP 10 UMBRALES:\n")
  print(head(resultados_f1, 10), digits = 7)
  
  cat("\nSUBMISSION:\n")
  cat("Pobres predichos:", sum(submission_xgb$pobre), "\n")
  cat("Proporción pobre:",
      round(mean(submission_xgb$pobre), 4), "\n")
  
  cat("\nArchivo generado:\n")
  cat(
    paste0(
      "02_outputs/predictions/",
      nombre_submission,
      ".csv\n"
    )
  )
  
  cat("\nTiempo total:",
      round(tiempo_total, 2),
      "minutos\n")
  
  cat("\nProceso terminado.\n")
  
  invisible(list(prob_oof = prob_oof, prob_test = prob_test,
                 umbral = umbral_final, f1_oof = f1_oof,
                 archivo = nombre_submission))
}

# Variantes del piloto ---------------------------------------
# Primer envío (Kaggle 0,73): configuración piloto con X_5_extendido.
xgb_X5 <- fun_xgb_piloto("X_5_extendido")
# Misma configuración con X_6_detallado, que agrega los conteos de personas
# por categoría.
xgb_X6 <- fun_xgb_piloto("X_6_detallado", "X_6_detallado_")
# Misma configuración con X_7_ampliado, el conjunto de LightGBM: compara los
# dos algoritmos con las mismas variables.
xgb_X7 <- fun_xgb_piloto("X_7_ampliado", "X_7_ampliado_")



############## xGBOOST EXHAUSTIVO

# La búsqueda aleatoria de 250 configuraciones tarda varias horas. Se deja
# desactivada por defecto para que correr este script no la dispare sin
# querer. Para reproducir el envío "search250", cambiar a TRUE: el proceso
# guarda su avance en 03_temp/ y se puede reanudar.
EJECUTAR_BUSQUEDA <- FALSE

if (EJECUTAR_BUSQUEDA) {
  
  library(xgboost)
  library(caret)
  library(dplyr)
  
  source("01_code/functions/fun_metricas.R")
  source("01_code/functions/fun_importancia_variables.R")
  source("01_code/functions/fun_save_submission.R")
  
  # 1. CONFIGURACIÓN -----------------------------------------
  SEMILLA <- 2026L
  N_EXPLORACION <- 200L
  N_REFINAMIENTO <- 50L
  N_RONDAS_MAX <- 2000L
  EARLY_STOPPING <- 50L
  NTHREAD <- 8L
  N_PERM <- 3L
  UMBRALES <- seq(0.10, 0.85, by = 0.01)
  MAX_ERRORES_CONSECUTIVOS <- 3L
  VERSION_SCRIPT <- "xgb-checkpoint-v1"
  
  RUTA_CHECKPOINT <- "03_temp/checkpoint_xgb_search250.rds"
  RUTA_BACKUP <- paste0(RUTA_CHECKPOINT, ".bak")
  RUTAS_DATOS <- c(
    train = "03_temp/train_clean.rds",
    test = "03_temp/test_clean.rds",
    predictores = "03_temp/predictores.rds",
    folds = "03_temp/cv_folds.rds"
  )
  
  # 2. COMPROBACIONES ANTES DE ENTRENAR ----------------------
  stopifnot(all(file.exists(RUTAS_DATOS)))
  stopifnot(file.exists("00_data/raw/sample_submission.csv"))
  
  train <- readRDS(RUTAS_DATOS["train"])
  test <- readRDS(RUTAS_DATOS["test"])
  predictores <- readRDS(RUTAS_DATOS["predictores"])
  cv_folds <- readRDS(RUTAS_DATOS["folds"])
  
  # Es el nombre ACTUAL del conjunto, no X_5_ampliado.
  if (!is.list(predictores) ||
      !"X_5_extendido" %in% names(predictores)) {
    print(names(predictores))
    stop("No existe X_5_extendido en predictores.rds.", call. = FALSE)
  }
  variables_xgb <- predictores[["X_5_extendido"]]
  
  stopifnot(
    is.character(variables_xgb), length(variables_xgb) > 0L,
    !anyNA(variables_xgb), !anyDuplicated(variables_xgb),
    all(variables_xgb %in% names(train)),
    all(variables_xgb %in% names(test)),
    identical(levels(train$Pobre), c("Yes", "No")),
    "id" %in% names(test),
    length(cv_folds) == 5L
  )
  
  # Comprobar la submission AHORA, no después de horas.
  ids_sample <- utils::read.csv(
    "00_data/raw/sample_submission.csv"
  )$id
  stopifnot(!anyDuplicated(test$id), setequal(test$id, ids_sample))
  
  matriz_train <- model.matrix(
    ~ ., data = train[, variables_xgb, drop = FALSE]
  )[, -1, drop = FALSE]
  matriz_test <- model.matrix(
    ~ ., data = test[, variables_xgb, drop = FALSE]
  )[, -1, drop = FALSE]
  y <- as.integer(train$Pobre == "Yes")
  
  stopifnot(
    identical(colnames(matriz_train), colnames(matriz_test)),
    nrow(matriz_train) == nrow(train),
    nrow(matriz_test) == nrow(test),
    !anyNA(matriz_train), !anyNA(matriz_test),
    all(is.finite(matriz_train)), all(is.finite(matriz_test))
  )
  
  stopifnot(all(vapply(cv_folds, function(idx) {
    is.numeric(idx) && length(idx) > 0L && !anyNA(idx) &&
      !anyDuplicated(idx) && all(idx >= 1 & idx <= nrow(train))
  }, logical(1))))
  
  indices_validacion <- unlist(lapply(cv_folds, function(idx) {
    setdiff(seq_len(nrow(train)), idx)
  }))
  stopifnot(
    length(indices_validacion) == nrow(train),
    !anyDuplicated(indices_validacion),
    setequal(indices_validacion, seq_len(nrow(train)))
  )
  
  cat("Train:", nrow(train), "| Test:", nrow(test),
      "| Variables:", length(variables_xgb),
      "| Columnas:", ncol(matriz_train), "\n")
  cat("Predictores, Test y cinco folds: verificados.\n")
  
  # 3. CHECKPOINTS Y SEGURIDAD -------------------------------
  # Huella de los archivos: impide reanudar con otros datos/folds.
  firma_actual <- list(
    version = VERSION_SCRIPT,
    md5 = unname(tools::md5sum(RUTAS_DATOS)),
    variables = variables_xgb,
    xgboost_version = as.character(utils::packageVersion("xgboost")),
    semilla = SEMILLA,
    n_exploracion = N_EXPLORACION,
    n_refinamiento = N_REFINAMIENTO,
    n_rondas_max = N_RONDAS_MAX,
    early_stopping = EARLY_STOPPING,
    umbrales = UMBRALES
  )
  
  leer_seguro <- function(ruta) {
    if (!file.exists(ruta)) return(NULL)
    tryCatch(readRDS(ruta), error = function(e) NULL)
  }
  
  guardar_checkpoint <- function(estado) {
    # Se escribe primero un temporal; el anterior queda como respaldo.
    temporal <- tempfile("xgb_ckpt_", tmpdir = "03_temp", fileext = ".rds")
    on.exit(unlink(temporal), add = TRUE)
    saveRDS(estado, temporal)
    if (is.null(leer_seguro(temporal))) {
      stop("No se pudo verificar el checkpoint temporal.")
    }
    if (!is.null(leer_seguro(RUTA_CHECKPOINT))) {
      if (!file.copy(RUTA_CHECKPOINT, RUTA_BACKUP, overwrite = TRUE)) {
        stop("No se pudo crear la copia de seguridad del checkpoint.")
      }
    }
    if (!file.copy(temporal, RUTA_CHECKPOINT, overwrite = TRUE) ||
        is.null(leer_seguro(RUTA_CHECKPOINT))) {
      stop("No se pudo guardar el checkpoint; comprueba el espacio en disco.")
    }
    invisible(NULL)
  }
  
  estado <- leer_seguro(RUTA_CHECKPOINT)
  if (is.null(estado)) {
    estado <- leer_seguro(RUTA_BACKUP)
    if (!is.null(estado)) cat("Recuperando checkpoint de respaldo.\n")
  }
  
  if (!is.null(estado)) {
    if (!identical(estado$firma, firma_actual)) {
      stop(
        "El checkpoint pertenece a otros datos, folds o parámetros. ",
        "NO lo mezcles: muévelo o renómbralo antes de iniciar otra búsqueda.",
        call. = FALSE
      )
    }
    cat("CHECKPOINT RECUPERADO:", nrow(estado$registro),
        "configuraciones ya procesadas.\n")
  } else {
    estado <- list(
      firma = firma_actual,
      exploracion = NULL,
      refinamiento = NULL,
      registro = data.frame(),
      mejor = NULL,
      submission_hecha = FALSE,
      archivo_submission = NULL,
      importancia_folds = vector("list", length(cv_folds))
    )
  }
  
  # 4. CONFIGURACIONES DETERMINISTAS -------------------------
  limitar <- function(x, inferior, superior) {
    pmax(inferior, pmin(superior, x))
  }
  aleatorio_log <- function(inferior, superior) {
    exp(runif(1, log(inferior), log(superior)))
  }
  
  generar_aleatorio <- function() {
    list(
      max_depth = sample(2:8, 1),
      eta = aleatorio_log(0.015, 0.30),
      min_child_weight = sample(c(1, 2, 3, 5, 8, 12, 20), 1),
      subsample = runif(1, 0.65, 1),
      colsample_bytree = runif(1, 0.65, 1),
      gamma = runif(1, 0, 5),
      lambda = aleatorio_log(0.05, 25),
      alpha = sample(c(0, 0.1, 0.5, 1, 2, 4), 1),
      scale_pos_weight = sample(c(1, 1.5, 2, 3), 1)
    )
  }
  
  refinar <- function(p) {
    list(
      max_depth = as.integer(limitar(
        p$max_depth + sample(-1:1, 1), 2, 8
      )),
      eta = limitar(p$eta * exp(rnorm(1, 0, 0.30)), 0.015, 0.30),
      min_child_weight = limitar(
        round(p$min_child_weight * exp(rnorm(1, 0, 0.30))), 1, 30
      ),
      subsample = limitar(p$subsample + rnorm(1, 0, 0.07), 0.65, 1),
      colsample_bytree = limitar(
        p$colsample_bytree + rnorm(1, 0, 0.07), 0.65, 1
      ),
      gamma = limitar(p$gamma + rnorm(1, 0, 0.65), 0, 6),
      lambda = limitar(p$lambda * exp(rnorm(1, 0, 0.40)), 0.01, 40),
      alpha = limitar(p$alpha + rnorm(1, 0, 0.40), 0, 5),
      scale_pos_weight = sample(
        c(p$scale_pos_weight, p$scale_pos_weight, 1, 1.5, 2, 3), 1
      )
    )
  }
  
  config_piloto <- list(
    max_depth = 5L, eta = 0.10, min_child_weight = 5,
    subsample = 0.8, colsample_bytree = 0.8,
    gamma = 0, lambda = 1, alpha = 0, scale_pos_weight = 1
  )
  
  if (is.null(estado$exploracion)) {
    set.seed(SEMILLA + 100L)
    estado$exploracion <- c(
      list(config_piloto),
      replicate(N_EXPLORACION - 1L, generar_aleatorio(), simplify = FALSE)
    )
    guardar_checkpoint(estado)
  }
  
  # 5. FOLDS E ÍNDICES INTERNOS REPRODUCIBLES ---------------
  datos_fold <- lapply(seq_along(cv_folds), function(k) {
    idx_entrenamiento <- cv_folds[[k]]
    idx_validacion <- setdiff(seq_len(nrow(train)), idx_entrenamiento)
    set.seed(SEMILLA + k)
    idx_interno <- as.vector(caret::createDataPartition(
      train$Pobre[idx_entrenamiento], p = 0.85, list = FALSE
    ))
    idx_ajuste <- idx_entrenamiento[idx_interno]
    idx_parada <- idx_entrenamiento[-idx_interno]
    list(
      idx_validacion = idx_validacion,
      d_ajuste = xgb.DMatrix(matriz_train[idx_ajuste, , drop = FALSE],
                             label = y[idx_ajuste]),
      d_parada = xgb.DMatrix(matriz_train[idx_parada, , drop = FALSE],
                             label = y[idx_parada]),
      d_entrenamiento = xgb.DMatrix(
        matriz_train[idx_entrenamiento, , drop = FALSE],
        label = y[idx_entrenamiento]
      ),
      d_validacion = xgb.DMatrix(
        matriz_train[idx_validacion, , drop = FALSE]
      )
    )
  })
  
  recuperar_ronda <- function(modelo) {
    ronda <- attr(modelo, "best_iteration")
    if (length(ronda) != 1L) ronda <- modelo$best_iteration
    if (length(ronda) != 1L) {
      atributo <- tryCatch(
        xgboost::xgb.attr(modelo, "best_iteration"),
        error = function(e) NULL
      )
      if (length(atributo) == 1L) ronda <- as.integer(atributo) + 1L
    }
    if (length(ronda) != 1L || !is.finite(ronda) || ronda < 1) {
      stop("No se pudo recuperar best_iteration de XGBoost.")
    }
    as.integer(ronda)
  }
  
  parametros_xgb <- function(p, k, id) {
    c(list(
      objective = "binary:logistic",
      eval_metric = "logloss",
      tree_method = "hist",
      nthread = NTHREAD,
      seed = as.integer(SEMILLA + id * 100L + k)
    ), p)
  }
  
  evaluar_configuracion <- function(p, id) {
    inicio <- Sys.time()
    prob_oof <- rep(NA_real_, nrow(train))
    mejores_rondas <- integer(length(datos_fold))
    f1_por_fold <- numeric(length(datos_fold))
    
    for (k in seq_along(datos_fold)) {
      f <- datos_fold[[k]]
      parametros_k <- parametros_xgb(p, k, id)
      
      modelo_temporal <- xgboost::xgb.train(
        params = parametros_k,
        data = f$d_ajuste,
        nrounds = N_RONDAS_MAX,
        evals = list(validacion_interna = f$d_parada),
        early_stopping_rounds = EARLY_STOPPING,
        maximize = FALSE,
        verbose = 0
      )
      rondas <- recuperar_ronda(modelo_temporal)
      mejores_rondas[k] <- rondas
      rm(modelo_temporal)
      
      # Segundo ajuste con el 80% completo del fold externo.
      modelo_fold <- xgboost::xgb.train(
        params = parametros_k,
        data = f$d_entrenamiento,
        nrounds = rondas,
        verbose = 0
      )
      prob_oof[f$idx_validacion] <- as.numeric(
        predict(modelo_fold, f$d_validacion)
      )
      rm(modelo_fold)
    }
    
    stopifnot(!anyNA(prob_oof), all(is.finite(prob_oof)))
    curva <- vapply(UMBRALES, function(u) {
      valor <- fun_f1_score(
        .actual = train$Pobre,
        .predicted = ifelse(prob_oof >= u, "Yes", "No")
      )
      if (!is.finite(valor)) 0 else as.numeric(valor)
    }, numeric(1))
    posicion <- which.max(curva)
    umbral <- UMBRALES[posicion]
    
    for (k in seq_along(datos_fold)) {
      idx <- datos_fold[[k]]$idx_validacion
      f1_por_fold[k] <- fun_f1_score(
        train$Pobre[idx], ifelse(prob_oof[idx] >= umbral, "Yes", "No")
      )
    }
    
    list(
      F1 = curva[posicion], Umbral = umbral,
      Rondas = mejores_rondas, F1_folds = f1_por_fold,
      Segundos = as.numeric(difftime(Sys.time(), inicio, units = "secs"))
    )
  }
  
  # 6. REGISTRAR CADA CONFIGURACIÓN Y GUARDAR ---------------
  registrar_configuracion <- function(id, etapa, p, resultado, error = NULL) {
    correcta <- is.null(error)
    fila <- data.frame(
      Id = id, Etapa = etapa, Estado = if (correcta) "OK" else "ERROR",
      F1 = if (correcta) resultado$F1 else NA_real_,
      Umbral = if (correcta) resultado$Umbral else NA_real_,
      Rondas = if (correcta) as.integer(round(mean(resultado$Rondas))) else NA_integer_,
      Rondas_folds = if (correcta) paste(resultado$Rondas, collapse = ",") else "",
      DE_F1_folds = if (correcta) stats::sd(resultado$F1_folds) else NA_real_,
      Segundos = if (correcta) resultado$Segundos else NA_real_,
      Error = if (correcta) "" else conditionMessage(error),
      as.data.frame(p, check.names = FALSE),
      check.names = FALSE
    )
    estado$registro <<- rbind(estado$registro, fila)
    if (correcta && (is.null(estado$mejor) ||
                     resultado$F1 > estado$mejor$F1)) {
      estado$mejor <<- list(
        Id = id, F1 = resultado$F1, Umbral = resultado$Umbral,
        Rondas = resultado$Rondas, F1_folds = resultado$F1_folds,
        parametros = p
      )
    }
    guardar_checkpoint(estado)
  }
  
  correr_etapa <- function(configuraciones, etapa, offset = 0L) {
    errores_seguidos <- 0L
    for (i in seq_along(configuraciones)) {
      id <- as.integer(offset + i)
      if (nrow(estado$registro) > 0L && id %in% estado$registro$Id) next
      cat(sprintf("\n[%s %d/%d] Iniciando...\n",
                  etapa, i, length(configuraciones)))
      error_actual <- NULL
      resultado <- tryCatch(
        evaluar_configuracion(configuraciones[[i]], id),
        error = function(e) {
          error_actual <<- e
          NULL
        }
      )
      registrar_configuracion(id, etapa, configuraciones[[i]],
                              resultado, error_actual)
      if (is.null(error_actual)) {
        errores_seguidos <- 0L
        cat(sprintf("OK | F1 %.7f | Mejor %.7f | %.1f s | checkpoint guardado\n",
                    resultado$F1, estado$mejor$F1, resultado$Segundos))
      } else {
        errores_seguidos <- errores_seguidos + 1L
        cat("ERROR registrado (se conserva el avance):",
            conditionMessage(error_actual), "\n")
        if (errores_seguidos >= MAX_ERRORES_CONSECUTIVOS) {
          stop("Tres configuraciones seguidas fallaron. Revisa el error; ",
               "el checkpoint está guardado.", call. = FALSE)
        }
      }
      if (i %% 10L == 0L) gc(verbose = FALSE)
    }
  }
  
  # 7. ETAPA 1: 200 CONFIGURACIONES -------------------------
  inicio_busqueda <- Sys.time()
  correr_etapa(estado$exploracion, "Exploracion")
  
  # 8. GENERAR Y GUARDAR LAS 50 REFINADAS --------------------
  if (is.null(estado$refinamiento)) {
    aprobadas <- subset(estado$registro,
                        Etapa == "Exploracion" & Estado == "OK")
    if (nrow(aprobadas) == 0L) {
      stop("Ninguna configuración funcionó; ver errores en checkpoint.")
    }
    aprobadas <- aprobadas[order(-aprobadas$F1), , drop = FALSE]
    top_ids <- head(aprobadas$Id, 10L)
    set.seed(SEMILLA + 200L)
    estado$refinamiento <- lapply(seq_len(N_REFINAMIENTO), function(i) {
      padre <- sample(top_ids, 1L)
      refinar(estado$exploracion[[padre]])
    })
    guardar_checkpoint(estado)
    cat("50 configuraciones de refinamiento creadas y guardadas.\n")
  }
  
  # 9. ETAPA 2: 50 REFINADAS ---------------------------------
  correr_etapa(estado$refinamiento, "Refinamiento", N_EXPLORACION)
  
  # 10. RESULTADOS Y MEJOR CONFIGURACIÓN ---------------------
  registro_xgb <- estado$registro %>% arrange(desc(F1))
  cat("\n=========== TOP 15 XGBOOST ===========\n")
  print(as.data.frame(head(subset(registro_xgb, Estado == "OK"), 15L)),
        digits = 7, row.names = FALSE)
  cat("Configuraciones completadas:", nrow(estado$registro),
      "| Exitosas:", sum(estado$registro$Estado == "OK"), "\n")
  if (is.null(estado$mejor)) stop("No hay modelos exitosos.")
  mejor_xgb <- estado$mejor
  mejor_parametros_xgb <- mejor_xgb$parametros
  cat("Mejor F1 OOF:", round(mejor_xgb$F1, 7),
      "| Umbral:", mejor_xgb$Umbral, "\n")
  cat("Rondas por fold:", mejor_xgb$Rondas, "\n")
  
  # 11. SUBMISSION PRIMERO (ANTES DE IMPORTANCIA) ------------
  nrounds_final <- as.integer(round(mean(mejor_xgb$Rondas)))
  codigo_decimal <- function(z) {
    gsub(".", "p", format(round(z, 4), scientific = FALSE,
                          trim = TRUE), fixed = TRUE)
  }
  nombre_submission <- paste0(
    "XGB_depth_", mejor_parametros_xgb$max_depth,
    "_eta_", codigo_decimal(mejor_parametros_xgb$eta),
    "_mcw_", codigo_decimal(mejor_parametros_xgb$min_child_weight),
    "_rounds_", nrounds_final, "_search250"
  )
  ruta_submission <- file.path(
    "02_outputs/predictions", paste0(nombre_submission, ".csv")
  )
  
  if (!isTRUE(estado$submission_hecha) || !file.exists(ruta_submission)) {
    cat("\nEntrenando modelo definitivo con todo Train...\n")
    dtrain_full <- xgb.DMatrix(matriz_train, label = y)
    dtest <- xgb.DMatrix(matriz_test)
    modelo_xgb_final <- xgb.train(
      params = parametros_xgb(mejor_parametros_xgb, 0L, mejor_xgb$Id),
      data = dtrain_full, nrounds = nrounds_final, verbose = 0
    )
    prob_test_xgb <- as.numeric(predict(modelo_xgb_final, dtest))
    stopifnot(
      length(prob_test_xgb) == nrow(test),
      all(is.finite(prob_test_xgb)),
      all(prob_test_xgb >= 0 & prob_test_xgb <= 1)
    )
    submission_xgb <- fun_save_submission(
      .db_test = test,
      .prob = prob_test_xgb,
      .umbral = mejor_xgb$Umbral,
      .nombre = nombre_submission
    )
    estado$submission_hecha <- TRUE
    estado$archivo_submission <- ruta_submission
    guardar_checkpoint(estado)
    cat("SUBMISSION GUARDADA:", ruta_submission, "\n")
  } else {
    cat("\nSubmission ya creada:", ruta_submission, "\n")
  }
  
  # 12. IMPORTANCIA DEL GANADOR, TAMBIÉN REANUDABLE ----------
  predecir_xgb <- function(modelo, nuevos_datos) {
    nueva_matriz <- model.matrix(
      ~ ., data = nuevos_datos[, variables_xgb, drop = FALSE]
    )[, -1, drop = FALSE]
    stopifnot(identical(colnames(nueva_matriz), colnames(matriz_train)))
    as.numeric(predict(modelo, xgb.DMatrix(nueva_matriz)))
  }
  
  for (k in seq_along(datos_fold)) {
    if (!is.null(estado$importancia_folds[[k]])) next
    cat("Calculando importancia del fold", k, "...\n")
    f <- datos_fold[[k]]
    resultado_importancia <- tryCatch({
      modelo_imp <- xgb.train(
        params = parametros_xgb(mejor_parametros_xgb, k, mejor_xgb$Id),
        data = f$d_entrenamiento,
        nrounds = mejor_xgb$Rondas[k], verbose = 0
      )
      fun_importancia_variables(
        .modelo = modelo_imp,
        .datos = train[f$idx_validacion, variables_xgb, drop = FALSE],
        .real = train$Pobre[f$idx_validacion],
        .predecir = predecir_xgb,
        .umbral = mejor_xgb$Umbral,
        .n_perm = N_PERM,
        .semilla = SEMILLA + k
      )
    }, error = function(e) {
      warning("No se pudo calcular importancia del fold ", k,
              ": ", conditionMessage(e), call. = FALSE)
      NULL
    })
    if (!is.null(resultado_importancia)) {
      resultado_importancia$Fold <- k
      estado$importancia_folds[[k]] <- resultado_importancia
      guardar_checkpoint(estado)
      cat("Importancia fold", k, ": checkpoint guardado.\n")
    }
    gc(verbose = FALSE)
  }
  
  if (all(vapply(estado$importancia_folds, Negate(is.null), logical(1)))) {
    importancia_xgb_folds <- dplyr::bind_rows(estado$importancia_folds)
    importancia_xgb_final <- importancia_xgb_folds %>%
      group_by(Variable) %>%
      summarise(
        F1_original = mean(F1_original),
        F1_permutado = mean(F1_permutado),
        Importancia_F1 = mean(Importancia_F1),
        DE_importancia = sd(Importancia_F1),
        Repeticiones = sum(Repeticiones),
        .groups = "drop"
      ) %>%
      arrange(desc(Importancia_F1))
    cat("\n=========== TOP 15 IMPORTANCIA ===========\n")
    print(as.data.frame(head(importancia_xgb_final, 15L)),
          digits = 7, row.names = FALSE)
  } else {
    cat("Importancia parcial: puedes volver a ejecutar para reintentar folds pendientes.\n")
  }
  
  cat("\n=========== RESUMEN DEFINITIVO ===========\n")
  cat("Mejor F1 OOF:", round(mejor_xgb$F1, 7), "\n")
  cat("Mejor umbral:", mejor_xgb$Umbral, "\n")
  cat("Rondas finales:", nrounds_final, "\n")
  cat("CSV:", ruta_submission, "\n")
  cat("Resultados guardados en:", RUTA_CHECKPOINT, "\n")
  cat("Horas en esta ejecución:",
      round(as.numeric(difftime(Sys.time(), inicio_busqueda,
                                units = "hours")), 2), "\n")
  cat("Proceso terminado.\n")
}
