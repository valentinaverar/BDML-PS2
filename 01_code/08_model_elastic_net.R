##########################################################
# 06_model_elastic_net.R
#
# Objetivo:
#   Entrenar Elastic Net logístico para predecir pobreza,
#   seleccionar alpha, lambda y umbral mediante F1 OOF,
#   evaluar la sensibilidad a Bogotá y generar submissions
#   para Kaggle.
#
# Etapa:
#   Training and Comparing Algorithms.
#
# Insumos:
#   03_temp/train_clean.rds
#   03_temp/test_clean.rds
#   03_temp/predictores.rds
#   03_temp/cv_folds.rds
#
# Funciones:
#   fun_metricas.R
#   fun_save_submission.R
#
# Salidas:
#   Únicamente archivos CSV en:
#   02_outputs/predictions/
#
# No guarda modelos ni tablas adicionales.
##########################################################


# 0. PREPARACIÓN --------------------------------------------

if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, caret, dplyr, pROC, glmnet)

source("01_code/functions/fun_metricas.R")
source("01_code/functions/fun_save_submission.R")

# Reutilizar CV existente cuando esté disponible. Se deja en TRUE: el
# archivo guardado en 03_temp/ puede venir de otro conjunto de predictores.
REENTRENAR_CV <- TRUE

# Ejecutar pruebas geográficas de sensibilidad.
CORRER_PRUEBAS_BOGOTA <- TRUE


# 1. CARGAR DATOS -------------------------------------------

train <- readRDS("03_temp/train_clean.rds")
test <- readRDS("03_temp/test_clean.rds")

predictores <- readRDS("03_temp/predictores.rds")
cv_folds_full <- readRDS("03_temp/cv_folds.rds")

X <- predictores$X_5_extendido

stopifnot(
  all(X %in% names(train)),
  all(X %in% names(test)),
  identical(levels(train$Pobre), c("Yes", "No")),
  length(cv_folds_full) == 5,
  all(vapply(
    cv_folds_full,
    function(i) all(i >= 1 & i <= nrow(train)),
    logical(1)
  ))
)


# 2. CONSTRUIR MATRICES -------------------------------------

matriz_train <- model.matrix(
  ~ ., data = train[, X, drop = FALSE]
)[, -1, drop = FALSE]

matriz_test <- model.matrix(
  ~ ., data = test[, X, drop = FALSE]
)[, -1, drop = FALSE]

stopifnot(
  identical(colnames(matriz_train),
            colnames(matriz_test)),
  !anyNA(matriz_train),
  !anyNA(matriz_test)
)

y_train <- as.integer(train$Pobre == "Yes")


# 3. FUNCIONES AUXILIARES -----------------------------------

# Ajustar un Elastic Net logístico.
ajustar_enet <- function(matriz, respuesta,
                         alpha, lambda) {
  
  glmnet(
    x = matriz,
    y = respuesta,
    family = "binomial",
    alpha = alpha,
    lambda = lambda,
    standardize = TRUE
  )
}

# Obtener probabilidades de pobreza.
predecir_enet <- function(modelo, matriz, lambda) {
  
  as.numeric(
    predict(
      modelo,
      newx = matriz,
      s = lambda,
      type = "response"
    )
  )
}

# Construir nombres de submissions.
# Se utiliza "p" como separador decimal.
#
# Ejemplos:
#   lambda = 0.0001 -> 0p0001
#   alpha = 0.5     -> 05
#   alpha = 1       -> 1

codigo_lambda <- function(lambda) {
  
  gsub(
    ".", "p",
    format(lambda, scientific = FALSE, trim = TRUE),
    fixed = TRUE
  )
}

codigo_alpha <- function(alpha) {
  
  sub(
    "^0\\.", "0",
    format(alpha, scientific = FALSE, trim = TRUE)
  )
}

nombre_enet <- function(lambda, alpha, sufijo = NULL) {
  
  nombre <- paste0(
    "EN_lambda_", codigo_lambda(lambda),
    "_alpha_", codigo_alpha(alpha)
  )
  
  if (!is.null(sufijo)) {
    nombre <- paste0(nombre, "_", sufijo)
  }
  
  nombre
}


# 4. VALIDACIÓN CRUZADA -------------------------------------

# Se utilizan todos los hogares etiquetados de Train.
#
# La validación es estratificada según Pobre.
# Se mantienen los cinco folds previamente generados.

grid_enet <- expand.grid(
  alpha = c(0.25, 0.50, 0.75, 1),
  lambda = c(0.0001, 0.0005, 0.001)
)

umbrales <- seq(0.20, 0.45, by = 0.01)

# No usamos fun_crear_ctrl() aquí porque esa función
# guarda únicamente las predicciones de la mejor
# combinación según la métrica de caret.
#
# Necesitamos las predicciones de TODAS las combinaciones
# para seleccionar alpha, lambda y umbral mediante F1.

control_enet <- trainControl(
  method = "cv",
  number = 5,
  index = cv_folds_full,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "all",
  allowParallel = TRUE
)

ruta_cv <- "03_temp/modelo_cv_full.rds"

if (!REENTRENAR_CV && file.exists(ruta_cv)) {
  
  modelo_cv <- readRDS(ruta_cv)
  
  cat("\nCV existente recuperada.\n")
  
} else {
  
  set.seed(2026)
  
  modelo_cv <- caret::train(
    x = matriz_train,
    y = train$Pobre,
    method = "glmnet",
    family = "binomial",
    metric = "ROC",
    trControl = control_enet,
    tuneGrid = grid_enet,
    standardize = TRUE
  )
  
  cat("\nValidación cruzada completada.\n")
}

# Comprobar que existen predicciones de las 12
# combinaciones para todo Train.

stopifnot(
  !is.null(modelo_cv$pred),
  nrow(modelo_cv$pred) ==
    nrow(train) * nrow(grid_enet),
  setequal(
    paste(modelo_cv$pred$alpha,
          modelo_cv$pred$lambda),
    paste(grid_enet$alpha,
          grid_enet$lambda)
  ),
  setequal(
    modelo_cv$pred$rowIndex,
    seq_len(nrow(train))
  )
)


# 5. SELECCIONAR ALPHA, LAMBDA Y UMBRAL POR F1 --------------

# Se utiliza fun_f1_score() de fun_metricas.R.
#
# La función de nuestra compañera recibe clases
# predichas, no probabilidades.
#
# Por eso primero clasificamos:
#   Probabilidad >= umbral -> "Yes"
#   Probabilidad < umbral  -> "No"

resultados_f1 <- modelo_cv$pred %>%
  group_by(alpha, lambda) %>%
  group_modify(~ tibble(
    Umbral = umbrales,
    F1 = sapply(
      umbrales,
      function(u) fun_f1_score(
        .actual = .x$obs,
        .predicted = ifelse(.x$Yes >= u, "Yes", "No"),
        .positive = "Yes"
      )
    )
  )) %>%
  ungroup() %>%
  arrange(desc(F1))

# Seleccionar mejor combinación.

mejor <- resultados_f1 %>%
  slice(1)

alpha_final <- mejor$alpha
lambda_final <- mejor$lambda
umbral_final <- mejor$Umbral

# Registro con las predicciones fuera de muestra de la combinación elegida,
# para la comparación de algoritmos.
oof_mejor <- modelo_cv$pred %>%
  filter(alpha == alpha_final, lambda == lambda_final) %>%
  arrange(rowIndex)

fun_registrar_oof(
  .archivo = nombre_enet(lambda = lambda_final, alpha = alpha_final),
  .algoritmo = "Elastic Net",
  .conjunto = "X_5_extendido",
  .n_predictores = length(X),
  .hiperparametros = paste0("alpha = ", alpha_final, "; lambda = ", lambda_final),
  .obs = oof_mejor$obs,
  .prob = oof_mejor$Yes,
  .umbral = umbral_final
)

cat("\n=== TOP 10 COMBINACIONES ===\n")

print(
  head(resultados_f1, 10),
  n = 10,
  digits = 8
)

cat("\n=== MEJOR COMBINACIÓN ===\n")

print(mejor, digits = 8)


# 6. SENSIBILIDAD: CON Y SIN BOGOTÁ -------------------------

# Se conservan los hiperparámetros originales de
# nuestro experimento geográfico:
#
# alpha = 1
# lambda = 0.0001
# umbral = 0.29

alpha_bogota <- 1
lambda_bogota <- 0.0001
umbral_bogota <- 0.29

if (CORRER_PRUEBAS_BOGOTA) {
  
  idx_no_bog <- which(
    as.character(train$cat_ubicacion) != "BOGOTA"
  )
  
  # PRUEBA SOBRE CINCO FOLDS -------------------------------
  
  # Se evalúan los mismos hogares no bogotanos.
  # La única diferencia está en el entrenamiento.
  # Se usan los mismos folds de Train completo.
  
  resultados_bogota_cv <- data.frame()
  
  for (k in seq_along(cv_folds_full)) {
    
    idx_valid <- intersect(
      setdiff(seq_len(nrow(train)), cv_folds_full[[k]]),
      idx_no_bog
    )
    
    idx_train_con <- cv_folds_full[[k]]
    
    idx_train_sin <- intersect(
      idx_train_con,
      idx_no_bog
    )
    
    # Entrenar modelos.
    
    modelo_con <- ajustar_enet(
      matriz_train[idx_train_con, , drop = FALSE],
      y_train[idx_train_con],
      alpha_bogota,
      lambda_bogota
    )
    
    modelo_sin <- ajustar_enet(
      matriz_train[idx_train_sin, , drop = FALSE],
      y_train[idx_train_sin],
      alpha_bogota,
      lambda_bogota
    )
    
    # Predecir sobre el mismo fold no bogotano.
    
    prob_con <- predecir_enet(
      modelo_con,
      matriz_train[idx_valid, , drop = FALSE],
      lambda_bogota
    )
    
    prob_sin <- predecir_enet(
      modelo_sin,
      matriz_train[idx_valid, , drop = FALSE],
      lambda_bogota
    )
    
    # F1 con la función compartida.
    
    f1_con <- fun_f1_score(
      .actual = train$Pobre[idx_valid],
      .predicted = ifelse(
        prob_con >= umbral_bogota, "Yes", "No"
      )
    )
    
    f1_sin <- fun_f1_score(
      .actual = train$Pobre[idx_valid],
      .predicted = ifelse(
        prob_sin >= umbral_bogota, "Yes", "No"
      )
    )
    
    resultados_bogota_cv <- rbind(
      resultados_bogota_cv,
      data.frame(
        Fold = k,
        F1_Con_Bogota = f1_con,
        F1_Sin_Bogota = f1_sin,
        Diferencia = f1_sin - f1_con
      )
    )
    
    cat("Fold", k, "terminado\n")
  }
  
  cat("\n=== BOGOTÁ: CINCO FOLDS ===\n")
  
  print(resultados_bogota_cv, digits = 8)
  
  cat("\n=== PROMEDIOS ===\n")
  
  print(colMeans(resultados_bogota_cv[, -1]))
  
  cat(
    "\nFolds donde gana SIN Bogotá:",
    sum(resultados_bogota_cv$Diferencia > 0),
    "de 5\n"
  )
}


# 7. ENTRENAMIENTO DEFINITIVO -------------------------------

# Reentrenar el modelo seleccionado sobre todo Train.
# Incluye todos los hogares de Train.

modelo_final <- ajustar_enet(
  matriz_train,
  y_train,
  alpha_final,
  lambda_final
)

prob_test_final <- predecir_enet(
  modelo_final,
  matriz_test,
  lambda_final
)

stopifnot(
  length(prob_test_final) == nrow(test),
  all(is.finite(prob_test_final)),
  all(prob_test_final >= 0 & prob_test_final <= 1)
)

# Se guardan las predicciones fuera de muestra de la combinación elegida,
# las del test y los coeficientes del modelo final, para la comparación de
# algoritmos (13_compare_algorithms.R).
saveRDS(
  list(oof = oof_mejor, prob_test = prob_test_final, umbral = umbral_final,
       alpha = alpha_final, lambda = lambda_final,
       coeficientes = as.matrix(coef(modelo_final, s = lambda_final))),
  file.path("03_temp", paste0(nombre_enet(lambda = lambda_final,
                                          alpha = alpha_final), ".rds"))
)


# 8. SUBMISSION: ELASTIC NET SELECCIONADO -------------------

nombre_final <- nombre_enet(
  lambda = lambda_final,
  alpha = alpha_final
)

submission_final <- fun_save_submission(
  .db_test = test,
  .prob = prob_test_final,
  .umbral = umbral_final,
  .nombre = nombre_final
)


# 9. SUBMISSIONS: CON Y SIN BOGOTÁ --------------------------

# Reproducimos las configuraciones geográficas:
# alpha = 1, lambda = 0.0001, umbral = 0.29.

idx_train_no_bog <- which(
  as.character(train$cat_ubicacion) != "BOGOTA"
)

modelo_bog_con <- ajustar_enet(
  matriz_train,
  y_train,
  alpha_bogota,
  lambda_bogota
)

modelo_bog_sin <- ajustar_enet(
  matriz_train[idx_train_no_bog, , drop = FALSE],
  y_train[idx_train_no_bog],
  alpha_bogota,
  lambda_bogota
)

prob_bog_con <- predecir_enet(
  modelo_bog_con,
  matriz_test,
  lambda_bogota
)

prob_bog_sin <- predecir_enet(
  modelo_bog_sin,
  matriz_test,
  lambda_bogota
)

nombre_con <- nombre_enet(
  lambda_bogota,
  alpha_bogota,
  "con_Bogota"
)

nombre_sin <- nombre_enet(
  lambda_bogota,
  alpha_bogota,
  "sin_Bogota"
)

submission_con <- fun_save_submission(
  .db_test = test,
  .prob = prob_bog_con,
  .umbral = umbral_bogota,
  .nombre = nombre_con
)

submission_sin <- fun_save_submission(
  .db_test = test,
  .prob = prob_bog_sin,
  .umbral = umbral_bogota,
  .nombre = nombre_sin
)


# 10. RESUMEN FINAL -----------------------------------------

cat("\n====================================\n")
cat("        RESUMEN ELASTIC NET\n")
cat("====================================\n")

cat("Hogares Train:", nrow(train), "\n")
cat("Hogares Test:", nrow(test), "\n")
cat("Predictores:", length(X), "\n")
cat("Columnas:", ncol(matriz_train), "\n")

cat("\n=== MODELO SELECCIONADO ===\n")

cat("Alpha:", alpha_final, "\n")
cat("Lambda:", lambda_final, "\n")
cat("Umbral:", umbral_final, "\n")
cat("F1 OOF:", round(mejor$F1, 7), "\n")

cat("\n=== SUBMISSIONS ===\n")

cat(
  paste0(nombre_final, ".csv"),
  "- hogares pobres:",
  sum(submission_final$pobre), "\n"
)

cat(
  paste0(nombre_con, ".csv"),
  "- hogares pobres:",
  sum(submission_con$pobre), "\n"
)

cat(
  paste0(nombre_sin, ".csv"),
  "- hogares pobres:",
  sum(submission_sin$pobre), "\n"
)

cat(
  "\nPredicciones distintas con/sin Bogotá:",
  sum(submission_con$pobre != submission_sin$pobre),
  "\n"
)

################## VARIABLE IMPORTANCE

##########################################################
# VARIABLE IMPORTANCE - ELASTIC NET (5 FOLDS)
##########################################################

library(glmnet)
library(dplyr)

source("01_code/functions/fun_metricas.R")
source("01_code/functions/fun_importancia_variables.R")


# 1. CARGAR DATOS -------------------------------------------

train <- readRDS("03_temp/train_clean.rds")
predictores <- readRDS("03_temp/predictores.rds")
cv_folds <- readRDS("03_temp/cv_folds.rds")

X <- predictores$X_5_extendido

alpha_enet <- 0.5
lambda_enet <- 0.0001
umbral_enet <- 0.30

# Matriz de diseño común a los cinco folds
matriz_full <- model.matrix(
  ~ ., data = train[, X, drop = FALSE]
)[, -1, drop = FALSE]


# 2. FUNCIÓN DE PREDICCIÓN ----------------------------------

predecir_enet <- function(modelo, nuevos_datos) {
  
  matriz_nueva <- model.matrix(
    ~ ., data = nuevos_datos
  )[, -1, drop = FALSE]
  
  stopifnot(
    identical(colnames(matriz_full),
              colnames(matriz_nueva))
  )
  
  as.numeric(predict(
    modelo,
    newx = matriz_nueva,
    s = lambda_enet,
    type = "response"
  ))
}


# 3. IMPORTANCIA EN LOS CINCO FOLDS -------------------------

resultados_folds <- vector("list", length(cv_folds))

# Guardar predicciones OOF solamente en memoria
prob_oof <- rep(NA_real_, nrow(train))

for (k in seq_along(cv_folds)) {
  
  cat("\n================================\n")
  cat("Procesando fold", k, "de", length(cv_folds), "\n")
  cat("================================\n")
  
  # Separar entrenamiento y validación
  idx_train <- cv_folds[[k]]
  
  idx_valid <- setdiff(
    seq_len(nrow(train)),
    idx_train
  )
  
  # Entrenar modelo
  modelo_fold <- glmnet(
    x = matriz_full[idx_train, , drop = FALSE],
    y = as.integer(train$Pobre[idx_train] == "Yes"),
    family = "binomial",
    alpha = alpha_enet,
    lambda = lambda_enet,
    standardize = TRUE
  )
  
  # Datos de validación
  datos_valid <- train[idx_valid, X, drop = FALSE]
  real_valid <- train$Pobre[idx_valid]
  
  # Probabilidades OOF
  prob_oof[idx_valid] <- predecir_enet(
    modelo_fold,
    datos_valid
  )
  
  # Importancia por permutación
  importancia_fold <- fun_importancia_variables(
    .modelo = modelo_fold,
    .datos = datos_valid,
    .real = real_valid,
    .predecir = predecir_enet,
    .umbral = umbral_enet,
    .n_perm = 3,
    .semilla = 2026 + k
  )
  
  importancia_fold$Fold <- k
  
  resultados_folds[[k]] <- importancia_fold
  
  cat(
    "F1 original:",
    round(importancia_fold$F1_original[1], 6),
    "\n"
  )
}


# 4. UNIR RESULTADOS ----------------------------------------

importancia_todos_folds <- bind_rows(resultados_folds)


# 5. PROMEDIAR IMPORTANCIA ENTRE FOLDS ---------------------

importancia_enet_final <- importancia_todos_folds %>%
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


# 6. CALCULAR F1 OOF GLOBAL ---------------------------------

stopifnot(!anyNA(prob_oof))

f1_oof <- fun_f1_score(
  .actual = train$Pobre,
  .predicted = ifelse(
    prob_oof >= umbral_enet, "Yes", "No"
  )
)

cat("\n================================\n")
cat("RESULTADOS FINALES\n")
cat("================================\n")

cat("F1 OOF global:", round(f1_oof, 7), "\n")
cat("Variables evaluadas:", nrow(importancia_enet_final), "\n")

cat("\nTOP 15 VARIABLES MÁS IMPORTANTES:\n")

print(
  head(importancia_enet_final, 15),
  n = 15,
  digits = 7
)


# 7. TABLA COMPLETA -----------------------------------------

print(
  importancia_enet_final,
  n = Inf,
  digits = 7
)

#################################




conservar <- c(
  "fun_colapse_personas",
  "fun_preprocess_hogares",
  "fun_preprocess_personas",
  "fun_save_submission",
  "fun_f1_score",
  "fun_multi_stats",
  "fun_crear_ctrl",
  "fun_umbral_f1",
  "fun_metricas_umbral",
  "fun_registrar_modelo",
  "fun_importancia_variables"
)

# Limpiar entorno
funciones <- ls(envir = .GlobalEnv)
funciones <- funciones[vapply(
  mget(funciones, envir = .GlobalEnv),
  is.function,
  logical(1)
)]

rm(list = setdiff(funciones, conservar), envir = .GlobalEnv)

rm(list = ls()[!sapply(mget(ls()), is.function)])

cat("\nProceso terminado.\n")
