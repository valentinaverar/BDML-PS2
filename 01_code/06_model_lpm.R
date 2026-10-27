##########################################################
# 06_model_lpm.R
# LPM con 10 predictores y con el conjunto X_5_extendido (43):
#   1. CV de 5 folds y selección del umbral por F1
#   2. Importancia de variables por permutación (5 folds)
#   3. Submissions para Kaggle
#
# Objetivo: Entrenar variantes de Modelo Lineal de Probabilidad (LPM) y generar sus
# predicciones para Kaggle.
#
# Etapa del Problem Set: Etapa 2: Training and Comparing Algorithms.
# Insumos:  03_temp/ (bases de modelación y particiones de validación cruzada).
# Salidas:  02_outputs/predictions/ (un .csv por variante), una fila por variante en
#           02_outputs/tables/registro_modelos.csv y el modelo en 03_temp/.
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
# ÚNICOS ARCHIVOS GENERADOS:
#   02_outputs/predictions/LPM_10.csv
#   02_outputs/predictions/LPM_X_5_extendido.csv
# Además, una fila por variante en 02_outputs/tables/registro_modelos.csv.
##########################################################

# 0. PAQUETES Y FUNCIONES COMPARTIDAS -----------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, caret, dplyr)

source("01_code/functions/fun_metricas.R")
source("01_code/functions/fun_importancia_variables.R")
source("01_code/functions/fun_save_submission.R")

# 1. DATOS Y ESPECIFICACIONES -------------------------------
train <- readRDS("03_temp/train_clean.rds")
test <- readRDS("03_temp/test_clean.rds")
cv_folds <- readRDS("03_temp/cv_folds.rds")
predictores <- readRDS("03_temp/predictores.rds")

X_10 <- c(
  "cat_ubicacion", "cat_tenencia", "num_personas_cuarto",
  "num_minors", "num_old", "cat_maxEduc", "prop_ocupados",
  "cat_pos_jefe", "bin_head_male", "bin_hogar_pensionado"
)
# X_5_extendido es el conjunto común con que se comparan todos los
# algoritmos.
X_5 <- predictores$X_5_extendido
variantes <- list(LPM_10 = X_10, LPM_X_5_extendido = X_5)
conjuntos <- c(LPM_10 = "LPM_10", LPM_X_5_extendido = "X_5_extendido")

umbrales <- seq(0.10, 0.60, by = 0.01)
n_perm <- 3L

stopifnot(
  length(cv_folds) == 5L,
  identical(levels(train$Pobre), c("Yes", "No")),
  all(vapply(variantes, function(v) {
    all(v %in% names(train)) && all(v %in% names(test))
  }, logical(1)))
)

y <- as.integer(train$Pobre == "Yes")

# 2. UNA FUNCIÓN DE PREDICCIÓN, SOLO EN ESTE SCRIPT ---------
# Se usa para los cinco folds, Test y permutaciones.
# Conserva el manejo previo de colinealidad (coeficientes NA -> 0).
predecir_lpm <- function(modelo, nuevos_datos) {
  matriz <- model.matrix(~ ., data = nuevos_datos[, modelo$variables, drop = FALSE])
  if (!identical(colnames(matriz), modelo$columnas)) {
    stop("Las columnas de predicción no coinciden con las del ajuste.")
  }
  scores <- as.numeric(matriz %*% modelo$beta)
  stopifnot(all(is.finite(scores)))
  scores
}

# 3. ENTRENAR Y EVALUAR AMBAS VARIANTES ----------------------
resultados_cv_lpm <- list()
resultados_importancia_por_fold <- list()
resumen_importancia_lpm <- list()
submissions_lpm <- list()

for (nombre in names(variantes)) {
  
  variables <- variantes[[nombre]]
  cat("\n========================================\n")
  cat("MODELO:", nombre, "\n")
  cat("========================================\n")
  
  # Construir las matrices una sola vez por variante.
  matriz_train <- model.matrix(~ ., data = train[, variables, drop = FALSE])
  matriz_test <- model.matrix(~ ., data = test[, variables, drop = FALSE])
  
  stopifnot(
    identical(colnames(matriz_train), colnames(matriz_test)),
    !anyNA(matriz_train), !anyNA(matriz_test)
  )
  
  # 3A. VALIDACIÓN CRUZADA: PREDICCIONES FUERA DE FOLD ------
  scores_oof <- rep(NA_real_, nrow(train))
  modelos_fold <- vector("list", length(cv_folds))
  rangos <- integer(length(cv_folds))
  
  for (k in seq_along(cv_folds)) {
    idx_train <- cv_folds[[k]]
    idx_valid <- setdiff(seq_len(nrow(train)), idx_train)
    
    ajuste <- lm.fit(
      x = matriz_train[idx_train, , drop = FALSE],
      y = y[idx_train]
    )
    beta <- ajuste$coefficients
    beta[is.na(beta)] <- 0
    
    modelos_fold[[k]] <- list(
      beta = beta,
      variables = variables,
      columnas = colnames(matriz_train)
    )
    rangos[k] <- ajuste$rank
    scores_oof[idx_valid] <- as.numeric(
      matriz_train[idx_valid, , drop = FALSE] %*% beta
    )
    cat("CV: fold", k, "terminado\n")
  }
  stopifnot(all(is.finite(scores_oof)))
  
  # 3B. ELEGIR UMBRAL QUE MAXIMIZA F1 OOF -------------------
  tabla_umbrales <- data.frame(
    Umbral = umbrales,
    F1 = vapply(umbrales, function(u) {
      fun_f1_score(
        .actual = train$Pobre,
        .predicted = ifelse(scores_oof >= u, "Yes", "No")
      )
    }, numeric(1))
  ) %>% arrange(desc(F1))
  
  mejor <- tabla_umbrales[1, , drop = FALSE]
  umbral <- mejor$Umbral
  resultados_cv_lpm[[nombre]] <- tabla_umbrales
  
  # Registro con las predicciones fuera de muestra, para la comparación de
  # algoritmos.
  fun_registrar_oof(.archivo = nombre, .algoritmo = "LPM",
                    .conjunto = conjuntos[[nombre]],
                    .n_predictores = length(variables),
                    .hiperparametros = "ninguno", .obs = train$Pobre,
                    .prob = scores_oof, .umbral = umbral)
  
  cat("\nTOP 10 UMBRALES:\n")
  print(head(tabla_umbrales, 10), digits = 7)
  cat("F1 OOF:", round(mejor$F1, 7),
      "| Umbral:", umbral,
      "| Rangos:", paste(rangos, collapse = ", "),
      "| Scores fuera de [0,1]:",
      sum(scores_oof < 0 | scores_oof > 1), "\n")
  
  # 3C. IMPORTANCIA POR PERMUTACIÓN EN LOS CINCO FOLDS ------
  por_fold <- vector("list", length(cv_folds))
  
  for (k in seq_along(cv_folds)) {
    idx_valid <- setdiff(seq_len(nrow(train)), cv_folds[[k]])
    
    importancia_fold <- fun_importancia_variables(
      .modelo = modelos_fold[[k]],
      .datos = train[idx_valid, variables, drop = FALSE],
      .real = train$Pobre[idx_valid],
      .predecir = predecir_lpm,
      .umbral = umbral,
      .n_perm = n_perm,
      .semilla = 2026 + k
    )
    importancia_fold$Fold <- k
    por_fold[[k]] <- importancia_fold
    
    cat("Importancia: fold", k, "terminado\n")
  }
  
  resultados_importancia_por_fold[[nombre]] <- bind_rows(por_fold)
  
  # Misma tabla del Elastic Net: 6 columnas, 15 permutaciones.
  tabla_importancia <- resultados_importancia_por_fold[[nombre]] %>%
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
  
  resumen_importancia_lpm[[nombre]] <- tabla_importancia
  cat("\nTOP 15 VARIABLES MÁS IMPORTANTES:\n")
  print(as.data.frame(head(tabla_importancia, 15)), digits = 7)
  
  # 3D. ENTRENAR CON TODO TRAIN Y GENERAR SUBMISSION --------
  ajuste_final <- lm.fit(x = matriz_train, y = y)
  beta_final <- ajuste_final$coefficients
  beta_final[is.na(beta_final)] <- 0
  scores_test <- as.numeric(matriz_test %*% beta_final)
  stopifnot(all(is.finite(scores_test)))
  
  # Solo para el formato de submission; no altera decisiones
  # porque el umbral elegido está estrictamente entre 0 y 1.
  prob_test <- pmin(pmax(scores_test, 0), 1)
  
  submissions_lpm[[nombre]] <- fun_save_submission(
    .db_test = test,
    .prob = prob_test,
    .umbral = umbral,
    .nombre = nombre
  )
  cat("Submission:", paste0(nombre, ".csv"),
      "| Pobres predichos:", sum(submissions_lpm[[nombre]]$pobre), "\n")
}

# 4. TABLAS FINALES EN MEMORIA -------------------------------
importancia_lpm_10 <- resumen_importancia_lpm$LPM_10
importancia_lpm_x5 <- resumen_importancia_lpm$LPM_X_5_extendido

cat("Clasificaciones distintas en Test:",
    sum(submissions_lpm$LPM_10$pobre != submissions_lpm$LPM_X_5_extendido$pobre), "\n")
cat("\nTablas en memoria: importancia_lpm_10 e importancia_lpm_x5.\n")

# Limpiar entorno
funciones <- ls(envir = .GlobalEnv)
funciones <- funciones[vapply(
  mget(funciones, envir = .GlobalEnv),
  is.function,
  logical(1)
)]

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

rm(list = setdiff(funciones, conservar), envir = .GlobalEnv)

rm(list = ls()[!sapply(mget(ls()), is.function)])

cat("\nProceso terminado.\n")
