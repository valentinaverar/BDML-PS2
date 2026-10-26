##########################################################
# 10_model_random_forest.R
#
# Objetivo: Entrenar variantes de Random Forest y generar sus
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

# Variante 1 --------------------------------------------------------------
# Código pendiente.


library(ranger)
library(caret)
library(dplyr)

source("01_code/functions/fun_metricas.R")
source("01_code/functions/fun_save_submission.R")


# 1. CARGAR DATOS ------------------------------------------

train <- readRDS("03_temp/train_clean.rds")
test <- readRDS("03_temp/test_clean.rds")

predictores <- readRDS("03_temp/predictores.rds")
cv_folds <- readRDS("03_temp/cv_folds.rds")

# Conjunto de predictores
X <- predictores$X_5_extendido

stopifnot(
  is.character(X),
  length(X) > 0,
  !anyNA(X),
  !anyDuplicated(X),
  all(X %in% names(train)),
  all(X %in% names(test)),
  identical(levels(train$Pobre), c("Yes", "No")),
  length(cv_folds) == 5,
  "id" %in% names(test)
)

cat("\nHogares Train:", nrow(train), "\n")
cat("Hogares Test:", nrow(test), "\n")
cat("Predictores:", length(X), "\n")


# 2. VERIFICAR LOS CINCO FOLDS ------------------------------

# Evitar repetir el problema que encontramos
# con los índices antiguos de Development.

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

stopifnot(
  length(indices_validacion) == nrow(train),
  !anyDuplicated(indices_validacion)
)

cat("Los cinco folds corresponden a Train completo.\n")


# 3. HIPERPARÁMETROS ---------------------------------------

num_arboles <- 300
mtry_rf <- max(1L, floor(sqrt(length(X))))
min_nodo <- 10
num_hilos <- 8

cat("\n=== CONFIGURACIÓN RANDOM FOREST ===\n")
cat("Árboles:", num_arboles, "\n")
cat("mtry:", mtry_rf, "\n")
cat("Tamaño mínimo de nodo:", min_nodo, "\n")


# 4. VALIDACIÓN CRUZADA ------------------------------------

prob_oof <- rep(NA_real_, nrow(train))
tiempos <- numeric(length(cv_folds))

set.seed(2026)

inicio_total <- Sys.time()

for (k in seq_along(cv_folds)) {
  
  cat("\n================================\n")
  cat("RANDOM FOREST - FOLD", k, "DE 5\n")
  cat("================================\n")
  
  inicio_fold <- Sys.time()
  
  # Separar entrenamiento y validación
  idx_train <- cv_folds[[k]]
  
  idx_valid <- setdiff(
    seq_len(nrow(train)),
    idx_train
  )
  
  # Entrenar Random Forest probabilístico
  modelo <- ranger(
    dependent.variable.name = "Pobre",
    data = train[idx_train, c("Pobre", X), drop = FALSE],
    num.trees = num_arboles,
    mtry = mtry_rf,
    min.node.size = min_nodo,
    probability = TRUE,
    num.threads = num_hilos,
    importance = "none",
    oob.error = FALSE,
    seed = 2026 + k,
    verbose = FALSE
  )
  
  # Probabilidades fuera de muestra
  prob_oof[idx_valid] <- as.numeric(
    predict(
      modelo,
      data = train[idx_valid, X, drop = FALSE],
      num.threads = num_hilos
    )$predictions[, "Yes"]
  )
  
  # Tiempo del fold
  tiempos[k] <- as.numeric(
    difftime(
      Sys.time(),
      inicio_fold,
      units = "secs"
    )
  )
  
  cat("Tiempo:", round(tiempos[k], 1), "segundos\n")
  
  rm(modelo)
  gc(verbose = FALSE)
}


# 5. SELECCIONAR UMBRAL POR F1 ------------------------------

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

cat("\n=== RESULTADOS DE VALIDACIÓN ===\n")
cat("Mejor umbral:", umbral_final, "\n")
cat("F1 OOF:", round(f1_oof, 7), "\n")

print(head(resultados_f1, 10), digits = 7)


# 6. ENTRENAMIENTO DEFINITIVO ------------------------------

# Ahora utilizamos todos los hogares etiquetados.

cat("\nEntrenando Random Forest final...\n")

modelo_final <- ranger(
  dependent.variable.name = "Pobre",
  data = train[, c("Pobre", X), drop = FALSE],
  num.trees = num_arboles,
  mtry = mtry_rf,
  min.node.size = min_nodo,
  probability = TRUE,
  num.threads = num_hilos,
  importance = "none",
  oob.error = FALSE,
  seed = 2026,
  verbose = FALSE
)


# 7. PREDICCIONES EN TEST ----------------------------------

prob_test <- as.numeric(
  predict(
    modelo_final,
    data = test[, X, drop = FALSE],
    num.threads = num_hilos
  )$predictions[, "Yes"]
)

stopifnot(
  length(prob_test) == nrow(test),
  all(is.finite(prob_test)),
  all(prob_test >= 0 & prob_test <= 1)
)


# 8. GENERAR SUBMISSION ------------------------------------

# Formato de nombres con hiperparámetros.

nombre_submission <- paste0(
  "RF_trees_", num_arboles,
  "_mtry_", mtry_rf,
  "_node_", min_nodo
)

submission_rf <- fun_save_submission(
  .db_test = test,
  .prob = prob_test,
  .umbral = umbral_final,
  .nombre = nombre_submission
)

# Registro con las predicciones fuera de muestra, para la comparación de
# algoritmos.
fun_registrar_oof(
  .archivo = nombre_submission,
  .algoritmo = "Random Forest",
  .conjunto = "X_5_extendido",
  .n_predictores = length(X),
  .hiperparametros = paste0("árboles = ", num_arboles, "; mtry = ", mtry_rf,
                            "; nodo mínimo = ", min_nodo),
  .obs = train$Pobre,
  .prob = prob_oof,
  .umbral = umbral_final
)


# 9. RESULTADOS FINALES ------------------------------------

tiempo_total <- as.numeric(
  difftime(
    Sys.time(),
    inicio_total,
    units = "mins"
  )
)

cat("\n========================================\n")
cat("         RESUMEN RANDOM FOREST\n")
cat("========================================\n")

cat("\nDATOS:\n")
cat("Hogares Train:", nrow(train), "\n")
cat("Hogares Test:", nrow(test), "\n")
cat("Predictores:", length(X), "\n")

cat("\nCONFIGURACIÓN:\n")
cat("Árboles:", num_arboles, "\n")
cat("mtry:", mtry_rf, "\n")
cat("Tamaño de nodo:", min_nodo, "\n")

cat("\nVALIDACIÓN CRUZADA:\n")
cat("Tiempo por fold:", round(tiempos, 1), "\n")
cat("F1 OOF:", round(f1_oof, 7), "\n")
cat("Umbral:", umbral_final, "\n")

cat("\nSUBMISSION:\n")
cat("Hogares pobres predichos:",
    sum(submission_rf$pobre), "\n")
cat("Proporción predicha pobre:",
    round(mean(submission_rf$pobre), 4), "\n")

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
