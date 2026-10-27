##########################################################
# 12_model_naive_bayes.R
#
# Objetivo: Entrenar Naive Bayes y generar el envío a Kaggle.
#
# Se usa el paquete naivebayes a través de caret, con los mismos folds
# del resto de modelos. Los factores se pasan como factores: la fórmula
# de caret los convertiría en dummies y Bayes dejaría de ver cada
# categoría como una sola variable.
#
# La grilla compara la densidad normal con un estimado de kernel, y dos
# suavizados de Laplace. Laplace en 0 deja probabilidades cero cuando
# una categoría no aparece entre los pobres de un fold, y eso anula el
# posterior. Por eso el mínimo de la grilla es 1.
#
# Etapa del Problem Set: Etapa 2: Training and Comparing Algorithms.
# Insumos:  03_temp/ (bases, predictores y folds).
# Salidas:  02_outputs/predictions/ y una fila en registro_modelos.csv.
##########################################################

# Preparación -------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, caret, pROC, naivebayes, MLmetrics)

source("01_code/functions/fun_metricas.R")
source("01_code/functions/fun_save_submission.R")

if (!file.exists("03_temp/train_clean.rds")) {
  stop("Faltan las bases de 03_temp/. Corra antes los scripts 01 a 05.",
       call. = FALSE)
}

train       <- readRDS("03_temp/train_clean.rds")
test        <- readRDS("03_temp/test_clean.rds")
predictores <- readRDS("03_temp/predictores.rds")
cv_folds    <- readRDS("03_temp/cv_folds.rds")

if (max(unlist(cv_folds)) != nrow(train) ||
    length(unique(unlist(cv_folds))) != nrow(train)) {
  stop("Las particiones no corresponden a train. Vuelva a correr ",
       "05_create_cv_folds.R.", call. = FALSE)
}

X <- predictores[["X_5_extendido"]]
x_train <- as.data.frame(train[, X])
x_test <- as.data.frame(test[, X])
y_train <- train$Pobre

# adjust solo aplica cuando usekernel es TRUE. Se deja fijo en 1 para
# no duplicar la grilla de la normal.
grid_nb <- expand.grid(
  laplace = c(1, 5),
  usekernel = c(FALSE, TRUE),
  adjust = 1
)

ctrl_nb <- trainControl(
  method = "cv",
  number = length(cv_folds),
  index = cv_folds,
  summaryFunction = fun_multi_stats,
  classProbs = TRUE,
  savePredictions = "all"
)

ruta_cv <- "03_temp/naivebayes_X_5_extendido_cv.rds"
REENTRENAR_CV <- FALSE
modelo_cv <- NULL
if (!REENTRENAR_CV && file.exists(ruta_cv)) {
  guardado <- readRDS(ruta_cv)
  if (is.list(guardado) && identical(guardado$grid, grid_nb)) {
    modelo_cv <- guardado$modelo
    message("Validación cruzada de Naive Bayes recuperada.")
  }
}

if (is.null(modelo_cv)) {
  set.seed(2026)
  modelo_cv <- train(
    x = x_train,
    y = y_train,
    method = "naive_bayes",
    trControl = ctrl_nb,
    tuneGrid = grid_nb,
    metric = "F",
    maximize = TRUE
  )
  dir.create("03_temp", showWarnings = FALSE)
  saveRDS(list(modelo = modelo_cv, grid = grid_nb), ruta_cv)
}

resumen <- modelo_cv$pred |>
  group_by(laplace, usekernel) |>
  group_modify(~ {
    res <- fun_umbral_f1(.x$obs, .x$Yes)
    met <- fun_metricas_umbral(.x$obs, .x$Yes, res$umbral)
    tibble(umbral = res$umbral,
           f1 = met$F1,
           precision = met$precision,
           recall = met$recall,
           auc = res$auc)
  }) |>
  ungroup() |>
  arrange(desc(f1), desc(auc))
print(as.data.frame(resumen), digits = 4)

mejor <- resumen |> slice(1)
oof <- modelo_cv$pred |>
  filter(laplace == mejor$laplace, usekernel == mejor$usekernel)
res_umbral <- fun_umbral_f1(oof$obs, oof$Yes)
comparacion <- bind_rows(
  fun_metricas_umbral(oof$obs, oof$Yes, 0.5) |>
    mutate(estrategia = "Umbral 0,5"),
  fun_metricas_umbral(oof$obs, oof$Yes, res_umbral$umbral) |>
    mutate(estrategia = "Umbral que maximiza F1")
) |>
  relocate(estrategia)
print(as.data.frame(comparacion), digits = 4)

kernel_txt <- ifelse(mejor$usekernel, "kernel", "normal")
archivo <- paste0(
  "nb_X_5_extendido_", kernel_txt,
  "_laplace_", fun_hiperparam_str(mejor$laplace, 0),
  "_umbral_", fun_hiperparam_str(res_umbral$umbral, 2)
)

set.seed(2026)
modelo_final <- train(
  x = x_train,
  y = y_train,
  method = "naive_bayes",
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid = expand.grid(
    laplace = mejor$laplace,
    usekernel = mejor$usekernel,
    adjust = 1
  )
)
prob_test <- predict(modelo_final, newdata = x_test, type = "prob")[["Yes"]]
fun_save_submission(test, prob_test, res_umbral$umbral, archivo)

metricas_cv <- comparacion |> filter(estrategia == "Umbral que maximiza F1")
fun_registrar_modelo(tibble(
  archivo = archivo,
  algoritmo = "Naive Bayes",
  conjunto = "X_5_extendido",
  n_predictores = length(X),
  hiperparametros = paste0(
    "usekernel = ", mejor$usekernel,
    "; laplace = ", mejor$laplace,
    "; adjust = 1"
  ),
  umbral = res_umbral$umbral,
  f1_cv = metricas_cv$F1,
  precision_cv = metricas_cv$precision,
  recall_cv = metricas_cv$recall,
  auc_cv = res_umbral$auc,
  fecha = as.character(Sys.Date())
))

saveRDS(list(oof = oof, prob_test = prob_test, umbral = res_umbral$umbral,
             comparacion = comparacion, resumen = resumen),
        file.path("03_temp", paste0(archivo, ".rds")))

message("F1 fuera de muestra: ", round(metricas_cv$F1, 4),
        " (Random Forest de referencia: 0.7148).")
message("Archivo para Kaggle: 02_outputs/predictions/", archivo, ".csv")
