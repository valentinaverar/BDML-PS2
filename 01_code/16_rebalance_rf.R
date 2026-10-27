##########################################################
# 16_rebalance_rf.R
#
# Random Forest con cada hogar pobre repetido 4 veces en el entrenamiento.
# Los no pobres entran una vez. La copia se hace solo dentro de cada fold
# de entrenamiento: la validación y el test conservan el 20% de pobres real.
#
# Mismos hiperparámetros del Random Forest sin rebalanceo:
#   mtry 31, min.node.size 20, 300 árboles, Gini.
# mtry 31 está en el espacio de dummies (93 columnas), no en las 43 variables.
##########################################################

if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, caret, pROC, ranger, MLmetrics)

source("01_code/functions/fun_metricas.R")
source("01_code/functions/fun_save_submission.R")

train <- readRDS("03_temp/train_clean.rds")
test <- readRDS("03_temp/test_clean.rds")
predictores <- readRDS("03_temp/predictores.rds")
cv_folds <- readRDS("03_temp/cv_folds.rds")
X <- predictores[["X_5_extendido"]]
y <- train[["Pobre"]]

if (max(unlist(cv_folds)) != nrow(train)) {
  stop("Las particiones no corresponden a train.", call. = FALSE)
}

fun_idx_x4 <- function(idx) {
  pobres <- idx[y[idx] == "Yes"]
  c(idx, pobres, pobres, pobres)
}

fun_cerrar <- function(nombre, obs, prob) {
  res <- fun_umbral_f1(obs, prob)
  met <- fun_metricas_umbral(obs, prob, res[["umbral"]])
  message(nombre, ": F1 ", round(met[["F1"]], 4),
          ", umbral ", round(res[["umbral"]], 4),
          ", precision ", round(met[["precision"]], 4),
          ", recall ", round(met[["recall"]], 4))
  list(umbral = res[["umbral"]], metricas = met, auc = res[["auc"]],
       comparacion = bind_rows(
         fun_metricas_umbral(obs, prob, 0.5) |>
           mutate(estrategia = "Umbral 0,5"),
         met |> mutate(estrategia = "Umbral que maximiza F1")
       ) |> relocate(estrategia))
}

n_pobres_fold <- sapply(cv_folds, function(idx) sum(y[idx] == "Yes"))
message("Pobres en cada fold de entrenamiento, antes de copiar: ",
        paste(n_pobres_fold, collapse = ", "),
        ". Despues de x4: ",
        paste(n_pobres_fold * 4, collapse = ", "), ".")

formula_x <- formula(paste("~", paste(X, collapse = " + ")))
mm_train <- model.matrix(formula_x, data = train)
mm_train <- mm_train[, colnames(mm_train) != "(Intercept)", drop = FALSE]
mm_test <- model.matrix(formula_x, data = test)
mm_test <- mm_test[, colnames(mm_test) != "(Intercept)", drop = FALSE]
faltan <- setdiff(colnames(mm_train), colnames(mm_test))
if (length(faltan) > 0) {
  mm_test <- cbind(mm_test, matrix(0, nrow(mm_test), length(faltan),
                                   dimnames = list(NULL, faltan)))
}
mm_test <- mm_test[, colnames(mm_train), drop = FALSE]
if (!identical(colnames(mm_train), colnames(mm_test)) ||
    nrow(mm_train) != nrow(train)) {
  stop("La matriz de dummies de train y test no coincide.", call. = FALSE)
}
message("Columnas dummy: ", ncol(mm_train), ".")

message("Random Forest con pobres x4...")
oof_rf <- rep(NA_real_, nrow(train))
for (k in seq_along(cv_folds)) {
  idx_train <- cv_folds[[k]]
  idx_valid <- setdiff(seq_len(nrow(train)), idx_train)
  idx_bal <- fun_idx_x4(idx_train)
  datos_k <- as.data.frame(mm_train[idx_bal, , drop = FALSE])
  datos_k[[".outcome"]] <- y[idx_bal]
  set.seed(2026 + k)
  ajuste <- ranger(
    dependent.variable.name = ".outcome",
    data = datos_k,
    num.trees = 300,
    mtry = 31,
    min.node.size = 20,
    splitrule = "gini",
    probability = TRUE,
    seed = 2026 + k
  )
  valid_k <- as.data.frame(mm_train[idx_valid, , drop = FALSE])
  oof_rf[idx_valid] <- predict(ajuste, data = valid_k)[["predictions"]][, "Yes"]
  message("  fold ", k, " listo")
  rm(ajuste, datos_k, valid_k)
  gc(verbose = FALSE)
}
cierre_rf <- fun_cerrar("Random Forest x4", y, oof_rf)

idx_final <- fun_idx_x4(seq_len(nrow(train)))
datos_final <- as.data.frame(mm_train[idx_final, , drop = FALSE])
datos_final[[".outcome"]] <- y[idx_final]
set.seed(2026)
ajuste_rf <- ranger(
  dependent.variable.name = ".outcome",
  data = datos_final,
  num.trees = 300,
  mtry = 31,
  min.node.size = 20,
  splitrule = "gini",
  probability = TRUE,
  seed = 2026
)
prob_rf <- predict(
  ajuste_rf,
  data = as.data.frame(mm_test)
)[["predictions"]][, "Yes"]
archivo_rf <- paste0(
  "rf_X_5_extendido_up4_mtry_31_node_20_trees_300_umbral_",
  fun_hiperparam_str(cierre_rf[["umbral"]], 2)
)
fun_save_submission(test, prob_rf, cierre_rf[["umbral"]], archivo_rf)
met_rf <- cierre_rf[["comparacion"]] |>
  filter(estrategia == "Umbral que maximiza F1")
fun_registrar_modelo(tibble(
  archivo = archivo_rf,
  algoritmo = "Random Forest",
  conjunto = "X_5_extendido",
  n_predictores = length(X),
  hiperparametros = "mtry = 31; min.node.size = 20; num.trees = 300; pobres x4 en train",
  umbral = cierre_rf[["umbral"]],
  f1_cv = met_rf[["F1"]],
  precision_cv = met_rf[["precision"]],
  recall_cv = met_rf[["recall"]],
  auc_cv = cierre_rf[["auc"]],
  fecha = as.character(Sys.Date())
))

message("Listo.")
message("RF ", archivo_rf)
