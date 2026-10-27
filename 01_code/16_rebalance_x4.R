##########################################################
# 16_rebalance_x4.R
#
# Cada hogar pobre entra 4 veces en el entrenamiento de cada fold.
# Los no pobres entran una vez. Con 20% de pobres, eso deja la muestra
# de entrenamiento cerca de mitad y mitad.
#
# La copia se hace solo dentro del fold de entrenamiento. La validación
# y el test conservan la proporción real: si no, el F1 mediría hogares
# repetidos y no sería comparable con los modelos anteriores.
#
# Se mantienen los hiperparámetros que ganaron sin rebalanceo. Lo único
# que cambia es el peso de los pobres y, después, el umbral.
#
#   Random Forest: mtry 31, min.node.size 20, 300 árboles, Gini
#   AdaBoost:      100 árboles, profundidad 5, coeflearn Breiman
#   Naive Bayes:   normal, Laplace 5
##########################################################

if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, caret, pROC, ranger, adabag, rpart,
               naivebayes, MLmetrics)

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

# Indices de entrenamiento con cada pobre repetido 4 veces.
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

# Matriz de dummies, la misma que uso el Random Forest original.
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

# Naive Bayes ------------------------------------------------------------
message("Naive Bayes con pobres x4...")
oof_nb <- rep(NA_real_, nrow(train))
for (k in seq_along(cv_folds)) {
  idx_train <- cv_folds[[k]]
  idx_valid <- setdiff(seq_len(nrow(train)), idx_train)
  idx_bal <- fun_idx_x4(idx_train)
  ajuste <- naive_bayes(
    as.data.frame(train[idx_bal, X, drop = FALSE]),
    y[idx_bal],
    usekernel = FALSE,
    laplace = 5
  )
  oof_nb[idx_valid] <- predict(
    ajuste,
    newdata = as.data.frame(train[idx_valid, X, drop = FALSE]),
    type = "prob"
  )[, "Yes"]
  message("  fold ", k, " listo")
}
cierre_nb <- fun_cerrar("Naive Bayes x4", y, oof_nb)

ajuste_nb <- naive_bayes(
  as.data.frame(train[fun_idx_x4(seq_len(nrow(train))), X, drop = FALSE]),
  y[fun_idx_x4(seq_len(nrow(train)))],
  usekernel = FALSE,
  laplace = 5
)
prob_nb <- predict(
  ajuste_nb,
  newdata = as.data.frame(test[, X, drop = FALSE]),
  type = "prob"
)[, "Yes"]
archivo_nb <- paste0(
  "nb_X_5_extendido_up4_normal_laplace_5_umbral_",
  fun_hiperparam_str(cierre_nb[["umbral"]], 2)
)
fun_save_submission(test, prob_nb, cierre_nb[["umbral"]], archivo_nb)
met_nb <- cierre_nb[["comparacion"]] |>
  filter(estrategia == "Umbral que maximiza F1")
fun_registrar_modelo(tibble(
  archivo = archivo_nb,
  algoritmo = "Naive Bayes",
  conjunto = "X_5_extendido",
  n_predictores = length(X),
  hiperparametros = "usekernel = FALSE; laplace = 5; pobres x4 en train",
  umbral = cierre_nb[["umbral"]],
  f1_cv = met_nb[["F1"]],
  precision_cv = met_nb[["precision"]],
  recall_cv = met_nb[["recall"]],
  auc_cv = cierre_nb[["auc"]],
  fecha = as.character(Sys.Date())
))

# Random Forest -----------------------------------------------------------
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
rm(ajuste_rf, datos_final)
gc(verbose = FALSE)

# AdaBoost ----------------------------------------------------------------
message("AdaBoost con pobres x4...")
formula_ada <- formula(paste("Pobre ~", paste(X, collapse = " + ")))
oof_ada <- rep(NA_real_, nrow(train))
for (k in seq_along(cv_folds)) {
  idx_train <- cv_folds[[k]]
  idx_valid <- setdiff(seq_len(nrow(train)), idx_train)
  set.seed(2026 + k)
  ajuste <- boosting(
    formula_ada,
    data = as.data.frame(train[fun_idx_x4(idx_train), ]),
    boos = FALSE,
    mfinal = 100,
    coeflearn = "Breiman",
    control = rpart.control(
      maxdepth = 5, cp = -1, minsplit = 40, minbucket = 20, xval = 0
    )
  )
  clases <- names(attr(ajuste, "vardep.summary"))
  oof_ada[idx_valid] <- predict(
    ajuste, newdata = as.data.frame(train[idx_valid, ])
  )[["prob"]][, which(clases == "Yes")]
  message("  fold ", k, " listo")
  rm(ajuste)
  gc(verbose = FALSE)
}
cierre_ada <- fun_cerrar("AdaBoost x4", y, oof_ada)

set.seed(2026)
ajuste_ada <- boosting(
  formula_ada,
  data = as.data.frame(train[fun_idx_x4(seq_len(nrow(train))), ]),
  boos = FALSE,
  mfinal = 100,
  coeflearn = "Breiman",
  control = rpart.control(
    maxdepth = 5, cp = -1, minsplit = 40, minbucket = 20, xval = 0
  )
)
clases <- names(attr(ajuste_ada, "vardep.summary"))
prob_ada <- predict(
  ajuste_ada, newdata = as.data.frame(test)
)[["prob"]][, which(clases == "Yes")]
archivo_ada <- paste0(
  "adaboost_X_5_extendido_up4_mfinal_100_depth_5_umbral_",
  fun_hiperparam_str(cierre_ada[["umbral"]], 2)
)
fun_save_submission(test, prob_ada, cierre_ada[["umbral"]], archivo_ada)
met_ada <- cierre_ada[["comparacion"]] |>
  filter(estrategia == "Umbral que maximiza F1")
fun_registrar_modelo(tibble(
  archivo = archivo_ada,
  algoritmo = "AdaBoost",
  conjunto = "X_5_extendido",
  n_predictores = length(X),
  hiperparametros = "mfinal = 100; maxdepth = 5; coeflearn = Breiman; pobres x4 en train",
  umbral = cierre_ada[["umbral"]],
  f1_cv = met_ada[["F1"]],
  precision_cv = met_ada[["precision"]],
  recall_cv = met_ada[["recall"]],
  auc_cv = cierre_ada[["auc"]],
  fecha = as.character(Sys.Date())
))

message("Listo.")
message("NB  ", archivo_nb)
message("RF  ", archivo_rf)
message("ADA ", archivo_ada)
