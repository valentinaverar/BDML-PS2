##########################################################
# 16_rebalance_adaboost.R
#
# AdaBoost.M1 con cada hogar pobre repetido 4 veces en el entrenamiento.
# Los no pobres entran una vez. La copia se hace solo dentro de cada fold
# de entrenamiento: la validación y el test conservan el 20% de pobres real.
#
# Mismos hiperparámetros del AdaBoost sin rebalanceo:
#   100 árboles, profundidad 5, coeflearn Breiman, sin bootstrap.
##########################################################

if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, caret, pROC, adabag, rpart)

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
message("ADA ", archivo_ada)
