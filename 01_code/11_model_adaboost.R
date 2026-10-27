##########################################################
# 11_model_adaboost.R
#
# Objetivo: Entrenar AdaBoost.M1 (adabag) y generar el envío a Kaggle.
#
# El cuaderno de la clase arma boosting como suma de árboles poco
# profundos. AdaBoost.M1 es la versión para clasificación: cada árbol
# nuevo se concentra en los hogares que el anterior clasificó mal.
# coeflearn = "Breiman" es el peso clásico del algoritmo,
# (1/2) * log((1 - error) / error). boos = FALSE usa toda la muestra
# con esos pesos; un bootstrap sería bagging, no AdaBoost.
#
# La grilla no usa tocones (profundidad 1). Con 43 predictores un
# tocón no combina variables, y el Random Forest de referencia ya
# llega a un F1 de 0,715. Se prueban profundidades 3 y 5.
#
# Etapa del Problem Set: Etapa 2: Training and Comparing Algorithms.
# Insumos:  03_temp/ (bases, predictores y folds).
# Salidas:  02_outputs/predictions/, registro_modelos.csv, importancia
#           y el modelo en 03_temp/.
##########################################################

# Preparación -------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, caret, pROC, adabag, rpart, MLmetrics)

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

# 100 árboles. En el ejemplo de clase son 500 tocones sobre una sola x;
# aquí cada árbol ve 43 variables, así que 100 de profundidad 3 o 5
# alcanzan a corregir errores sin repetir el costo del bosque.
mfinal <- 100L
profundidades <- c(3L, 5L)
coeflearn <- "Breiman"
REENTRENAR_CV <- FALSE

fun_prob_pobre_adaboost <- function(modelo, datos) {
  #' Probabilidad de ser pobre según el orden de clases de AdaBoost
  clases <- names(attr(modelo, "vardep.summary"))
  prob <- predict(modelo, newdata = datos)$prob
  prob[, which(clases == "Yes")]
}

fun_adaboost <- function(datos, .mfinal, .maxdepth) {
  X <- predictores[["X_5_extendido"]]
  formula_ada <- formula(paste("Pobre ~", paste(X, collapse = " + ")))
  boosting(
    formula_ada,
    data = as.data.frame(datos),
    boos = FALSE,
    mfinal = .mfinal,
    coeflearn = coeflearn,
    control = rpart.control(
      maxdepth = .maxdepth,
      cp = -1,
      minsplit = 40,
      minbucket = 20,
      xval = 0
    )
  )
}

# Validación cruzada ------------------------------------------------------
X <- predictores[["X_5_extendido"]]
ruta_cv <- "03_temp/adaboost_X_5_extendido_cv.rds"
oof <- NULL
if (!REENTRENAR_CV && file.exists(ruta_cv)) {
  guardado <- readRDS(ruta_cv)
  if (is.list(guardado) && identical(guardado$mfinal, mfinal) &&
      identical(guardado$profundidades, profundidades)) {
    oof <- guardado$oof
    message("Validación cruzada de AdaBoost recuperada de ", ruta_cv, ".")
  }
}

if (is.null(oof)) {
  oof <- matrix(NA_real_, nrow = nrow(train), ncol = length(profundidades))
  colnames(oof) <- paste0("depth_", profundidades)
  for (j in seq_along(profundidades)) {
    for (k in seq_along(cv_folds)) {
      idx_train <- cv_folds[[k]]
      idx_valid <- setdiff(seq_len(nrow(train)), idx_train)
      message("AdaBoost profundidad ", profundidades[j],
              ", fold ", k, " de ", length(cv_folds), "...")
      set.seed(2026 + j * 10 + k)
      modelo_k <- fun_adaboost(train[idx_train, ], mfinal, profundidades[j])
      oof[idx_valid, j] <- fun_prob_pobre_adaboost(
        modelo_k, as.data.frame(train[idx_valid, ])
      )
      rm(modelo_k)
      gc(verbose = FALSE)
    }
    if (anyNA(oof[, j])) {
      stop("Quedaron probabilidades faltantes en la profundidad ",
           profundidades[j], ".", call. = FALSE)
    }
    # Con 20% de hogares pobres, la probabilidad media no puede quedar
    # cerca de 0,8: eso indicaría que la columna de "Yes" está al revés.
    if (mean(oof[, j]) > 0.5) {
      stop("La probabilidad media de pobre es ", round(mean(oof[, j]), 3),
           ". Revise el orden de las clases de AdaBoost.", call. = FALSE)
    }
    dir.create("03_temp", showWarnings = FALSE)
    saveRDS(list(oof = oof, mfinal = mfinal, profundidades = profundidades),
            ruta_cv)
  }
}

resumen <- lapply(seq_along(profundidades), function(j) {
  res <- fun_umbral_f1(train$Pobre, oof[, j])
  met <- fun_metricas_umbral(train$Pobre, oof[, j], res$umbral)
  tibble(maxdepth = profundidades[j],
         mfinal = mfinal,
         umbral = res$umbral,
         f1 = met$F1,
         precision = met$precision,
         recall = met$recall,
         auc = res$auc)
}) |>
  bind_rows() |>
  arrange(desc(f1), desc(auc))
print(as.data.frame(resumen), digits = 4)

mejor <- resumen |> slice(1)
j_mejor <- which(profundidades == mejor$maxdepth)
res_umbral <- fun_umbral_f1(train$Pobre, oof[, j_mejor])
comparacion <- bind_rows(
  fun_metricas_umbral(train$Pobre, oof[, j_mejor], 0.5) |>
    mutate(estrategia = "Umbral 0,5"),
  fun_metricas_umbral(train$Pobre, oof[, j_mejor], res_umbral$umbral) |>
    mutate(estrategia = "Umbral que maximiza F1")
) |>
  relocate(estrategia)
print(as.data.frame(comparacion), digits = 4)

archivo <- paste0(
  "adaboost_X_5_extendido_mfinal_", mfinal,
  "_depth_", mejor$maxdepth,
  "_umbral_", fun_hiperparam_str(res_umbral$umbral, 2)
)

# Modelo final e importancia ----------------------------------------------
message("Reentrenando AdaBoost con todo train (profundidad ",
        mejor$maxdepth, ")...")
set.seed(2026)
modelo_final <- fun_adaboost(train, mfinal, mejor$maxdepth)
prob_test <- fun_prob_pobre_adaboost(modelo_final, as.data.frame(test))
fun_save_submission(test, prob_test, res_umbral$umbral, archivo)

importancia <- tibble(
  Variable = names(modelo_final$importance),
  Importancia = as.numeric(modelo_final$importance)
) |>
  mutate(Importancia_0_100 = 100 * Importancia / max(Importancia)) |>
  arrange(desc(Importancia))

dir.create("02_outputs/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("02_outputs/figures", recursive = TRUE, showWarnings = FALSE)
ruta_imp <- file.path("02_outputs/tables", paste0("importancia_", archivo, ".csv"))
ruta_fig <- file.path("02_outputs/figures", paste0("importancia_", archivo, ".pdf"))
write.csv(importancia, ruta_imp, row.names = FALSE)

figura <- importancia |>
  slice_head(n = 15) |>
  ggplot(aes(x = reorder(Variable, Importancia_0_100),
             y = Importancia_0_100)) +
  geom_col(fill = "steelblue", width = 0.7) +
  coord_flip(clip = "off") +
  scale_y_continuous(limits = c(0, 100), expand = expansion(mult = c(0, 0.02))) +
  labs(
    x = NULL,
    y = "Importancia relativa (0 a 100)",
    title = "AdaBoost: importancia de variables",
    subtitle = paste0("Profundidad ", mejor$maxdepth, ", ", mfinal, " árboles")
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.margin = margin(8, 16, 8, 8))
ggsave(ruta_fig, plot = figura, width = 9, height = 6.5, bg = "white",
       device = grDevices::cairo_pdf)

metricas_cv <- comparacion |> filter(estrategia == "Umbral que maximiza F1")
fun_registrar_modelo(tibble(
  archivo = archivo,
  algoritmo = "AdaBoost",
  conjunto = "X_5_extendido",
  n_predictores = length(X),
  hiperparametros = paste0(
    "mfinal = ", mfinal,
    "; maxdepth = ", mejor$maxdepth,
    "; coeflearn = ", coeflearn,
    "; boos = FALSE"
  ),
  umbral = res_umbral$umbral,
  f1_cv = metricas_cv$F1,
  precision_cv = metricas_cv$precision,
  recall_cv = metricas_cv$recall,
  auc_cv = res_umbral$auc,
  fecha = as.character(Sys.Date())
))

saveRDS(list(oof = oof[, j_mejor], prob_test = prob_test,
             umbral = res_umbral$umbral, comparacion = comparacion,
             resumen = resumen, importancia = importancia,
             mfinal = mfinal, maxdepth = mejor$maxdepth),
        file.path("03_temp", paste0(archivo, ".rds")))

message("F1 fuera de muestra: ", round(metricas_cv$F1, 4),
        " (Random Forest de referencia: 0.7148).")
message("Archivo para Kaggle: 02_outputs/predictions/", archivo, ".csv")
