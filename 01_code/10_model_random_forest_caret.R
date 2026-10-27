##########################################################
# 10_model_random_forest_caret.R
#
# Objetivo: Entrenar un Random Forest (primer envío a Kaggle de este
# algoritmo) y reportar la importancia de variables por permutación.
#
# Sigue el cuaderno de la clase (Lecture 10 / Notebook_Tree_Methods):
# caret + ranger, grilla de mtry y min.node.size, splitrule fijo, y
# varImp por permutación en el modelo final. La validación usa los
# mismos folds que el resto del repositorio, y el umbral se elige
# maximizando el F1 fuera de muestra, como en el logit y el CART.
#
# Etapa del Problem Set: Etapa 2: Training and Comparing Algorithms.
# Insumos:  03_temp/ (bases de modelación, conjuntos de predictores y
#           particiones de validación cruzada).
# Salidas:  02_outputs/predictions/ (el .csv para Kaggle), una fila en
#           02_outputs/tables/registro_modelos.csv, la tabla y la figura
#           de importancia, y el modelo en 03_temp/.
##########################################################
#
# Este script se puede correr solo (no requiere el script maestro), siempre
# que antes se hayan corrido los scripts 01 a 05. Así se puede iterar con
# Kaggle sin repetir la preparación de los datos.
#
# Estructura de la variante:
#   1. Entrenar con train(method = "ranger") y los folds de cv_folds.rds.
#   2. Elegir mtry, min.node.size y el umbral que maximizan el F1 fuera
#      de muestra (no el F1 del umbral 0,5 que usa caret por dentro).
#   3. Reentrenar ese bosque con todo train y calcular la importancia
#      por permutación una sola vez, en el modelo final.
#   4. Predecir el test y guardar el envío con fun_save_submission().
#
# Envío a Kaggle, desde la raíz del repositorio (el nombre del archivo
# lo imprime el script al terminar):
#   kaggle competitions submit -c uniandes-bdml-2026-20-ps-2 ^
#     -f 02_outputs/predictions/NOMBRE.csv -m "Random Forest"
##########################################################

# Preparación -------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, caret, pROC, ranger, MLmetrics)

source("01_code/functions/fun_metricas.R")
source("01_code/functions/fun_save_submission.R")

if (!file.exists("03_temp/train_clean.rds") ||
    !file.exists("03_temp/test_clean.rds") ||
    !file.exists("03_temp/predictores.rds") ||
    !file.exists("03_temp/cv_folds.rds")) {
  stop(
    "\n\nFaltan las bases de 03_temp/. Este script no las construye.\n",
    "Desde la raíz del repositorio:\n",
    "  1. Deja los CSV de la competencia en 00_data/raw/, o descárgalos con\n",
    "     kaggle competitions download -c uniandes-bdml-2026-20-ps-2 -p 00_data/raw\n",
    "  2. Corre source(\"01_code/01_get_data.R\") hasta\n",
    "     source(\"01_code/05_create_cv_folds.R\").\n",
    "  3. Vuelve a correr este script.\n",
    call. = FALSE
  )
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

# 300 árboles. ranger usa 500 por defecto. Con 165 mil hogares y una grilla
# evaluada en 5 folds, 300 mantiene estable el promedio del bosque y el
# script termina en un tiempo razonable. Si el puntaje queda corto, se sube.
n_trees <- 300L

# Si la validación cruzada ya se guardó con esta misma grilla y este mismo
# número de árboles, se reutiliza. Ponga TRUE para volver a entrenarla.
REENTRENAR_CV <- FALSE

# Función de la variante --------------------------------------------------
fun_rf_variante <- function(.nombre_conjunto) {
  #' Entrenar un Random Forest y generar su envío a Kaggle
  #'
  #' @description \code{fun_rf_variante()} entrena un bosque con ranger,
  #' elige \code{mtry} y \code{min.node.size} por el F1 fuera de muestra,
  #' elige el umbral que maximiza ese F1, reentrena el bosque ganador con
  #' importancia por permutación y guarda el envío a Kaggle.
  #'
  #' @param .nombre_conjunto \code{character}. Nombre de un conjunto de
  #' \code{03_temp/predictores.rds}, por ejemplo "X_5_extendido".
  #'
  #' @returns \code{list} (invisible) con el modelo final, el umbral, la
  #' comparación de umbrales, la importancia y el nombre del archivo.
  
  X <- predictores[[.nombre_conjunto]]
  if (is.null(X)) {
    stop("No existe el conjunto ", .nombre_conjunto, " en predictores.rds.",
         call. = FALSE)
  }
  formula_rf <- formula(paste("Pobre ~", paste(X, collapse = " + ")))
  
  # caret, igual que en el cuaderno, expande los factores a dummies antes
  # de pasárselos a ranger. mtry se cuenta en ese espacio, no en las 43
  # columnas originales. La categoría base de cada factor no entra y por
  # eso no aparece en la importancia.
  p_exp <- ncol(model.matrix(formula_rf, data = train)) - 1L
  mtry_vals <- unique(as.integer(pmax(
    2, round(c(sqrt(p_exp), p_exp / 5, p_exp / 3))
  )))
  mtry_vals <- mtry_vals[mtry_vals <= p_exp]
  
  # En el cuaderno, con 3 mil viviendas, min.node.size es 1, 3 y 5. Aquí
  # hay 165 mil hogares: hojas de 5 y de 20 observaciones cubren un árbol
  # más profundo y uno más regularizado, sin el costo de hojas de tamaño 1.
  grid_rf <- expand.grid(
    mtry = mtry_vals,
    splitrule = "gini",
    min.node.size = c(5, 20)
  )
  
  message("Predictores originales: ", length(X),
          ". Columnas tras las dummies: ", p_exp, ".")
  message("Grilla: ", nrow(grid_rf), " combinaciones x 5 folds x ",
          n_trees, " árboles.")
  
  # No usamos fun_crear_ctrl(): esa función guarda solo las predicciones
  # del mejor modelo según el F1 al umbral 0,5. Aquí hace falta la
  # predicción de cada combinación, para elegir mtry, tamaño de hoja y
  # umbral con el mismo F1.
  ctrl_cv <- trainControl(
    method = "cv",
    number = length(cv_folds),
    index = cv_folds,
    summaryFunction = fun_multi_stats,
    classProbs = TRUE,
    savePredictions = "all"
  )
  
  ruta_cv <- file.path("03_temp", paste0("rf_", .nombre_conjunto, "_cv.rds"))
  modelo_cv <- NULL
  if (!REENTRENAR_CV && file.exists(ruta_cv)) {
    guardado <- readRDS(ruta_cv)
    if (is.list(guardado) && identical(guardado$n_trees, n_trees) &&
        identical(guardado$grid, grid_rf)) {
      modelo_cv <- guardado$modelo
      message("Validación cruzada recuperada de ", ruta_cv, ".")
    }
  }
  if (is.null(modelo_cv)) {
    set.seed(2026)
    modelo_cv <- train(
      formula_rf,
      data = train,
      method = "ranger",
      trControl = ctrl_cv,
      tuneGrid = grid_rf,
      metric = "F",
      maximize = TRUE,
      num.trees = n_trees,
      seed = 2026
    )
    dir.create("03_temp", showWarnings = FALSE)
    saveRDS(list(modelo = modelo_cv, n_trees = n_trees, grid = grid_rf),
            ruta_cv)
    message("Validación cruzada guardada en ", ruta_cv, ".")
  }
  
  # 2. Para cada bosque, umbral que maximiza el F1 en los hogares que no
  # vio. Se queda la combinación con mayor F1. Si empatan, la de mayor AUC.
  resumen_tunes <- modelo_cv$pred |>
    group_by(mtry, min.node.size) |>
    group_modify(~ {
      res <- fun_umbral_f1(.x$obs, .x$Yes)
      tibble(umbral = res$umbral,
             f1 = res$mejor$F1,
             precision = res$mejor$precision,
             recall = res$mejor$recall,
             auc = res$auc)
    }) |>
    ungroup() |>
    arrange(desc(f1), desc(auc))
  print(as.data.frame(resumen_tunes), digits = 4)
  
  mejor <- resumen_tunes |> slice(1)
  oof <- modelo_cv$pred |>
    filter(mtry == mejor$mtry, min.node.size == mejor$min.node.size)
  res_umbral <- fun_umbral_f1(oof$obs, oof$Yes)
  
  comparacion <- bind_rows(
    fun_metricas_umbral(oof$obs, oof$Yes, 0.5) |>
      mutate(estrategia = "Umbral 0,5"),
    fun_metricas_umbral(oof$obs, oof$Yes, res_umbral$umbral) |>
      mutate(estrategia = "Umbral que maximiza F1")
  ) |>
    relocate(estrategia)
  print(as.data.frame(comparacion), digits = 4)
  
  archivo <- paste0(
    "rf_", .nombre_conjunto,
    "_mtry_", fun_hiperparam_str(mejor$mtry, 0),
    "_node_", fun_hiperparam_str(mejor$min.node.size, 0),
    "_trees_", n_trees,
    "_umbral_", fun_hiperparam_str(res_umbral$umbral, 2)
  )
  
  # 3. Modelo final. El cuaderno calcula la importancia UNA vez, en el
  # bosque reentrenado con todos los datos, con las predicciones OOB de
  # ranger. No es el promedio de los cinco folds. Por eso la grilla de
  # arriba no pide importancia (sería 30 veces el mismo costo) y este
  # ajuste sí: importance = "permutation".
  # Con probability = TRUE, ranger mide la importancia como la caída del
  # Brier score al permutar la variable. Un número más alto quiere decir
  # que esa variable mueve más la predicción de pobreza.
  message("Reentrenando el bosque ganador y calculando importancia ",
          "por permutación...")
  set.seed(2026)
  modelo_final <- train(
    formula_rf,
    data = train,
    method = "ranger",
    trControl = trainControl(method = "none", classProbs = TRUE),
    tuneGrid = expand.grid(mtry = mejor$mtry,
                           splitrule = "gini",
                           min.node.size = mejor$min.node.size),
    num.trees = n_trees,
    importance = "permutation",
    seed = 2026
  )
  
  importancia <- tibble(
    Variable = names(modelo_final$finalModel$variable.importance),
    Importancia = as.numeric(modelo_final$finalModel$variable.importance)
  ) |>
    mutate(Importancia_0_100 = 100 * Importancia / max(Importancia)) |>
    arrange(desc(Importancia))
  
  message("Variables más importantes (caída del Brier score, OOB):")
  print(as.data.frame(head(importancia, 15)), digits = 4)
  
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
    scale_y_continuous(limits = c(0, 100),
                       expand = expansion(mult = c(0, 0.02))) +
    labs(
      x = NULL,
      y = "Importancia relativa (0 a 100)",
      title = "Random forest: importancia por permutación",
      subtitle = paste0("mtry = ", mejor$mtry,
                        ", min.node.size = ", mejor$min.node.size,
                        ", árboles = ", n_trees,
                        ". La categoría base de cada factor no aparece.")
    ) +
    theme_minimal(base_size = 11) +
    theme(plot.margin = margin(8, 16, 8, 8))
  ggsave(ruta_fig, plot = figura, width = 9, height = 6.5, bg = "white",
         device = grDevices::cairo_pdf)
  message("Importancia guardada en ", ruta_imp, " y ", ruta_fig, ".")
  
  # 4. Predicción en el test, con el bosque reentrenado con todo train.
  prob_test <- predict(modelo_final, newdata = test, type = "prob")[["Yes"]]
  fun_save_submission(test, prob_test, res_umbral$umbral, archivo)
  
  metricas_cv <- comparacion |> filter(estrategia == "Umbral que maximiza F1")
  fun_registrar_modelo(tibble(
    archivo = archivo,
    algoritmo = "Random Forest",
    conjunto = .nombre_conjunto,
    n_predictores = length(X),
    hiperparametros = paste0(
      "mtry = ", mejor$mtry,
      "; min.node.size = ", mejor$min.node.size,
      "; splitrule = gini; num.trees = ", n_trees,
      "; columnas dummy = ", p_exp
    ),
    umbral = res_umbral$umbral,
    f1_cv = metricas_cv$F1,
    precision_cv = metricas_cv$precision,
    recall_cv = metricas_cv$recall,
    auc_cv = res_umbral$auc,
    fecha = as.character(Sys.Date())
  ))
  
  saveRDS(list(oof = oof, prob_test = prob_test, umbral = res_umbral$umbral,
               comparacion = comparacion, resumen_tunes = resumen_tunes,
               mtry = mejor$mtry, min.node.size = mejor$min.node.size,
               n_trees = n_trees, importancia = importancia),
          file.path("03_temp", paste0(archivo, ".rds")))
  
  message(
    "\nPara subirlo a Kaggle, desde la raíz del repositorio:\n",
    "kaggle competitions submit -c uniandes-bdml-2026-20-ps-2 ",
    "-f 02_outputs/predictions/", archivo, ".csv ",
    "-m \"Random Forest ", .nombre_conjunto,
    "; mtry ", mejor$mtry,
    "; min.node.size ", mejor$min.node.size,
    "; umbral ", round(res_umbral$umbral, 2), "\"\n"
  )
  
  invisible(list(modelo = modelo_final, umbral = res_umbral$umbral,
                 comparacion = comparacion, importancia = importancia,
                 archivo = archivo))
}

# Variante ----------------------------------------------------------------
# El mismo conjunto de 43 predictores del mejor logit y del CART, para que
# la diferencia de puntaje sea del algoritmo. Sin rebalanceo: el umbral
# se encarga del 20% de hogares pobres, igual que en el logit.
rf_X5 <- fun_rf_variante("X_5_extendido")
