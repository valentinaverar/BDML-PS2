##########################################################
# fun_metricas.R
#
# Funciones de evaluación que comparten todos los modelos:
# el F1, el resumen de métricas para la validación cruzada, la configuración
# de la validación cruzada, la elección del umbral, las métricas con un umbral
# dado y el registro de cada variante. Requiere que los paquetes tidyverse,
# caret y pROC estén cargados.
##########################################################

fun_f1_score <- function(.actual, .predicted, .positive = "Yes") {
  #' Calcular el F1 de una clasificación
  #'
  #' @description \code{fun_f1_score()} calcula F1 = 2TP / (2TP + FP + FN).
  #' Escrito así, da 0 (y no NA) cuando el modelo no acierta ningún positivo,
  #' que es lo que corresponde: un modelo que nunca identifica hogares pobres
  #' tiene el peor F1 posible.
  #'
  #' @param .actual Clase observada.
  #' @param .predicted Clase predicha.
  #' @param .positive \code{character}. Clase positiva (hogar pobre).
  #'
  #' @returns \code{numeric}. El F1, entre 0 y 1.
  tp <- sum(.actual == .positive & .predicted == .positive)
  fp <- sum(.actual != .positive & .predicted == .positive)
  fn <- sum(.actual == .positive & .predicted != .positive)
  2 * tp / (2 * tp + fp + fn)
}

fun_multi_stats <- function(...) {
  #' Resumen de métricas para la validación cruzada de caret
  #'
  #' @description \code{fun_multi_stats()} se usa como \code{summaryFunction}
  #' en \code{trainControl()}. Reúne las métricas de \code{defaultSummary()}
  #' (Accuracy, Kappa) y de \code{prSummary()} (AUC de la curva PR,
  #' precisión, recall y F), y reemplaza F por \code{fun_f1_score()}: así un
  #' fold sin aciertos positivos tiene F1 = 0 en lugar de NA. Estas métricas
  #' usan el umbral de 0,5 que caret aplica para asignar la clase.
  #'
  #' @param ... Argumentos que caret pasa a la función de resumen.
  #'
  #' @returns \code{numeric}. Vector con nombre de las métricas.
  summary_values <- c(
    caret::defaultSummary(...),
    caret::prSummary(...)
  )
  predictions <- list(...)[[1]]
  summary_values["F"] <- fun_f1_score(predictions$obs, predictions$pred)
  summary_values
}

fun_crear_ctrl <- function(.cv_folds, .sampling = NULL) {
  #' Configurar la validación cruzada común a todos los modelos
  #'
  #' @description \code{fun_crear_ctrl()} arma el \code{trainControl()} con
  #' las particiones fijas de \code{03_temp/cv_folds.rds}, para que todos los
  #' modelos se validen con los mismos hogares. Guarda las probabilidades
  #' (\code{classProbs}) y las predicciones fuera de muestra de la mejor
  #' combinación de hiperparámetros (\code{savePredictions = "final"}), que
  #' son las que se usan para elegir el umbral.
  #'
  #' @param .cv_folds \code{list}. Filas de entrenamiento de cada fold.
  #' @param .sampling \code{character} o \code{NULL}. Rebalanceo de clases
  #' dentro de cada fold (por ejemplo, "up" o "down"); NULL para no
  #' rebalancear.
  #'
  #' @returns Objeto de \code{trainControl()}.
  trainControl(method = "cv",
               number = length(.cv_folds),
               index = .cv_folds,
               summaryFunction = fun_multi_stats,
               classProbs = TRUE,
               savePredictions = "final",
               sampling = .sampling)
}

fun_umbral_f1 <- function(.obs, .prob, .positive = "Yes") {
  #' Elegir el umbral que maximiza el F1
  #'
  #' @description \code{fun_umbral_f1()} recorre 100 umbrales entre 0 y 1
  #' sobre la curva de precisión y recall, calcula el F1 de cada uno y
  #' elige el mayor. Debe recibir predicciones fuera de muestra (las de la
  #' validación cruzada), nunca las de los mismos hogares con que se
  #' entrenó el modelo.
  #'
  #' @param .obs Clase observada (factor con niveles "Yes" y "No").
  #' @param .prob \code{numeric}. Probabilidad predicha de ser pobre.
  #' @param .positive \code{character}. Clase positiva.
  #'
  #' @returns \code{list} con el umbral elegido, \code{umbral}; la fila de
  #' la curva con ese umbral, \code{mejor}; la curva completa,
  #' \code{curva}; y el área bajo la curva ROC, \code{auc}.
  roc_obj <- pROC::roc(response = .obs,
                       predictor = .prob,
                       levels = c(setdiff(levels(.obs), .positive), .positive),
                       direction = "<",
                       quiet = TRUE)
  
  curva <- data.frame(pROC::coords(roc = roc_obj,
                                   x = seq(0, 1, length = 100),
                                   ret = c("threshold", "precision",
                                           "recall"))) |>
    mutate(F1 = (2 * precision * recall) / (precision + recall))
  
  # Si dos umbrales empatan, se conserva el primero.
  mejor <- curva |> slice_max(F1, n = 1, with_ties = FALSE)
  
  list(umbral = mejor$threshold,
       mejor = mejor,
       curva = curva,
       auc = as.numeric(pROC::auc(roc_obj)))
}

fun_metricas_umbral <- function(.obs, .prob, .umbral, .positive = "Yes") {
  #' Métricas de clasificación con un umbral dado
  #'
  #' @description \code{fun_metricas_umbral()} clasifica como pobre a quien
  #' tiene probabilidad mayor o igual al umbral y calcula precisión, recall
  #' y F1, además del número de falsos positivos (hogares no pobres que
  #' recibirían el beneficio: leakage) y de falsos negativos (hogares pobres
  #' que quedarían por fuera: undercoverage).
  #'
  #' @param .obs Clase observada (factor con niveles "Yes" y "No").
  #' @param .prob \code{numeric}. Probabilidad predicha de ser pobre.
  #' @param .umbral \code{numeric}. Punto de corte.
  #' @param .positive \code{character}. Clase positiva.
  #'
  #' @returns \code{tibble} de una fila con el umbral y las métricas.
  pred <- factor(ifelse(.prob >= .umbral, .positive,
                        setdiff(levels(.obs), .positive)),
                 levels = levels(.obs))
  cm <- confusionMatrix(data = pred, reference = .obs,
                        positive = .positive, mode = "prec_recall")
  
  tibble(umbral = .umbral,
         precision = unname(cm$byClass["Precision"]),
         recall = unname(cm$byClass["Recall"]),
         F1 = fun_f1_score(.obs, pred, .positive),
         falsos_positivos = sum(pred == .positive & .obs != .positive),
         falsos_negativos = sum(pred != .positive & .obs == .positive))
}

fun_registrar_modelo <- function(.fila,
                                 .ruta = "02_outputs/tables/registro_modelos.csv") {
  #' Registrar una variante en la tabla de modelos
  #'
  #' @description \code{fun_registrar_modelo()} agrega la fila de una
  #' variante a la tabla de registro. Si ya existe una fila con el mismo
  #' archivo (porque el script se volvió a correr), la reemplaza en lugar de
  #' duplicarla. El puntaje público de Kaggle se anota aparte, en
  #' \code{00_data/kaggle_scores.csv}, y las dos tablas se unen por
  #' \code{archivo}.
  #'
  #' @param .fila \code{data.frame} de una fila. Debe tener la columna
  #' \code{archivo}.
  #' @param .ruta \code{character}. Ruta de la tabla de registro.
  #'
  #' @returns La tabla de registro actualizada (invisible). Como efecto
  #' secundario, la guarda en \code{.ruta}.
  if (file.exists(.ruta)) {
    registro <- read.csv(.ruta) |>
      filter(archivo != .fila$archivo) |>
      bind_rows(.fila)
  } else {
    dir.create(dirname(.ruta), recursive = TRUE, showWarnings = FALSE)
    registro <- .fila
  }
  write.csv(registro, .ruta, row.names = FALSE)
  invisible(registro)
}

fun_registrar_oof <- function(.archivo, .algoritmo, .conjunto, .n_predictores,
                              .hiperparametros, .obs, .prob, .umbral) {
  #' Registrar una variante a partir de sus predicciones fuera de muestra
  #'
  #' @description \code{fun_registrar_oof()} calcula las métricas de
  #' validación cruzada de una variante (F1, precisión y recall con el umbral
  #' elegido, y el AUC) y la agrega a la tabla de registro con
  #' \code{fun_registrar_modelo()}. Sirve para los modelos que no se
  #' entrenan con caret (LPM, Random Forest, XGBoost), de modo que todos
  #' queden en la misma tabla y se puedan comparar.
  #'
  #' @param .archivo \code{character}. Nombre del envío (sin .csv).
  #' @param .algoritmo \code{character}. Nombre del algoritmo.
  #' @param .conjunto \code{character}. Conjunto de predictores.
  #' @param .n_predictores \code{integer}. Número de predictores.
  #' @param .hiperparametros \code{character}. Descripción de los
  #' hiperparámetros.
  #' @param .obs Clase observada (factor con niveles "Yes" y "No").
  #' @param .prob \code{numeric}. Probabilidad (o puntaje) fuera de muestra.
  #' @param .umbral \code{numeric}. Umbral elegido.
  #'
  #' @returns La tabla de registro actualizada (invisible).
  metricas <- fun_metricas_umbral(.obs, .prob, .umbral)
  roc_obj <- pROC::roc(response = .obs, predictor = .prob,
                       levels = c("No", "Yes"), direction = "<",
                       quiet = TRUE)
  fun_registrar_modelo(tibble(
    archivo = .archivo,
    algoritmo = .algoritmo,
    conjunto = .conjunto,
    n_predictores = .n_predictores,
    hiperparametros = .hiperparametros,
    umbral = .umbral,
    f1_cv = metricas$F1,
    precision_cv = metricas$precision,
    recall_cv = metricas$recall,
    auc_cv = as.numeric(pROC::auc(roc_obj)),
    fecha = as.character(Sys.Date())
  ))
}
