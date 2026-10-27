##########################################################
# fun_save_submission.R
#
# Funciones para convertir las probabilidades predichas por un modelo en un
# archivo listo para subir a Kaggle. Reúnen los pasos que se repiten en
# todos los envíos: aplicar el umbral, armar id/pobre, nombrar el archivo,
# verificar su formato y guardarlo.
##########################################################

fun_hiperparam_str <- function(.x, .digits = 4) {
  #' Convertir un hiperparámetro numérico en texto para nombres de archivo
  #'
  #' @description \code{fun_hiperparam_str()} redondea un número y reemplaza
  #' el punto decimal por un guion bajo, para que los nombres de archivo no
  #' tengan puntos. Por ejemplo, 0.0123 se convierte en "0_0123".
  #'
  #' @param .x \code{numeric}. Valor del hiperparámetro.
  #' @param .digits \code{integer}. Número de decimales al redondear.
  #'
  #' @returns \code{character}. El valor como texto, sin puntos.
  gsub(pattern = ".", replacement = "_",
       x = as.character(round(.x, .digits)), fixed = TRUE)
}

fun_save_submission <- function(.db_test, .prob, .umbral, .nombre,
                                .dir = "02_outputs/predictions",
                                .sample = "00_data/raw/sample_submission.csv") {
  #' Guardar una predicción en el formato de envío de Kaggle
  #'
  #' @description \code{fun_save_submission()} clasifica como pobre (1) a los
  #' hogares cuya probabilidad predicha es mayor o igual al umbral, arma la
  #' base con las columnas \code{id} y \code{pobre}, verifica que cumpla el
  #' formato de la competencia y la guarda como .csv. Recibe probabilidades
  #' (y no el modelo) para servir con cualquier algoritmo, incluido el
  #' Modelo Lineal de Probabilidad, cuyas predicciones no salen de
  #' \code{predict(type = "prob")}.
  #'
  #' @param .db_test \code{data.frame}. Base de prueba a nivel hogar; debe
  #' incluir la columna \code{id}.
  #' @param .prob \code{numeric}. Probabilidad predicha de ser pobre para cada
  #' fila de \code{.db_test}, en el mismo orden.
  #' @param .umbral \code{numeric}. Punto de corte entre 0 y 1. Un hogar se
  #' clasifica como pobre si \code{.prob >= .umbral}.
  #' @param .nombre \code{character}. Nombre del archivo, sin la extensión
  #' .csv. Debe indicar el algoritmo y los hiperparámetros, por ejemplo
  #' "EN_lambda_0_001_alpha_0_5".
  #' @param .dir \code{character}. Carpeta de destino.
  #' @param .sample \code{character}. Ruta a sample_submission.csv, usada para
  #' verificar que estén todos los hogares del test.
  #'
  #' @returns \code{data.frame} (invisible) con las columnas \code{id} y
  #' \code{pobre}. Como efecto secundario, guarda el archivo en \code{.dir}.

  # Validación de los insumos ---------------------------------------------
  # Detenerse aquí es mejor que gastar un envío diario de Kaggle en un
  # archivo con errores.
  if (!"id" %in% names(.db_test)) {
    stop("La base de prueba no tiene la columna 'id'.", call. = FALSE)
  }
  if (length(.prob) != nrow(.db_test)) {
    stop("Hay ", length(.prob), " probabilidades para ", nrow(.db_test),
         " hogares. Deben coincidir.", call. = FALSE)
  }
  if (anyNA(.prob)) {
    stop("Hay ", sum(is.na(.prob)), " probabilidades faltantes (NA). ",
         "Revise los valores faltantes en las variables del test.",
         call. = FALSE)
  }
  if (!is.numeric(.umbral) || length(.umbral) != 1 ||
      .umbral < 0 || .umbral > 1) {
    stop("El umbral debe ser un único número entre 0 y 1.", call. = FALSE)
  }
  if (!grepl("^[A-Za-z0-9_]+$", .nombre)) {
    stop("El nombre '", .nombre, "' solo puede tener letras, números y ",
         "guiones bajos (sin puntos ni espacios, y sin la extensión .csv).",
         call. = FALSE)
  }

  # De probabilidades a clase ----------------------------------------------
  db_submission <- data.frame(
    id = .db_test$id,
    pobre = as.integer(.prob >= .umbral)
  )

  # Verificación contra el formato de la competencia -----------------------
  db_sample <- utils::read.csv(.sample)
  if (anyDuplicated(db_submission$id) > 0) {
    stop("Hay hogares repetidos en la predicción.", call. = FALSE)
  }
  if (!setequal(db_submission$id, db_sample$id)) {
    stop("Los id de la predicción no coinciden con los de ",
         "sample_submission.csv.", call. = FALSE)
  }

  # Guardar ---------------------------------------------------------------
  dir.create(.dir, recursive = TRUE, showWarnings = FALSE)
  ruta <- file.path(.dir, paste0(.nombre, ".csv"))
  if (file.exists(ruta)) {
    warning("Se sobrescribió el archivo existente ", ruta, call. = FALSE)
  }
  utils::write.csv(db_submission, ruta, row.names = FALSE)

  message("Guardado: ", ruta, "\n",
          "  Hogares clasificados como pobres: ", sum(db_submission$pobre),
          " de ", nrow(db_submission), " (",
          round(100 * mean(db_submission$pobre), 1), "%)")

  invisible(db_submission)
}
