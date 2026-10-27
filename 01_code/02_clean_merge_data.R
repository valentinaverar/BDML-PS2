##########################################################
# 02_clean_merge_data.R
#
# Objetivo: Limpiar las bases de personas y hogares, construir variables a nivel
# hogar a partir de las personas y unir ambas bases.
#
# Etapa del Problem Set: Etapa 1: Understanding the Data.
# Insumos:  00_data/raw/ (train/test de hogares y personas).
# Salidas:  03_temp/train_clean.rds y 03_temp/test_clean.rds (variables de
#           X_1 a X_6), y 03_temp/train_X7.rds y 03_temp/test_X7.rds
#           (conjunto ampliado X_7).
##########################################################

# Paquetes y funciones -----------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(data.table, tidyverse)

source("01_code/functions/fun_preprocess_personas.R")
source("01_code/functions/fun_colapse_personas.R")
source("01_code/functions/fun_preprocess_hogares.R")
source("01_code/functions/fun_conteos_personas.R")
source("01_code/functions/fun_variables_lgbm.R")

# Lectura de los datos crudos ----------------------------------------------
db_train_hogares  <- read.csv("00_data/raw/train_hogares.csv")
db_train_personas <- read.csv("00_data/raw/train_personas.csv")
db_test_hogares   <- read.csv("00_data/raw/test_hogares.csv")
db_test_personas  <- read.csv("00_data/raw/test_personas.csv")

# Personas: limpieza y colapso a nivel hogar -------------------------------
db_train_personas_hogar <- db_train_personas |>
  fun_preprocess_personas() |>
  fun_colapse_personas() |>
  left_join(fun_conteos_personas(db_train_personas), by = "id")

db_test_personas_hogar <- db_test_personas |>
  fun_preprocess_personas() |>
  fun_colapse_personas() |>
  left_join(fun_conteos_personas(db_test_personas), by = "id")

# Hogares: preparación -----------------------------------------------------
db_train_hogares <- fun_preprocess_hogares(db_train_hogares)
db_test_hogares  <- fun_preprocess_hogares(db_test_hogares)

# Unión de hogares y personas ----------------------------------------------
# La base de hogares define las filas: una por hogar.
db_train <- db_train_hogares |>
  left_join(db_train_personas_hogar, by = "id")

db_test <- db_test_hogares |>
  left_join(db_test_personas_hogar, by = "id")

# Verificaciones -----------------------------------------------------------
# Antes de guardar, se confirma que la unión no perdió, duplicó ni dejó sin
# información a ningún hogar. Si algo falla, el script se detiene aquí.
fun_verificar_base <- function(.db, .db_hogares, .nombre) {
  if (nrow(.db) != nrow(.db_hogares)) {
    stop(.nombre, ": la unión cambió el número de hogares (",
         nrow(.db_hogares), " antes, ", nrow(.db), " después).", call. = FALSE)
  }
  if (anyDuplicated(.db$id) > 0) {
    stop(.nombre, ": hay hogares repetidos.", call. = FALSE)
  }
  if (anyNA(.db)) {
    columnas_na <- names(.db)[colSums(is.na(.db)) > 0]
    stop(.nombre, ": hay valores faltantes en ",
         paste(columnas_na, collapse = ", "), ".", call. = FALSE)
  }
  message(.nombre, ": ", nrow(.db), " hogares y ", ncol(.db),
          " columnas, sin faltantes ni duplicados.")
}

fun_verificar_base(db_train, db_train_hogares, "Train")
fun_verificar_base(db_test, db_test_hogares, "Test")

# Train y test deben tener los mismos predictores; train además trae Pobre.
if (!setequal(setdiff(names(db_train), "Pobre"), names(db_test))) {
  stop("Train y test no tienen las mismas columnas.", call. = FALSE)
}

# Id -----------------------------------------------------------------------
# En train, el id no es un predictor: se elimina para que ningún modelo lo
# use por error. En test se conserva, porque es la llave del archivo que se 
# sube a Kaggle. Se elimina después de verificar, porque la revisión de hogares 
# repetidos lo necesita.
ids_train <- db_train$id
db_train <- db_train |>
  select(-id) |>
  relocate(Pobre)

# Guardar ------------------------------------------------------------------
dir.create("03_temp", showWarnings = FALSE)
saveRDS(db_train, "03_temp/train_clean.rds")
saveRDS(db_test, "03_temp/test_clean.rds")
message("Bases guardadas en 03_temp/train_clean.rds y 03_temp/test_clean.rds.")


# Se liberan los objetos de este script: las bases de personas ocupan cientos
# de MB y los scripts siguientes leen las bases limpias desde 03_temp/.
rm(db_train_hogares, db_train_personas, db_test_hogares, db_test_personas,
   db_train_personas_hogar, db_test_personas_hogar, db_train, db_test,
   fun_verificar_base)
gc()

# Conjunto ampliado X_7 --------------------------------------------------------
# Descripción mucho más detallada del hogar (1.296 variables, ver
# fun_variables_lgbm.R), pensada para algoritmos de árboles que manejan
# muchas variables y valores faltantes, como el boosting. Se guarda en bases
# aparte, y no en train_clean.rds, porque ocupa unos 2 GB en memoria y los
# demás modelos no la necesitan. Se construye desde los CSV crudos porque
# usa columnas que la limpieza anterior no conserva (P5010, P5100, Depto,
# Clase, Npersug, Lp y Li). Solo se usan columnas que existen en train y en
# test.
cabeceras <- lapply(c(hh_train = "00_data/raw/train_hogares.csv",
                      hh_test = "00_data/raw/test_hogares.csv",
                      pp_train = "00_data/raw/train_personas.csv",
                      pp_test = "00_data/raw/test_personas.csv"),
                    function(r) names(fread(r, nrows = 0)))
comunes_hh <- intersect(cabeceras$hh_train, cabeceras$hh_test)
comunes_pp <- intersect(cabeceras$pp_train, cabeceras$pp_test)

fun_leer <- function(.ruta, .columnas) {
  # id, Depto y Clase se leen como texto: son códigos, no números.
  fread(.ruta, select = .columnas,
        colClasses = list(character = intersect(c("id", "Depto", "Clase"),
                                                .columnas)))
}

# Categóricas de ubicación: se guardan como códigos enteros, con los niveles
# de train y test juntos y ordenados, para que un mismo lugar tenga el mismo
# código en las dos bases.
ubicacion <- rbind(fun_leer("00_data/raw/train_hogares.csv", c("Dominio", "Depto", "Clase")),
                   fun_leer("00_data/raw/test_hogares.csv", c("Dominio", "Depto", "Clase")))
niveles_ubicacion <- lapply(c(hh_Dominio = "Dominio", hh_Depto = "Depto",
                              hh_Clase = "Clase"),
                            function(v) sort(unique(ubicacion[[v]]), method = "radix"))
rm(ubicacion)

# Train. Los factores de expansión (prefijo survey_) no son predictores:
# describen el diseño de la encuesta, no al hogar.
db_X7_train <- fun_variables_lgbm(
  .db_hogares = fun_leer("00_data/raw/train_hogares.csv", comunes_hh),
  .db_personas = fun_leer("00_data/raw/train_personas.csv", comunes_pp)
)
columnas_X7 <- copy(attr(db_X7_train, "lgbm_feature_columns"))
columnas_X7 <- columnas_X7[!startsWith(columnas_X7, "survey_")]
# Los hogares deben estar en el mismo orden que en train_clean.rds, porque
# las particiones de validación cruzada son posiciones de fila.
stopifnot(identical(db_X7_train$id, ids_train), length(columnas_X7) == 1296L)
db_X7_train <- db_X7_train[, c("id", columnas_X7), with = FALSE]
for (v in names(niveles_ubicacion)) {
  set(db_X7_train, j = v,
      value = match(db_X7_train[[v]], niveles_ubicacion[[v]]) - 1L)
}
saveRDS(db_X7_train, "03_temp/train_X7.rds")
rm(db_X7_train); gc()

# Test, con las mismas columnas y en el mismo orden que train. Si en el test
# no aparece alguna categoría, su conteo es 0.
db_X7_test <- fun_variables_lgbm(
  .db_hogares = fun_leer("00_data/raw/test_hogares.csv", comunes_hh),
  .db_personas = fun_leer("00_data/raw/test_personas.csv", comunes_pp)
)
for (v in setdiff(columnas_X7, names(db_X7_test))) {
  set(db_X7_test, j = v,
      value = if (startsWith(v, "category_")) 0 else NA_real_)
}
db_X7_test <- db_X7_test[, c("id", columnas_X7), with = FALSE]
for (v in names(niveles_ubicacion)) {
  set(db_X7_test, j = v,
      value = match(db_X7_test[[v]], niveles_ubicacion[[v]]) - 1L)
}
stopifnot(!anyNA(db_X7_test$hh_Dominio))
saveRDS(db_X7_test, "03_temp/test_X7.rds")
message("Conjunto X_7 guardado en 03_temp/train_X7.rds y 03_temp/test_X7.rds (",
        length(columnas_X7), " variables).")

rm(db_X7_test, cabeceras, comunes_hh, comunes_pp, fun_leer, niveles_ubicacion,
   columnas_X7, ids_train, v)
gc()
