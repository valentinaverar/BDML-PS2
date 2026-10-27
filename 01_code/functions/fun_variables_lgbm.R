##########################################################
# fun_variables_lgbm.R
#
# Funciones para el conjunto ampliado de predictores X_7: 1.296 variables
# del hogar construidas a partir de todas las columnas que train y test
# comparten (se construyen en 02_clean_merge_data.R), y su paso a matriz
# para los modelos de boosting. Requiere que el paquete data.table esté
# cargado.
#
# Es independiente de las funciones fun_preprocess_*, fun_colapse_personas y
# fun_conteos_personas, que construyen los conjuntos X_1 a X_6 de los demás
# modelos: así ningún cambio aquí altera esos conjuntos.
##########################################################

fun_variables_lgbm <- function(.db_hogares, .db_personas) {
  #' Construir las variables del conjunto ampliado X_7
  #'
  #' @description \code{fun_variables_lgbm()} combina tres tipos de
  #' información: (1) las variables del hogar tal como vienen en la base;
  #' (2) agregados de las personas del hogar (conteos, proporciones y
  #' estadísticas como media, máximo, mínimo, desviación y suma) para cuatro
  #' grupos: todos los miembros, la unidad de gasto, los adultos y los
  #' ocupados; y (3) las características del jefe, del cónyuge y de los tres
  #' miembros de mayor edad, más razones del arriendo, la hipoteca y la
  #' composición del hogar. Los nombres de las variables (en inglés) se
  #' conservan como en el envío original a Kaggle.
  #'
  #' @param .db_hogares \code{data.table}. Base de hogares de Kaggle con solo
  #' las columnas que existen en train y en test (más \code{Pobre}, en
  #' train). Las columnas id, Depto y Clase deben leerse como texto.
  #' @param .db_personas \code{data.table}. Base de personas de Kaggle con
  #' solo las columnas que existen en train y en test.
  #'
  #' @returns \code{data.table}. Una fila por hogar, en el orden de
  #' \code{.db_hogares}, con \code{id}, las variables y, en train,
  #' \code{Pobre} (0/1). El atributo \code{"lgbm_feature_columns"} guarda
  #' los nombres de las variables en el orden en que se crearon.
  
  # Variables que se calculan a partir del ingreso: nunca pueden ser
  # predictores, porque no existen en el test y definen el resultado.
  prohibidas <- c("Indigente", "Npobres", "Nindigentes", "Ingpcug",
                  "Ingtotug", "Ingtotugarr")
  stopifnot(!any(prohibidas %in% names(.db_hogares)))
  
  # Funciones auxiliares ----------------------------------------------------
  # Indicador sin NA: un NA cuenta como "no cumple la condición".
  indicador <- function(x) { x[is.na(x)] <- FALSE; x }
  eq <- function(x, valor) indicador(x == valor)
  # División que devuelve NA (y no Inf) cuando el denominador es 0.
  dividir <- function(a, b) { b[b == 0 & !is.na(b)] <- NA_real_; a / b }
  # Redondeo a precisión simple (32 bits), como en el envío original.
  f32 <- function(x) {
    x <- as.numeric(x)
    x[!is.finite(x)] <- NA_real_
    readBin(writeBin(x, raw(), size = 4L, endian = "little"),
            what = "numeric", n = length(x), size = 4L, endian = "little")
  }
  
  # 1. Variables del hogar ------------------------------------------------------
  h <- data.table::copy(data.table::as.data.table(.db_hogares))
  stopifnot(!anyDuplicated(h$id), !anyNA(h$id))
  ids <- h$id
  nh <- nrow(h)
  features <- list()
  agregar <- function(nombre_col, valor) {
    stopifnot(length(valor) == nh)
    features[[nombre_col]] <<- valor
  }
  for (c in setdiff(names(h), c("id", "Pobre"))) {
    s <- h[[c]]
    # Hipoteca (P5100) y arriendo (P5130, P5140): 98 = no sabe, 99 = no
    # informa. Se dejan como NA.
    if (c %in% c("P5100", "P5130", "P5140")) s[s %in% c(98, 99)] <- NA_real_
    # Los factores de expansión se marcan con "survey_" para excluirlos
    # después: describen el diseño de la encuesta, no al hogar.
    prefijo <- if (startsWith(c, "Fex_")) "survey_" else "hh_"
    agregar(paste0(prefijo, c), s)
  }
  if ("Pobre" %in% names(h)) stopifnot(all(h$Pobre %in% c(0L, 1L)))
  
  # 2. Variables de las personas ------------------------------------------------
  p <- data.table::copy(data.table::as.data.table(.db_personas))
  stopifnot(!anyDuplicated(p[, .(id, Orden)]))
  # Códigos de "no sabe": P6210s1 = 99 (grado) y P6426 >= 998 (antigüedad).
  p[P6210s1 == 99, P6210s1 := NA_real_]
  p[P6426 >= 998, P6426 := NA_real_]
  # Años de educación: 0 para ninguno o preescolar; el grado para básica y
  # media; 11 + grado para superior.
  p[, education_years := fifelse(P6210 %in% c(3, 4, 5), as.numeric(P6210s1), NA_real_)]
  p[P6210 %in% c(1, 2), education_years := 0]
  p[P6210 == 6, education_years := 11 + P6210s1]
  p[, education_level := as.numeric(P6210)]
  p[P6210 == 9, education_level := NA_real_]
  # Primer dígito del oficio.
  p[, occupation_group := floor(Oficio / 10)]
  
  edad <- p$P6040
  # Grupos sobre los que se agrega: todos, unidad de gasto (sin servicio
  # doméstico, pensionistas ni trabajadores), adultos y ocupados.
  grupos <- list(all = rep(TRUE, nrow(p)), ug = !(p$P6050 %in% c(6, 7, 8)),
                 adult = indicador(edad >= 18), work = eq(p$Oc, 1))
  # Indicadores por persona que se cuentan en cada grupo.
  flags <- list(
    age_0_5 = indicador(edad < 6), age_6_11 = indicador(edad >= 6 & edad <= 11),
    age_12_17 = indicador(edad >= 12 & edad <= 17),
    age_18_29 = indicador(edad >= 18 & edad <= 29),
    age_30_59 = indicador(edad >= 30 & edad <= 59),
    age_60_plus = indicador(edad >= 60), age_65_plus = indicador(edad >= 65),
    adult = indicador(edad >= 18), child = indicador(edad < 18),
    working_age = indicador(edad >= 18 & edad <= 64), female = eq(p$P6020, 2),
    employed = eq(p$Oc, 1), unemployed = eq(p$Des, 1), inactive = eq(p$Ina, 1),
    pension_contributor = eq(p$P6920, 1), pensioner_job = eq(p$P6920, 3),
    health_contributive = eq(p$P6100, 1), health_special = eq(p$P6100, 2),
    health_subsidized = eq(p$P6100, 3), health_uninsured = eq(p$P6090, 2),
    education_none = eq(p$P6210, 1), education_primary = eq(p$P6210, 3),
    education_secondary = p$P6210 %in% c(4, 5), education_higher = eq(p$P6210, 6),
    private_employee = eq(p$P6430, 1), public_employee = eq(p$P6430, 2),
    self_employed = eq(p$P6430, 4), employer = eq(p$P6430, 5),
    unpaid_worker = p$P6430 %in% c(6, 7), day_laborer = eq(p$P6430, 8),
    domestic_employee = eq(p$P6430, 3), small_firm = indicador(p$P6870 <= 3),
    large_firm = indicador(p$P6870 >= 7), part_time = indicador(p$P6800 < 35),
    long_hours = indicador(p$P6800 > 48), second_job = eq(p$P7040, 1),
    pension_income = eq(p$P7500s2, 1), domestic_transfers = eq(p$P7510s1, 1),
    foreign_transfers = eq(p$P7510s2, 1), institution_transfers = eq(p$P7510s3, 1),
    female_employed = eq(p$P6020, 2) & eq(p$Oc, 1),
    child_employed = indicador(edad < 18) & eq(p$Oc, 1),
    formal_proxy = eq(p$Oc, 1) & (eq(p$P6920, 1) | p$P6100 %in% c(1, 2)),
    informal_proxy = eq(p$Oc, 1) & !eq(p$P6920, 1) & !(p$P6100 %in% c(1, 2)))
  # Columnas de identificación y diseño: no se usan como variables de la
  # persona.
  meta <- c("id", "Orden", "Clase", "Dominio", "Depto", "Fex_c", "Fex_dpto")
  numericas_persona <- c("P6040", "P6210s1", "P6426", "P6800", "P7045")
  # Columnas que se cuentan por categoría y columnas que se copian para el
  # jefe y el cónyuge (el orden de los nombres define el orden de las
  # variables).
  categoricas <- setdiff(names(p), c(meta, numericas_persona,
                                     "education_years", "education_level"))
  persona_cols <- setdiff(names(p), meta)
  
  # Posición de cada persona en la base de hogares.
  p[, hh_idx := match(id, ids)]
  stopifnot(!anyNA(p$hh_idx), all(tabulate(p$hh_idx, nbins = nh) == h$Nper))
  
  # 3. Agregados por grupo ----------------------------------------------------------
  flag_names <- names(flags)
  flag_dt <- data.table::as.data.table(lapply(flags, as.numeric))
  flag_dt[, hh_idx := p$hh_idx]
  rm(flags)
  for (grupo in names(grupos)) {
    filas <- which(grupos[[grupo]])
    tamanio <- tabulate(p$hh_idx[filas], nbins = nh)
    agregar(paste0(grupo, "_n"), tamanio)
    # Conteo y proporción de cada indicador dentro del grupo.
    sumas <- flag_dt[filas, lapply(.SD, sum), by = hh_idx, .SDcols = flag_names]
    pos <- match(seq_len(nh), sumas$hh_idx)
    for (c in flag_names) {
      conteo <- sumas[[c]][pos]
      conteo[is.na(conteo)] <- 0
      agregar(paste0(grupo, "_count_", c), conteo)
      agregar(paste0(grupo, "_share_", c), dividir(conteo, tamanio))
    }
    # Media, máximo, mínimo, desviación y suma de las variables numéricas.
    numericas <- c("P6040", "education_years", "education_level", "P6210s1")
    if (grupo %in% c("all", "ug", "work")) numericas <- c(numericas, "P6426", "P6800", "P7045")
    datos_grupo <- p[filas, c("hh_idx", numericas), with = FALSE]
    for (c in numericas) {
      expresion <- substitute(list(mean = mean(V, na.rm = TRUE), max = max(V, na.rm = TRUE),
                                   min = min(V, na.rm = TRUE), std = sd(V, na.rm = TRUE), sum = sum(V, na.rm = TRUE)),
                              list(V = as.name(c)))
      estadisticas <- suppressWarnings(datos_grupo[, eval(expresion), by = hh_idx])
      posicion <- match(seq_len(nh), estadisticas$hh_idx)
      for (est in c("mean", "max", "min", "std", "sum")) {
        agregar(paste0(grupo, "_", c, "_", est), estadisticas[[est]][posicion])
      }
    }
  }
  rm(flag_dt, datos_grupo, estadisticas, sumas, grupos)
  
  # 4. Conteo y proporción de cada categoría de cada pregunta -----------------------
  # Sobre todos los miembros del hogar; NA se cuenta como la categoría -1.
  for (c in categoricas) {
    valores <- p[[c]]
    valores[is.na(valores)] <- -1
    valores <- as.integer(valores)
    # Las categorías se recorren en el orden en que aparecen en la base.
    niveles <- unique(valores)
    conteos <- data.table::data.table(hh_idx = p$hh_idx, nivel = valores)[
      , .(n = .N), by = .(hh_idx, nivel)]
    for (nivel in niveles) {
      cuenta <- numeric(nh)
      seleccion <- which(conteos$nivel == nivel)
      cuenta[conteos$hh_idx[seleccion]] <- conteos$n[seleccion]
      agregar(paste0("category_", c, "_count_", nivel), cuenta)
      agregar(paste0("category_", c, "_share_", nivel), cuenta / h$Nper)
    }
  }
  
  # 5. Jefe, cónyuge y los tres miembros de mayor edad -------------------------------
  for (rol in c("head", "partner")) {
    codigo <- if (rol == "head") 1 else 2
    filas <- which(eq(p$P6050, codigo))
    filas <- filas[!duplicated(p$id[filas])]
    pos <- match(ids, p$id[filas])
    for (c in persona_cols) agregar(paste0(rol, "_", c), p[[c]][filas][pos])
  }
  orden <- order(p$id, -p$P6040, p$P6050, na.last = TRUE, method = "radix")
  rango <- sequence(rle(p$id[orden])$lengths)
  miembro_cols <- c("P6040", "P6020", "P6050", "P6210", "P6210s1", "P6100",
                    "P6240", "P6430", "P6800", "P6920", "Oficio", "P6870")
  for (k in 1:3) {
    filas <- orden[rango == k]
    pos <- match(ids, p$id[filas])
    for (c in miembro_cols) agregar(paste0("member_", k, "_", c), p[[c]][filas][pos])
  }
  
  # 6. Arriendo, hipoteca y composición del hogar ---------------------------------
  # Arriendo: P5140 (quienes arriendan) o, si falta, P5130 (estimado).
  renta <- features$hh_P5140
  faltante <- is.na(renta)
  renta[faltante] <- features$hh_P5130[faltante]
  agregar("hh_rent_combined", renta)
  for (tipo in c("rent", "mortgage")) {
    s <- if (tipo == "rent") renta else features$hh_P5100
    for (den in c("Nper", "Npersug", "P5000", "P5010", "Lp")) {
      agregar(paste0("hh_", tipo, "_per_", den), dividir(s, h[[den]]))
    }
    agregar(paste0("hh_log_", tipo), log1p(pmax(s, 0)))
    agregar(paste0("hh_", tipo, "_missing"), as.integer(is.na(s)))
  }
  agregar("hh_crowding_bedroom", dividir(h$Nper, h$P5010))
  agregar("hh_crowding_room", dividir(h$Nper, h$P5000))
  agregar("hh_nper_minus_ug", h$Nper - h$Npersug)
  agregar("hh_rent_ug_povertyline", dividir(renta, h$Npersug * h$Lp))
  for (grupo in c("all", "ug")) {
    for (tipo in c("employed", "formal_proxy", "pension_income", "health_contributive")) {
      agregar(paste0("demo_", grupo, "_persons_per_", tipo),
              dividir(h$Npersug, features[[paste0(grupo, "_count_", tipo)]]))
    }
    agregar(paste0("demo_", grupo, "_dependent_per_worker"),
            dividir(features[[paste0(grupo, "_n")]] - features[[paste0(grupo, "_count_employed")]],
                    features[[paste0(grupo, "_count_employed")]]))
    agregar(paste0("demo_", grupo, "_dependent_age_ratio"),
            dividir(features[[paste0(grupo, "_count_child")]] + features[[paste0(grupo, "_count_age_65_plus")]],
                    features[[paste0(grupo, "_count_working_age")]]))
  }
  agregar("demo_no_partner", as.integer(is.na(features$partner_P6040)))
  agregar("demo_head_partner_age_gap", features$head_P6040 - features$partner_P6040)
  agregar("demo_head_partner_education_gap",
          features$head_education_years - features$partner_education_years)
  
  # 7. Resultado ---------------------------------------------------------------------------
  # Todo en precisión simple, salvo las categóricas de ubicación.
  cat_hh <- c("hh_Dominio", "hh_Depto", "hh_Clase")
  for (c in setdiff(names(features), cat_hh)) features[[c]] <- f32(features[[c]])
  stopifnot(all(features$all_n == h$Nper), all(features$ug_n == h$Npersug),
            all(features$head_P6050 == 1))
  db_result <- data.table::as.data.table(c(list(id = ids), features))
  if ("Pobre" %in% names(h)) db_result[, Pobre := as.integer(h$Pobre)]
  data.table::setattr(db_result, "lgbm_feature_columns", names(features))
  
  return(db_result)
}

fun_matriz_x7 <- function(.base, .columnas) {
  #' Pasar una base del conjunto X_7 a una matriz numérica
  #'
  #' @description \code{fun_matriz_x7()} arma la matriz que reciben
  #' LightGBM y XGBoost. La llena columna por columna y borra cada columna
  #' de la base apenas la copia: así nunca hay dos copias completas en
  #' memoria (cada una ocupa cerca de 2 GB).
  #'
  #' @param .base \code{data.table} leída de 03_temp/train_X7.rds o
  #' test_X7.rds. Se modifica: sus columnas se borran.
  #' @param .columnas \code{character}. Columnas de la matriz, en orden.
  #'
  #' @returns \code{matrix} numérica con una fila por hogar.
  # copy(): si .columnas fuera names(.base), cambiaría al borrar columnas.
  .columnas <- data.table::copy(.columnas)
  stopifnot(all(.columnas %in% names(.base)))
  matriz <- matrix(NA_real_, nrow = nrow(.base), ncol = length(.columnas),
                   dimnames = list(NULL, .columnas))
  for (j in seq_along(.columnas)) {
    matriz[, j] <- as.numeric(.base[[.columnas[j]]])
    data.table::set(.base, j = .columnas[j], value = NULL)
  }
  matriz
}
