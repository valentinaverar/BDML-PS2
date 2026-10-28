##########################################################
# 04_descriptive_analysis.R
#
# Describe la pobreza y compara el balance train-test con y sin Bogotá.
# Conserva la comparación de las 15 variables con mayor cambio de balance
# y agrega balance, importancia y curvas descriptivas de los predictores de X7.
#
# Insumos: train_clean.rds, test_clean.rds, train_X7.rds y test_X7.rds.
# Salidas: métricas, figuras y resultados reutilizables en 02_outputs/artifacts/.
##########################################################

# TRUE calcula lo que falte; FALSE solo recupera los resultados guardados.
# Un artifact existente se reutiliza para no repetir las permutaciones.
RUN_HEAVY <- FALSE
ruta_importancia <- "02_outputs/artifacts/importancia_permutacion_lgbm.rds"

# Paquetes y datos ---------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, scales, gridExtra)

train <- readRDS("03_temp/train_clean.rds")
test <- readRDS("03_temp/test_clean.rds")
# 1. Distribución de pobreza ------------------------------------------------
dist_pobreza <- train %>%
  count(Pobre) %>%
  mutate(
    categoria = factor(if_else(Pobre == "Yes", "Poor", "Not Poor"),
                       levels = c("Poor", "Not Poor")),
    proporcion = n / sum(n),
    etiqueta = paste0(percent(proporcion, accuracy = 0.1),
                      " (n = ", comma(n), ")")
  )

distribucion_clase <- ggplot(dist_pobreza, aes(categoria, proporcion)) +
  geom_col(fill = "steelblue", width = 0.5) +
  geom_text(aes(label = etiqueta), vjust = -0.5, size = 3.5) +
  scale_y_continuous(labels = percent, limits = c(0, 1),
                     expand = expansion(mult = c(0, 0.05))) +
  labs(x = NULL, y = "Proportion of households by class") +
  theme_minimal()

ggsave("02_outputs/figures/distribucion_clase.pdf",
       plot = distribucion_clase, width = 7, height = 4)

# 2. Clasificador ingenuo: todos son NO POBRES -------------------------------
TP <- FP <- 0L
TN <- sum(train$Pobre == "No")
FN <- sum(train$Pobre == "Yes")

accuracy <- (TP + TN) / (TP + TN + FP + FN)
precision <- ifelse(TP + FP == 0, 0, TP / (TP + FP))
sensibilidad <- ifelse(TP + FN == 0, NA_real_, TP / (TP + FN))
especificidad <- ifelse(TN + FP == 0, NA_real_, TN / (TN + FP))
F1 <- ifelse(precision + sensibilidad == 0, 0,
             2 * precision * sensibilidad / (precision + sensibilidad))

metricas <- data.frame(
  Metrica = c("Accuracy", "F1-score", "Sensibilidad", "Especificidad"),
  Valor = c(accuracy, F1, sensibilidad, especificidad)
)
print(metricas)

valor <- function(m) {
  sprintf("%.2f", 100 * metricas$Valor[metricas$Metrica == m])
}
lineas <- c(
  paste0("\\def\\accBase{", valor("Accuracy"), "\\%}"),
  paste0("\\def\\f1Base{", valor("F1-score"), "\\%}"),
  paste0("\\def\\sensBase{", valor("Sensibilidad"), "\\%}"),
  paste0("\\def\\espBase{", valor("Especificidad"), "\\%}")
)
writeLines(lineas, "02_outputs/tables/metricas_baseline.tex")

# Etiquetas únicamente para las 15 variables del balance original.
etiquetas_balance <- c(
  num_si_P7510s6 = " per Recibieron cesantías o sus intereses",
  num_posicion_1 = "Empleados de empresa particular",
  num_asalariados = "Empleados asalariados",
  num_si_P6630s1 = "Personas que recibieron prima de servicios",
  num_si_P7510s7 = "Personas con ingresos de otras fuentes",
  bin_prima_servicios_ug = "Hogar que recibió prima de servicios",
  num_si_P6585s2 = "Personas que recibieron subsidio de transporte",
  prop_cotizantes_ocupados_ug = "Proporción de ocupados que cotizan a pensión",
  num_cotizantes = "Cotizantes a pensión (agregado)",
  num_pension_1 = "Cotizantes a pensión (conteo)",
  num_actividad_4 = "Personas dedicadas a oficios del hogar",
  num_tam_emp_1 = "Personas que trabajan solas",
  num_cuentapropia = "Trabajadores por cuenta propia (agregado)",
  num_posicion_4 = "Trabajadores por cuenta propia (conteo)",
  num_pension_2 = "Personas que no cotizan a pensión"
)

# Solo se construyen etiquetas para los 20 primeros del ranking de LightGBM.
# Son etiquetas del gráfico; las columnas de las bases conservan sus nombres.
etiquetar_top20 <- function(variables) {
  fuentes <- c(
    Dominio = "ciudad o zona", Depto = "departamento", Clase = "zona de residencia",
    Nper = "personas del hogar", Npersug = "personas de la unidad de gasto",
    P5000 = "cuartos", P5010 = "dormitorios", P5090 = "tenencia de la vivienda",
    P5100 = "cuota de vivienda", P5130 = "arriendo estimado", P5140 = "arriendo pagado",
    Lp = "línea de pobreza", Li = "línea de pobreza extrema",
    P6040 = "edad", P6210 = "nivel educativo", P6210s1 = "grado escolar",
    P6426 = "antigüedad laboral", P6430 = "posición ocupacional", P6800 = "horas trabajadas",
    P6870 = "tamaño de la empresa", P6920 = "cotización a pensión",
    P6090 = "afiliación a salud", P6100 = "régimen de salud", P6240 = "actividad principal",
    P6630s1 = "prima de servicios", P7500s2 = "ingresos por pensión",
    P7510s1 = "dinero de otros hogares", P7510s6 = "cesantías", P7045 = "horas en segundo trabajo",
    education_years = "años de educación", education_level = "nivel educativo",
    employed = "ocupados", unemployed = "desocupados", inactive = "inactivos",
    pension_contributor = "cotizantes a pensión", pension_income = "personas que reciben pensión",
    formal_proxy = "ocupados formales", informal_proxy = "ocupados informales",
    health_contributive = "afiliados a salud contributiva", health_subsidized = "afiliados a salud subsidiada",
    working_age = "personas en edad de trabajar", child = "menores de edad", female = "mujeres",
    age_0_5 = "menores de 6 años", age_65_plus = "personas de 65 años o más",
    rent_combined = "arriendo pagado o estimado", log_rent = "arriendo (logaritmo)",
    crowding_room = "personas por cuarto", crowding_bedroom = "personas por dormitorio",
    rent_ug_povertyline = "arriendo por persona / línea de pobreza"
  )
  traducir <- function(x) {
    texto <- unname(fuentes[x])
    ifelse(is.na(texto), gsub("_", " ", x), texto)
  }
  grupos <- c(all = "hogar", ug = "unidad de gasto", adult = "adultos", work = "ocupados")
  estadisticas <- c(mean = "promedio", max = "máximo", min = "mínimo", std = "desviación", sum = "total")
  variables <- head(variables, 20)
  setNames(vapply(variables, function(v) {
    partes <- function(patron) regmatches(v, regexec(patron, v))[[1]][-1]
    m <- partes("^(all|ug|adult|work)_(count|share)_(.+)$")
    if (length(m)) return(paste0(if (m[2] == "count") "Número de " else "Proporción de ", traducir(m[3]), " (", grupos[m[1]], ")"))
    m <- partes("^(all|ug|adult|work)_(.+)_(mean|max|min|std|sum)$")
    if (length(m)) return(paste0(traducir(m[2]), ": ", estadisticas[m[3]], " (", grupos[m[1]], ")"))
    m <- partes("^(head|partner|member_[1-3])_(.+)$")
    if (length(m)) return(paste0(traducir(m[2]), " (", c(head = "jefe", partner = "cónyuge", member_1 = "miembro de mayor edad", member_2 = "segundo miembro de mayor edad", member_3 = "tercer miembro de mayor edad")[m[1]], ")"))
    m <- partes("^category_(.+)_(count|share)_(-?[0-9]+)$")
    if (length(m)) return(paste0(if (m[2] == "count") "Personas: " else "Proporción: ", traducir(m[1]), " (categoría ", m[3], ")"))
    m <- partes("^hh_(rent|mortgage)_per_(.+)$")
    if (length(m)) return(paste0(if (m[1] == "rent") "Arriendo" else "Cuota de vivienda", " / ", traducir(m[2])))
    m <- partes("^demo_(all|ug)_persons_per_(.+)$")
    if (length(m)) return(paste0("Personas por número de ", traducir(m[2]), " (", grupos[m[1]], ")"))
    m <- partes("^demo_(all|ug)_dependent_(per_worker|age_ratio)$")
    if (length(m)) return(paste0(if (m[2] == "per_worker") "Dependientes por ocupado" else "Razón de dependencia por edad", " (", grupos[m[1]], ")"))
    traducir(sub("^hh_", "", v))
  }, character(1)), variables)
}

es_categorica <- function(v) {
  grepl("^hh_(Dominio|Depto|Clase|P5090)$", v) |
    (grepl("^(head|partner|member_[1-3])_", v) &
       !grepl("_(P6040|P6210s1|P6426|P6800|P7045|education_years|education_level)$", v))
}

# 3. Comparación con y sin Bogotá ------------------------------------------
# Estas funciones se usan en las dos figuras: cambia la selección de variables.
sin_bogota <- train$cat_ubicacion != "BOGOTA"

calcular_ic <- function(x, y) {
  x <- x[is.finite(x)]
  y <- y[is.finite(y)]
  if (length(x) < 2 || length(y) < 2) {
    return(c(Estimacion = NA, Inferior = NA, Superior = NA))
  }
  sd_pool <- sqrt((var(x) + var(y)) / 2)
  if (!is.finite(sd_pool) || sd_pool == 0) {
    return(c(Estimacion = NA, Inferior = NA, Superior = NA))
  }
  prueba <- t.test(x, y, conf.level = 0.95)
  c(Estimacion = unname(prueba$estimate[1] - prueba$estimate[2]) / sd_pool,
    Inferior = prueba$conf.int[1] / sd_pool,
    Superior = prueba$conf.int[2] / sd_pool)
}

comparar_bogota <- function(datos_train, datos_test, variables) {
  bind_rows(lapply(variables, function(v) {
    bind_rows(
      data.frame(Variable = v, Grupo = "Con Bogotá",
                 t(calcular_ic(datos_train[[v]], datos_test[[v]]))),
      data.frame(Variable = v, Grupo = "Sin Bogotá",
                 t(calcular_ic(datos_train[[v]][sin_bogota], datos_test[[v]])))
    )
  }))
}

graficar_balance <- function(balance, variables, subtitulo,
                             etiquetas = etiquetas_balance[variables]) {
  datos <- balance %>%
    filter(Variable %in% variables) %>%
    mutate(Variable = factor(Variable, levels = rev(variables)))

  ggplot(datos, aes(Estimacion, Variable, color = Grupo, group = Grupo)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray45") +
    geom_errorbar(aes(xmin = Inferior, xmax = Superior), orientation = "y",
                  position = position_dodge(width = 0.5),
                  width = 0.25, linewidth = 0.7) +
    geom_point(position = position_dodge(width = 0.5), size = 2) +
    scale_color_manual(values = c("Con Bogotá" = "#D97736", "Sin Bogotá" = "#3285BA")) +
    scale_y_discrete(labels = function(x) stringr::str_wrap(setNames(etiquetas, variables)[x], 42)) +
    labs(title = "Balance con y sin Bogotá", subtitle = subtitulo,
         x = "Diferencia estandarizada (train - test)", y = NULL, color = NULL) +
    theme_minimal(base_size = 11) +
    theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold"),
          axis.text.y = element_text(size = 9), legend.position = "bottom")
}

vars <- intersect(names(train), names(test))
vars <- vars[sapply(train[vars], is.numeric) & sapply(test[vars], is.numeric)]
balance_ic <- comparar_bogota(train, test, vars)

# Se conserva el criterio original: mayor cambio en |SMD| al quitar Bogotá.
top_vars <- balance_ic %>%
  mutate(SMD_abs = abs(Estimacion)) %>%
  select(Variable, Grupo, SMD_abs) %>%
  pivot_wider(names_from = Grupo, values_from = SMD_abs) %>%
  mutate(cambio = abs(.data[["Con Bogotá"]] - .data[["Sin Bogotá"]])) %>%
  filter(is.finite(cambio)) %>%
  arrange(desc(cambio)) %>%
  slice_head(n = 15)

comparacion_bogota_ic <- graficar_balance(
  balance_ic, top_vars$Variable, "Diferencias de medias estandarizadas e IC del 95%"
)
ggsave("02_outputs/figures/comparacion_Bogota_IC.pdf",
       plot = comparacion_bogota_ic, width = 8, height = 5)

# Curvas descriptivas para los cuatro primeros del ranking.
# Continuas: tasa observada por grupos de cuantiles.
# Categóricas: tasa por categoría; no se traza una curva entre códigos nominales.
calcular_formas <- function(datos, y, dominios = NULL) {
  setNames(lapply(names(datos), function(v) {
    x <- as.numeric(datos[[v]])
    finitos <- is.finite(x)
    categorica <- es_categorica(v) || length(unique(x[finitos])) <= 2

    if (categorica) {
      valores <- ifelse(finitos, as.character(x), "Sin dato")
      niveles <- names(head(sort(table(valores), decreasing = TRUE), 12))
      grupos <- factor(valores, levels = niveles)
      mostrar <- !is.na(grupos)
      grupos <- droplevels(grupos[mostrar])
      x_plot <- x[mostrar]
      if (v == "hh_Dominio" && !is.null(dominios)) {
        texto <- as.character(dominios[match(niveles, as.character(x))])
        niveles[!is.na(texto)] <- texto[!is.na(texto)]
      }
      nota <- "Tasas por categoría (hasta 12 más frecuentes)"
    } else {
      rango <- as.numeric(quantile(x[finitos], c(0.01, 0.99)))
      if (diff(rango) == 0) rango <- range(x[finitos])
      mostrar <- finitos & x >= rango[1] & x <= rango[2]
      x_plot <- x[mostrar]
      cortes <- unique(as.numeric(quantile(x_plot, seq(0, 1, length.out = 21))))
      grupos <- if (length(unique(x_plot)) <= 20) factor(x_plot) else
        cut(x_plot, breaks = cortes, include.lowest = TRUE)
      nota <- "Tasas por grupos; rango P1–P99"
    }

    indices <- split(seq_along(x_plot), grupos, drop = TRUE)
    resumen <- do.call(rbind, lapply(seq_along(indices), function(i) {
      idx <- indices[[i]]
      data.frame(Valor = if (categorica) i else mean(x_plot[idx]),
                 n = length(idx), Tasa = mean(y[mostrar][idx]),
                 Etiqueta = if (categorica) niveles[i] else "")
    }))
    # Intervalos binomiales de Wilson al 95%, también válidos si la tasa es 0 o 1.
    z <- qnorm(0.975)
    centro <- (resumen$Tasa + z^2 / (2 * resumen$n)) / (1 + z^2 / resumen$n)
    margen <- z * sqrt(resumen$Tasa * (1 - resumen$Tasa) / resumen$n +
                         z^2 / (4 * resumen$n^2)) / (1 + z^2 / resumen$n)
    resumen$Inferior <- pmax(0, centro - margen)
    resumen$Superior <- pmin(1, centro + margen)
    list(observado = resumen, categorica = categorica,
         tasa_total = mean(y), nota = nota)
  }), names(datos))
}

# 4. Importancia por permutación del LightGBM ganador ------------------------
calcular_importancia_lgbm <- function() {
  pacman::p_load(data.table, caret, lightgbm)
  # Misma configuración de X7: 31 hojas, 748 rondas y umbral elegido por F1.
  train_x7 <- readRDS("03_temp/train_X7.rds")
  X_7_ampliado <- setdiff(names(train_x7), c("id", "Pobre"))

  parametros_lgbm <- list(
    objective = "binary", metric = "binary_logloss", boosting = "gbdt",
    learning_rate = 0.04, num_leaves = 31L, max_depth = -1L,
    min_data_in_leaf = 100L, min_sum_hessian_in_leaf = 0.001,
    feature_fraction = 0.85, bagging_fraction = 0.85, bagging_freq = 1L,
    lambda_l1 = 0.1, lambda_l2 = 10, min_gain_to_split = 0,
    max_bin = 127L, bin_construct_sample_cnt = 200000L,
    seed = 20261022L, deterministic = TRUE, force_col_wise = TRUE,
    num_threads = max(1L, min(6L, parallel::detectCores())), verbosity = -1L
  )
  umbral_lgbm <- 0.393939393939394
  categorias_lgbm <- c("hh_Dominio", "hh_Depto", "hh_Clase")
  n_perm <- 3L
  n_validacion <- 2000L
  y <- as.integer(train$Pobre == "Yes")

  # Se crean en memoria los mismos folds de validación del proyecto.
  set.seed(2026)
  folds <- caret::createFolds(train$Pobre, k = 5, returnTrain = TRUE)

  # Se libera cada columna de la base al copiarla para no duplicar todo X7.
  x_train <- matrix(NA_real_, nrow(train), length(X_7_ampliado),
                    dimnames = list(NULL, X_7_ampliado))
  for (v in X_7_ampliado) {
    x_train[, v] <- as.numeric(train_x7[[v]])
    data.table::set(train_x7, j = v, value = NULL)
  }
  rm(train_x7)
  gc()

  f1_lgbm <- function(real, prob) {
    pred <- prob >= umbral_lgbm
    2 * sum(real == 1 & pred) / (sum(real == 1) + sum(pred))
  }

  importancia_folds <- matrix(NA_real_, length(X_7_ampliado), length(folds),
                              dimnames = list(X_7_ampliado, names(folds)))

  for (k in seq_along(folds)) {
    idx_train <- folds[[k]]
    idx_valid <- setdiff(seq_len(nrow(train)), idx_train)

    # Muestra estratificada de unos 2.000 hogares por fold para hacer viable
    # permutar las 1.296 variables. Ninguno se usa para entrenar ese modelo.
    set.seed(2026 + k)
    muestra <- caret::createDataPartition(
      train$Pobre[idx_valid], p = min(1, n_validacion / length(idx_valid)), list = FALSE
    )
    idx_valid <- idx_valid[as.vector(muestra)]

    datos_fold <- lightgbm::lgb.Dataset(
      x_train[idx_train, , drop = FALSE], label = y[idx_train],
      categorical_feature = categorias_lgbm, params = parametros_lgbm,
      free_raw_data = TRUE
    )
    modelo_fold <- lightgbm::lgb.train(
      params = parametros_lgbm, data = datos_fold, nrounds = 748L, verbose = -1L
    )
    datos_valid <- x_train[idx_valid, , drop = FALSE]
    f1_base <- f1_lgbm(y[idx_valid], predict(modelo_fold, datos_valid))

    set.seed(2026 + k)
    for (v in X_7_ampliado) {
      original <- datos_valid[, v]
      f1_permutado <- numeric(n_perm)
      for (r in seq_len(n_perm)) {
        datos_valid[, v] <- original[sample.int(nrow(datos_valid))]
        f1_permutado[r] <- f1_lgbm(y[idx_valid], predict(modelo_fold, datos_valid))
      }
      datos_valid[, v] <- original
      importancia_folds[v, k] <- f1_base - mean(f1_permutado)
    }
    message("Importancia por permutación: fold ", k, " de ", length(folds), " listo.")
    rm(modelo_fold, datos_fold, datos_valid)
    gc()
  }

  importancia_lgbm <- tibble(
    Variable = X_7_ampliado,
    Importancia_F1 = rowMeans(importancia_folds)
  ) %>%
    arrange(desc(Importancia_F1))

  # Balance de las variables seleccionadas, sin volver a entrenar.
  vars_importantes <- head(importancia_lgbm$Variable, 15)
  train_importantes <- as.data.frame(x_train[, vars_importantes, drop = FALSE],
                                     optional = TRUE)
  rm(x_train)
  gc()

  test_x7 <- readRDS("03_temp/test_X7.rds")
  test_importantes <- as.data.frame(test_x7[, vars_importantes, with = FALSE])
  rm(test_x7)
  gc()

  # Las categóricas se comparan como proporciones de su categoría modal.
  detalles <- setNames(rep("", length(vars_importantes)), vars_importantes)

  for (v in vars_importantes) {
    if (es_categorica(v)) {
      moda <- as.numeric(names(sort(table(train_importantes[[v]]), decreasing = TRUE))[1])
      etiqueta_categoria <- if (v == "hh_Dominio") {
        as.character(train$cat_ubicacion[which(train_importantes[[v]] == moda)[1]])
      } else {
        paste("categoría", moda)
      }
      detalles[v] <- etiqueta_categoria
      train_importantes[[v]] <- as.numeric(!is.na(train_importantes[[v]]) &
                                           train_importantes[[v]] == moda)
      test_importantes[[v]] <- as.numeric(!is.na(test_importantes[[v]]) &
                                          test_importantes[[v]] == moda)
    } else if (length(unique(na.omit(train_importantes[[v]]))) <= 1) {
      # Si el único cambio posible es que falte el dato, se compara esa proporción.
      detalles[v] <- "dato faltante"
      train_importantes[[v]] <- as.numeric(is.na(train_importantes[[v]]))
      test_importantes[[v]] <- as.numeric(is.na(test_importantes[[v]]))
    }
  }

  balance_importancia <- comparar_bogota(train_importantes, test_importantes,
                                        vars_importantes)

  list(importancia = importancia_lgbm, balance = balance_importancia, detalles = detalles,
       etiquetas = etiquetar_top20(importancia_lgbm$Variable))
}

# 5. Recuperar o completar el artifact --------------------------------------
guardar_resultados <- function(resultado) {
  dir.create(dirname(ruta_importancia), recursive = TRUE, showWarnings = FALSE)
  saveRDS(resultado, ruta_importancia)
}
resultados_lgbm <- if (file.exists(ruta_importancia)) {
  readRDS(ruta_importancia)
} else if (RUN_HEAVY) {
  resultado <- calcular_importancia_lgbm()
  guardar_resultados(resultado)  # Guardar antes de preparar las curvas.
  resultado
} else {
  message("Falta el artifact de LightGBM. Use RUN_HEAVY <- TRUE una vez.")
  NULL
}

if (!is.null(resultados_lgbm)) {
  importancia_lgbm <- resultados_lgbm$importancia
  etiquetas_top20 <- etiquetar_top20(importancia_lgbm$Variable)

  # Si ya existe la importancia, completar las curvas no repite LightGBM.
  if (is.null(resultados_lgbm$formas) && RUN_HEAVY) {
    pacman::p_load(data.table)
    top4 <- head(importancia_lgbm$Variable, 4)
    x7 <- readRDS("03_temp/train_X7.rds")
    datos_top4 <- as.data.frame(x7[, top4, with = FALSE])
    rm(x7)
    gc()
    resultados_lgbm$formas <- calcular_formas(
      datos_top4, as.integer(train$Pobre == "Yes"), train$cat_ubicacion
    )
    resultados_lgbm$etiquetas <- etiquetas_top20
    guardar_resultados(resultados_lgbm)
    rm(datos_top4)
  }

  # Balance de los 15 primeros; conserva el archivo y el formato anteriores.
  vars_importantes <- head(importancia_lgbm$Variable, 15)
  etiquetas_importantes <- etiquetas_top20[vars_importantes]
  detalle <- resultados_lgbm$detalles[vars_importantes]
  con_detalle <- !is.na(detalle) & nzchar(detalle)
  etiquetas_importantes[con_detalle] <- paste0(etiquetas_importantes[con_detalle], ": ", detalle[con_detalle])
  comparacion_bogota_importancia <- graficar_balance(
    resultados_lgbm$balance, vars_importantes,
    "15 variables más importantes del LightGBM ganador (X7); IC del 95%",
    etiquetas = etiquetas_importantes
  ) +
    labs(caption = "Importancia: caída del F1 por permutación en validación.\nCategóricas: proporción de la categoría indicada.")
  ggsave("02_outputs/figures/comparacion_Bogota_importancia_LGBM.pdf",
         plot = comparacion_bogota_importancia, width = 8, height = 7)

  # 6. Barras: caída del F1 en puntos, sin normalizar ni cambiar el ranking.
  datos_importancia <- head(importancia_lgbm, 15) %>%
    mutate(Variable = factor(Variable, levels = rev(Variable)),
           Puntos_F1 = 100 * Importancia_F1)
  importancia_top15 <- ggplot(datos_importancia, aes(Puntos_F1, Variable)) +
    geom_col(fill = "#3285BA", width = 0.7) +
    geom_vline(xintercept = 0, linewidth = 0.3) +
    scale_y_discrete(labels = function(x) stringr::str_wrap(etiquetas_top20[x], 42)) +
    labs(title = "Importancia por permutación del LightGBM",
         subtitle = "15 predictores principales; promedio de los cinco folds",
         x = "Caída media del F1 (puntos)", y = NULL) +
    theme_minimal(base_size = 11) +
    theme(panel.grid.major.y = element_blank())
  ggsave("02_outputs/figures/importancia_permutacion_LGBM_top15.pdf",
         plot = importancia_top15, width = 8, height = 6)

  # 7. Cuatro paneles: relación observada con pobreza.
  if (!is.null(resultados_lgbm$formas)) {
    pacman::p_load(gridExtra)
    paneles <- lapply(head(importancia_lgbm$Variable, 4), function(v) {
      forma <- resultados_lgbm$formas[[v]]
      p <- ggplot(forma$observado, aes(Valor, Tasa)) +
        geom_errorbar(aes(ymin = Inferior, ymax = Superior), width = 0, color = "#3285BA") +
        geom_point(color = "#3285BA", size = 2) +
        scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
        labs(title = stringr::str_wrap(etiquetas_top20[[v]], 38),
             subtitle = forma$nota, x = NULL, y = "Hogares pobres") +
        theme_minimal(base_size = 10) +
        theme(legend.position = "bottom", plot.title = element_text(size = 11))
      if (forma$categorica) {
        p + aes(x = factor(Valor)) +
          scale_x_discrete(labels = setNames(forma$observado$Etiqueta, forma$observado$Valor)) +
          geom_hline(yintercept = forma$tasa_total, color = "#D97736", linetype = "dashed") +
          theme(axis.text.x = element_text(angle = 35, hjust = 1))
      } else {
        p + geom_line(color = "#3285BA", linewidth = 0.5)
      }
    })
    formas_top4 <- gridExtra::arrangeGrob(
      grobs = paneles, ncol = 2,
      top = grid::textGrob("Relación observada con pobreza: cuatro predictores principales"),
      bottom = grid::textGrob(
        "Tasas observadas e IC del 95%. En categóricas, línea punteada: pobreza total.\nAsociaciones descriptivas; la elección del algoritmo se evalúa con F1 en validación.",
        gp = grid::gpar(fontsize = 8)
      )
    )
    ggsave("02_outputs/figures/forma_funcional_LGBM_top4.pdf",
           plot = formas_top4, width = 11, height = 8)
  } else {
    message("Use RUN_HEAVY <- TRUE para completar las curvas; se reutilizará la importancia guardada.")
  }
}
