##########################################################
# 17_eda_reporte.R
#
# Informe exploratorio en un solo PDF: cómo está armada la base
# (hogares, con las personas ya colapsadas) y cómo se distribuyen
# la clase y los 43 predictores. No reentrena modelos.
#
# Insumos: 03_temp/train_clean.rds, 03_temp/test_clean.rds
#          y los CSV crudos, solo para contar personas.
# Salida:  02_outputs/figures/analisis_exploratorio.pdf
##########################################################

if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, scales, grid, gridExtra)

if (!capabilities("cairo")) {
  stop("Hace falta el dispositivo cairo para guardar el PDF con tildes.",
       call. = FALSE)
}

train <- readRDS("03_temp/train_clean.rds")
test <- readRDS("03_temp/test_clean.rds")

# Cuenta filas de un CSV grande sin cargarlo: cada salto de línea es una fila.
fun_n_lineas <- function(ruta) {
  con <- file(ruta, open = "rb")
  on.exit(close(con), add = TRUE)
  n <- 0
  repeat {
    buf <- readBin(con, what = "raw", n = 1e7)
    if (length(buf) == 0) break
    n <- n + sum(buf == as.raw(10L))
  }
  n
}

n_personas_train <- fun_n_lineas("00_data/raw/train_personas.csv") - 1L
n_personas_test <- fun_n_lineas("00_data/raw/test_personas.csv") - 1L

inventario <- tribble(
  ~variable, ~bloque, ~tipo, ~definicion,
  "cat_ubicacion", "Vivienda", "Factor", "Dominio: 23 ciudades, resto urbano y rural",
  "cat_tenencia", "Vivienda", "Factor", "P5090. Seis formas de tenencia de la vivienda",
  "num_personas_cuarto", "Vivienda", "Razón", "Personas del hogar (Nper) / cuartos (P5000)",
  "num_personas_ug", "Vivienda", "Conteo", "Miembros que comparten el presupuesto",
  "num_minors", "Vivienda", "Conteo", "Personas con menos de 18 años",
  "bin_head_male", "Demografía", "Indicador", "1 si el jefe del hogar es hombre",
  "num_old", "Demografía", "Conteo", "Personas mayores de 65 años",
  "num_edad_trabajar", "Demografía", "Conteo", "Personas de 18 a 65 años",
  "prop_dependencia", "Demografía", "Razón", "(Menores + mayores de 65) / personas de 18 a 65",
  "cat_maxEduc", "Demografía", "Factor", "Máximo nivel educativo del hogar",
  "num_educ_jefe_conyuge", "Demografía", "Promedio", "Código educativo 1 a 6 del jefe y del cónyuge",
  "num_ocupados", "Empleo", "Conteo", "Ocupados en la unidad de gasto",
  "prop_ocupados", "Empleo", "Razón", "Ocupados / personas de la unidad de gasto",
  "num_inactivos", "Empleo", "Conteo", "Inactivos que tienen de 18 a 65 años",
  "prop_inactividad", "Empleo", "Razón", "Inactivos / personas de 18 a 65",
  "num_cotizantes", "Formalidad", "Conteo", "Cotizan a pensión (P6920) en la unidad de gasto",
  "bin_hogar_pensionado", "Formalidad", "Indicador", "Alguien de la unidad de gasto recibió pensión",
  "num_horas_ug", "Formalidad", "Suma", "Horas semanales del trabajo principal (P6800)",
  "num_asalariados", "Formalidad", "Conteo", "Empleados particulares o del gobierno",
  "num_cuentapropia", "Formalidad", "Conteo", "Trabajadores por cuenta propia",
  "cat_pos_jefe", "Formalidad", "Factor", "Posición ocupacional del jefe (P6430)",
  "num_personas_hogar", "Extendida", "Conteo", "Nper. Todas las personas que viven en el hogar",
  "num_cuartos", "Extendida", "Conteo", "P5000. Cuartos, incluida la sala-comedor",
  "num_log_arriendo_pc", "Extendida", "Continua", "log(arriendo por persona). 0 si no informa",
  "bin_arriendo_nd", "Extendida", "Indicador", "1 si el arriendo es 98 o 99 (no sabe o no informa)",
  "num_edad_jefe", "Extendida", "Continua", "Edad del jefe",
  "num_edad_promedio_ug", "Extendida", "Promedio", "Edad promedio de la unidad de gasto",
  "prop_menores_ug", "Extendida", "Razón", "Menores de 18 en la unidad de gasto / personas",
  "prop_mujeres_ug", "Extendida", "Razón", "Mujeres en la unidad de gasto / personas",
  "num_menores_6_ug", "Extendida", "Conteo", "Menores de 6 años en la unidad de gasto",
  "cat_educ_jefe", "Extendida", "Factor", "Nivel educativo del jefe",
  "num_adultos_universitarios_ug", "Extendida", "Conteo", "Adultos con educación universitaria en la UG",
  "bin_jefe_ocupado", "Extendida", "Indicador", "1 si el jefe está ocupado",
  "num_desocupados_ug", "Extendida", "Conteo", "Desocupados en la unidad de gasto",
  "tasa_desempleo_ug", "Extendida", "Razón", "Desocupados / (ocupados + desocupados)",
  "bin_sin_activos_ug", "Extendida", "Indicador", "1 si no hay ocupados ni desocupados",
  "num_estudiantes_ug", "Extendida", "Conteo", "Estudiaron la mayor parte de la semana (P6240)",
  "prop_cotizantes_ocupados_ug", "Extendida", "Razón", "Cotizantes / ocupados. 0 si no hay ocupados",
  "num_no_afiliados_salud_ug", "Extendida", "Conteo", "No afiliados a salud (P6090 = 2)",
  "num_subempleo_ug", "Extendida", "Conteo", "Ocupados que quieren trabajar más horas",
  "bin_prima_servicios_ug", "Extendida", "Indicador", "Alguien recibió prima de servicios",
  "bin_dinero_hogares_ug", "Extendida", "Indicador", "Alguien recibió dinero de otros hogares",
  "cat_tam_empresa_jefe", "Extendida", "Factor", "Tamaño de la empresa donde trabaja el jefe"
)

if (!setequal(inventario$variable, setdiff(names(train), "Pobre"))) {
  stop("El inventario no coincide con las columnas de train.", call. = FALSE)
}

nums <- inventario$variable[vapply(inventario$variable, function(v) {
  is.numeric(train[[v]])
}, logical(1))]
facs <- inventario$variable[vapply(inventario$variable, function(v) {
  is.factor(train[[v]])
}, logical(1))]
bins <- nums[vapply(nums, function(v) {
  all(train[[v]] %in% c(0, 1))
}, logical(1))]

fun_smd <- function(x, y) {
  sp <- sqrt((var(x) + var(y)) / 2)
  if (!is.finite(sp) || sp == 0) return(NA_real_)
  (mean(x) - mean(y)) / sp
}

tasa_global <- mean(train$Pobre == "Yes")
n_pobres <- sum(train$Pobre == "Yes")
n_no <- sum(train$Pobre == "No")

smd_clase <- tibble(variable = nums) |>
  mutate(
    smd = vapply(variable, function(v) {
      fun_smd(train[[v]][train$Pobre == "Yes"],
              train[[v]][train$Pobre == "No"])
    }, numeric(1)),
    mediana_pobre = vapply(variable, function(v) {
      median(train[[v]][train$Pobre == "Yes"])
    }, numeric(1)),
    mediana_no = vapply(variable, function(v) {
      median(train[[v]][train$Pobre == "No"])
    }, numeric(1)),
    pct_cero = vapply(variable, function(v) mean(train[[v]] == 0), numeric(1))
  ) |>
  arrange(desc(abs(smd)))

smd_muestra <- tibble(variable = nums) |>
  mutate(smd = vapply(variable, function(v) {
    fun_smd(train[[v]], test[[v]])
  }, numeric(1))) |>
  arrange(desc(abs(smd)))

indicadores <- tibble(variable = bins) |>
  mutate(
    n_uno = vapply(variable, function(v) sum(train[[v]] == 1), numeric(1)),
    tasa_0 = vapply(variable, function(v) {
      mean(train$Pobre[train[[v]] == 0] == "Yes")
    }, numeric(1)),
    tasa_1 = vapply(variable, function(v) {
      mean(train$Pobre[train[[v]] == 1] == "Yes")
    }, numeric(1))
  ) |>
  mutate(dif_pp = 100 * (tasa_1 - tasa_0)) |>
  arrange(desc(abs(dif_pp)))

p_exp <- ncol(model.matrix(
  reformulate(c(nums, facs)), data = train
)) - 1L
bases_factor <- tibble(
  factor = facs,
  niveles = vapply(facs, function(v) nlevels(train[[v]]), integer(1)),
  base = vapply(facs, function(v) levels(train[[v]])[1], character(1))
) |>
  mutate(columnas = niveles - 1L)

# Relaciones que salen de la construcción, no de un hallazgo estadístico.
dentro <- function(x) max(abs(x), na.rm = TRUE)
identidades <- tribble(
  ~relacion, ~error_maximo,
  "personas por cuarto = personas del hogar / cuartos",
  dentro(train$num_personas_cuarto - train$num_personas_hogar / train$num_cuartos),
  "proporción de ocupados = ocupados / personas de la UG",
  dentro(train$prop_ocupados - train$num_ocupados / train$num_personas_ug),
  "inactividad = inactivos / personas de 18 a 65, si las hay",
  dentro((train$prop_inactividad - train$num_inactivos / train$num_edad_trabajar)[
    train$num_edad_trabajar > 0]),
  "dependencia = (menores + mayores) / personas de 18 a 65, si las hay",
  dentro((train$prop_dependencia - (train$num_minors + train$num_old) /
            train$num_edad_trabajar)[train$num_edad_trabajar > 0]),
  "desempleo = desocupados / activos, si los hay",
  dentro((train$tasa_desempleo_ug - train$num_desocupados_ug /
            (train$num_ocupados + train$num_desocupados_ug))[
              train$num_ocupados + train$num_desocupados_ug > 0])
)

cm <- cor(train[nums])
pares <- which(upper.tri(cm) & abs(cm) >= 0.75, arr.ind = TRUE)
pares_altos <- if (nrow(pares) == 0) {
  tibble(variable_a = character(), variable_b = character(), correlacion = numeric())
} else {
  tibble(
    variable_a = nums[pares[, 1]],
    variable_b = nums[pares[, 2]],
    correlacion = cm[pares]
  ) |>
    arrange(desc(abs(correlacion)))
}

tasa_bogota <- mean(train$Pobre[train$cat_ubicacion == "BOGOTA"] == "Yes")
tasa_resto <- mean(train$Pobre[train$cat_ubicacion != "BOGOTA"] == "Yes")
n_bogota <- sum(train$cat_ubicacion == "BOGOTA")
n_bogota_test <- sum(test$cat_ubicacion == "BOGOTA")

acc_ingenuo <- n_no / nrow(train)
f1_ingenuo <- 0

message("Hogares train: ", nrow(train),
        ". Personas train: ", n_personas_train,
        ". Suma Nper: ", sum(train$num_personas_hogar), ".")
message("Hogares test: ", nrow(test),
        ". Personas test: ", n_personas_test,
        ". Suma Nper: ", sum(test$num_personas_hogar), ".")
message("Pobres: ", n_pobres, " (", round(100 * tasa_global, 2), "%).")
message("Columnas del bosque: ", p_exp, ".")
message("Max |SMD| pobre vs no pobre: ",
        round(max(abs(smd_clase$smd)), 3), " en ", smd_clase$variable[1], ".")
message("Covariables con |SMD| train-test > 0.1: ",
        sum(abs(smd_muestra$smd) > 0.1), ".")

# Figuras ------------------------------------------------------------------
col_pobre <- "#9C4A2B"
col_no <- "#2F6F5E"
col_barra <- "#3C5A78"
col_train <- "#3C5A78"
col_test <- "#C47B3A"

tema <- theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 9, colour = "grey25"),
    plot.caption = element_text(size = 8, colour = "grey45", hjust = 0),
    legend.position = "bottom",
    plot.margin = margin(14, 18, 12, 14)
  )

tema_tab <- ttheme_minimal(
  base_size = 8,
  core = list(
    fg_params = list(hjust = 0, x = 0.03, fontsize = 8),
    bg_params = list(fill = c("white", "grey95"))
  ),
  colhead = list(
    fg_params = list(hjust = 0, x = 0.03, fontface = "bold", fontsize = 8),
    bg_params = list(fill = "grey90")
  )
)

pag <- 0L
fun_pie <- function(nota) {
  pag <<- pag + 1L
  paste0(nota, "   ·   BDML Problem Set 2  ·  página ", pag)
}

fun_cortar <- function(texto, ancho = 86) {
  if (!nzchar(trimws(texto))) return(" ")
  palabras <- strsplit(texto, " ", fixed = TRUE)[[1]]
  lineas <- character()
  actual <- ""
  for (p in palabras) {
    candidato <- if (!nzchar(actual)) p else paste(actual, p)
    if (nchar(candidato) > ancho && nzchar(actual)) {
      lineas <- c(lineas, actual)
      actual <- p
    } else {
      actual <- candidato
    }
  }
  c(lineas, actual)
}

fun_texto <- function(titulo, lineas) {
  lineas <- unlist(lapply(lineas, fun_cortar), use.names = FALSE)
  nota <- fun_pie("Distribuciones observadas. El test no tiene la clase Pobre")
  alto <- 34
  df <- tibble(y = alto - seq_along(lineas), label = lineas)
  ggplot(df, aes(0, y, label = label)) +
    geom_text(hjust = 0, vjust = 1, size = 3.5) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, alto), expand = c(0, 0)) +
    labs(title = titulo, caption = nota) +
    theme_void(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 15, hjust = 0,
                                margin = margin(6, 0, 10, 0)),
      plot.caption = element_text(size = 8, colour = "grey45", hjust = 0),
      plot.margin = margin(16, 26, 14, 26)
    )
}

fun_tabla <- function(df, titulo, subtitulo, nota, anchos = NULL) {
  pie <- fun_pie(nota)
  tg <- tableGrob(df, rows = NULL, theme = tema_tab)
  if (!is.null(anchos)) tg$widths <- unit(anchos, "in")
  alt_tab <- convertHeight(sum(tg$heights), "in", valueOnly = TRUE)
  sub_lineas <- fun_cortar(subtitulo, 92)
  sub <- arrangeGrob(grobs = lapply(sub_lineas, function(tx) {
    textGrob(tx, x = 0.02, just = "left",
             gp = gpar(fontsize = 9, col = "grey25"))
  }), ncol = 1)
  grid.arrange(
    textGrob(titulo, x = 0.02, just = "left",
             gp = gpar(fontsize = 14, fontface = "bold")),
    sub,
    tg,
    nullGrob(),
    textGrob(pie, x = 0.02, just = "left",
             gp = gpar(fontsize = 8, col = "grey45")),
    ncol = 1,
    heights = unit(
      c(0.38, 0.28 * length(sub_lineas), alt_tab + 0.15, 1, 0.28),
      c("in", "in", "in", "null", "in")
    )
  )
}

fmt_num <- function(x, digitos = 2) sprintf(paste0("%.", digitos, "f"), x)
fmt_smd <- function(x) sprintf("%+.2f", x)

fun_tasa_factor <- function(variable, titulo) {
  d <- train |>
    group_by(nivel = .data[[variable]]) |>
    summarise(n = n(), tasa = mean(Pobre == "Yes"), .groups = "drop") |>
    arrange(tasa) |>
    mutate(eje = factor(
      paste0(nivel, "   n = ", comma(n)),
      levels = paste0(nivel, "   n = ", comma(n))
    ))
  ggplot(d, aes(eje, tasa)) +
    geom_col(fill = col_barra, width = 0.72) +
    geom_hline(yintercept = tasa_global, linetype = "dashed",
               linewidth = 0.4, colour = "grey25") +
    coord_flip() +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, max(d$tasa, tasa_global) * 1.12),
      expand = c(0, 0)
    ) +
    labs(title = titulo, x = NULL, y = "Hogares pobres") +
    tema +
    theme(axis.text.y = element_text(size = 8))
}

dir.create("02_outputs/figures", recursive = TRUE, showWarnings = FALSE)
ruta_pdf <- "02_outputs/figures/analisis_exploratorio.pdf"
cairo_pdf(ruta_pdf, width = 8.27, height = 11.69, bg = "white", onefile = TRUE)
on.exit(if (dev.cur() > 1) dev.off(), add = TRUE)

# 1. Constitución
personas_ok <- n_personas_train == sum(train$num_personas_hogar) &&
  n_personas_test == sum(test$num_personas_hogar)
print(fun_texto(
  "Cómo está constituida la base",
  c(
    "La unidad de análisis es el hogar. Pobre, la variable que se predice,",
    "también es del hogar: 1 si el hogar es pobre y 0 si no lo es. En la",
    "base de entrenamiento esos valores se guardan como Yes y No.",
    " ",
    paste0("Train: ", comma(nrow(train)), " hogares y ",
           comma(n_personas_train), " personas."),
    paste0("Test: ", comma(nrow(test)), " hogares y ",
           comma(n_personas_test), " personas. El test no trae Pobre."),
    paste0("Pobres en train: ", comma(n_pobres), " (",
           percent(tasa_global, accuracy = 0.01), "). No pobres: ",
           comma(n_no), "."),
    paste0("Valores faltantes después de la limpieza: ",
           sum(is.na(train)) + sum(is.na(test)), "."),
    paste0("Personas y hogares cuadran: ",
           if (personas_ok) "la suma de Nper es igual al número de filas de personas."
           else "la suma de Nper no coincide con las filas de personas."),
    " ",
    "Las personas no entran al modelo como filas. Cada persona se resume",
    "dentro de su hogar (edad, educación, ocupación, horas, cotización) y",
    "esa fila se une al hogar por id. Quedan fuera de la unidad de gasto",
    "el servicio doméstico, los pensionistas y los trabajadores que no",
    "comparten el presupuesto. El arriendo, la ciudad, la tenencia y el",
    "número de cuartos vienen de la base de hogares.",
    " ",
    "No hay variables de ingreso. Esas preguntas no están en el test, así",
    "que no sirven para armar el archivo que se sube a Kaggle.",
    " ",
    paste0("Bogotá tiene ", comma(n_bogota), " hogares en train (",
           percent(n_bogota / nrow(train), accuracy = 0.1),
           ") y ", n_bogota_test, " en test. La pobreza en Bogotá es ",
           percent(tasa_bogota, accuracy = 0.1), " y en el resto del train ",
           percent(tasa_resto, accuracy = 0.1), ".")
  )
))

# 2. Clase
print(fun_texto(
  "La clase está desbalanceada",
  c(
    paste0("Uno de cada cinco hogares de train es pobre (",
           percent(tasa_global, accuracy = 0.01), ")."),
    " ",
    "Un clasificador que dijera que nadie es pobre acertaría el",
    paste0(percent(acc_ingenuo, accuracy = 0.01),
           " de los hogares y tendría F1 igual a ", f1_ingenuo, "."),
    "El acierto sale alto porque la clase mayoritaria es no ser pobre.",
    "No mide si el modelo encuentra a los hogares pobres. La métrica de",
    "la competencia es el F1, que cae a cero si no hay ningún acierto",
    "sobre la clase pobre.",
    " ",
    "Por eso el punto de corte de los modelos no se deja en 0,5. Se elige",
    "el umbral que maximiza el F1 en hogares que el modelo no usó para",
    "entrenar ese fold. Esas predicciones conservan el 20% de pobres.",
    " ",
    "Repetir cuatro veces a los hogares pobres es solo un recurso de",
    "entrenamiento, dentro de cada fold. No cambia esta distribución.",
    "La validación y el test siguen con la proporción real de la muestra.",
    " ",
    paste0("Bogotá, que no está en el test, tiene una pobreza de ",
           percent(tasa_bogota, accuracy = 0.1),
           ", frente a ", percent(tasa_resto, accuracy = 0.1),
           " en el resto del train. Entrenar con Bogotá cambia un poco"),
    "la mezcla de hogares que el modelo ve, respecto del test."
  )
))

dist_clase <- tibble(
  clase = factor(c("Pobre", "No pobre"), levels = c("Pobre", "No pobre")),
  n = c(n_pobres, n_no)
) |>
  mutate(prop = n / sum(n))
print(
  ggplot(dist_clase, aes(clase, prop, fill = clase)) +
    geom_col(width = 0.55, show.legend = FALSE) +
    geom_text(aes(label = paste0(percent(prop, accuracy = 0.01),
                                 "\n", comma(n), " hogares")),
              vjust = -0.2, size = 3.6) +
    scale_fill_manual(values = c("Pobre" = col_pobre, "No pobre" = col_no)) +
    scale_y_continuous(labels = percent, limits = c(0, 1),
                       expand = expansion(mult = c(0, 0.08))) +
    labs(title = "Distribución de la pobreza en train",
         subtitle = "Cada barra es la proporción de hogares. El test no tiene esta etiqueta.",
         x = NULL, y = NULL,
         caption = fun_pie("Proporción sobre los hogares de train")) +
    tema
)

# 3. Expansión a columnas del bosque
tabla_factores <- bases_factor |>
  transmute(
    Factor = factor,
    Niveles = niveles,
    `Categoría que no entra` = base,
    `Columnas` = columnas
  )
fun_tabla(
  tabla_factores,
  "De 43 variables a las columnas del Random Forest",
  paste0(
    "Los factores sueltan la primera categoría. Las otras ",
    length(nums), " variables entran como una columna cada una. ",
    "Total: ", p_exp, " columnas. En cada corte el árbol sortea 31."
  ),
  "La categoría base sigue en los datos; el contraste la usa como referencia",
  anchos = c(2.3, 0.9, 2.4, 1.0)
)

# 4 y 5. Inventario
inv_txt <- inventario |>
  transmute(Variable = variable, Bloque = bloque, Tipo = tipo,
            Definición = definicion)
fun_tabla(
  inv_txt[1:21, ],
  "Predictores de los cuatro primeros bloques",
  "Vivienda, demografía, empleo y formalidad. Una fila de esta tabla es una columna de la base de hogares.",
  "Definiciones tomadas de la limpieza de hogares y del colapso de personas",
  anchos = c(2.15, 1.05, 0.95, 3.15)
)
fun_tabla(
  inv_txt[22:43, ],
  "Predictores del bloque extendido",
  "Se agregan a los 21 anteriores. El Random Forest, AdaBoost y Naive Bayes usan los 43.",
  "UG es la unidad de gasto: quienes comparten el presupuesto del hogar",
  anchos = c(2.35, 1.05, 0.9, 3.0)
)

# 6. Indicadores
tabla_bin <- indicadores |>
  transmute(
    Indicador = variable,
    `Hogares en 1` = comma(n_uno),
    `Pobreza si es 0` = percent(tasa_0, accuracy = 0.1),
    `Pobreza si es 1` = percent(tasa_1, accuracy = 0.1),
    `Diferencia, pp` = sprintf("%+.1f", dif_pp)
  )
fun_tabla(
  tabla_bin,
  "Indicadores: pobreza cuando valen 0 y cuando valen 1",
  paste0("La pobreza de todo el train es ",
         percent(tasa_global, accuracy = 0.1),
         ". La diferencia está en puntos porcentuales (vale 1 menos vale 0)."),
  "Un indicador puede separar clases aunque solo tome dos valores",
  anchos = c(2.3, 1.2, 1.3, 1.3, 1.2)
)

# 7. SMD de clase
smd_plot <- smd_clase |>
  mutate(
    sentido = if_else(smd >= 0, "Más alta en pobres", "Más alta en no pobres"),
    variable = reorder(variable, smd)
  )
print(
  ggplot(smd_plot, aes(variable, smd, fill = sentido)) +
    geom_col(width = 0.72) +
    geom_hline(yintercept = 0, linewidth = 0.3) +
    coord_flip() +
    scale_fill_manual(values = c(
      "Más alta en pobres" = col_pobre,
      "Más alta en no pobres" = col_no
    )) +
    labs(title = "Qué tan distinta es cada variable entre pobres y no pobres",
         subtitle = "SMD = (media en pobres − media en no pobres) / desviación pooled. Importa el tamaño, no el p-valor.",
         x = NULL, y = "Diferencia estandarizada",
         fill = NULL,
         caption = fun_pie("Solo variables numéricas e indicadores. Los factores van en las tasas por categoría")) +
    tema +
    theme(axis.text.y = element_text(size = 7),
          legend.position = "bottom")
)

tabla_smd <- function(df) {
  df |>
    transmute(
      Variable = variable,
      `Mediana pobre` = fmt_num(mediana_pobre),
      `Mediana no pobre` = fmt_num(mediana_no),
      `% en cero` = percent(pct_cero, accuracy = 0.1),
      SMD = fmt_smd(smd)
    )
}
fun_tabla(
  tabla_smd(slice_head(smd_clase, n = 19)),
  "Distribución de las variables que más separan",
  "Ordenadas por el valor absoluto del SMD. La mediana es de cada clase. El porcentaje en cero es de todo el train.",
  "Un SMD positivo quiere decir que la media es más alta en los hogares pobres",
  anchos = c(2.5, 1.2, 1.45, 0.95, 0.7)
)
fun_tabla(
  tabla_smd(slice_tail(smd_clase, n = nrow(smd_clase) - 19)),
  "Distribución del resto de variables numéricas",
  "Las de esta página separan menos las dos clases. Siguen entrando al modelo.",
  "El porcentaje en cero muestra las variables con una masa grande en el cero",
  anchos = c(2.5, 1.2, 1.45, 0.95, 0.7)
)

# 8. Densidades
top_vars <- smd_clase |>
  filter(!variable %in% bins) |>
  slice_head(n = 6) |>
  mutate(etiqueta = paste0(variable, "\nSMD ", fmt_smd(smd)))
dens <- purrr::map_dfr(seq_len(nrow(top_vars)), function(i) {
  v <- top_vars$variable[i]
  tibble(
    etiqueta = top_vars$etiqueta[i],
    clase = if_else(train$Pobre == "Yes", "Pobre", "No pobre"),
    valor = train[[v]]
  )
})
dens$etiqueta <- factor(dens$etiqueta, levels = top_vars$etiqueta)
dens$clase <- factor(dens$clase, levels = c("Pobre", "No pobre"))
print(
  ggplot(dens, aes(valor, fill = clase)) +
    geom_density(alpha = 0.45, colour = NA, linewidth = 0) +
    facet_wrap(~etiqueta, scales = "free", ncol = 2) +
    scale_fill_manual(values = c("Pobre" = col_pobre, "No pobre" = col_no)) +
    labs(title = "Distribución de las seis variables continuas que más separan",
         subtitle = "Densidad dentro de cada clase. Las escalas horizontales no se comparten: las unidades son distintas.",
         x = NULL, y = "Densidad", fill = NULL,
         caption = fun_pie("Se excluyen los indicadores 0/1, que están en la tabla anterior")) +
    tema
)

# 9. Tasas por categoría
print(
  fun_tasa_factor("cat_tenencia", "Pobreza según la tenencia de la vivienda") +
    labs(subtitle = paste0(
      "Línea punteada: ", percent(tasa_global, accuracy = 0.1),
      " de los hogares de train son pobres."
    ), caption = fun_pie("Train. El tamaño de la categoría va en el eje"))
)
print(
  fun_tasa_factor("cat_educ_jefe", "Pobreza según la educación del jefe") +
    labs(subtitle = "Ninguno incluye a quien no responde y a los menores de 3 años, por la regla de limpieza.",
         caption = fun_pie("Train"))
)
print(
  fun_tasa_factor("cat_maxEduc", "Pobreza según la educación más alta del hogar") +
    labs(subtitle = "Es el máximo entre los miembros, no solo el jefe.",
         caption = fun_pie("Train"))
)
print(
  fun_tasa_factor("cat_pos_jefe", "Pobreza según la posición ocupacional del jefe") +
    labs(subtitle = "Quien no trabaja queda en No ocupado. Sin remuneración junta dos códigos poco frecuentes.",
         caption = fun_pie("Train"))
)
print(
  fun_tasa_factor("cat_tam_empresa_jefe", "Pobreza según el tamaño de la empresa del jefe") +
    labs(subtitle = "No ocupado es el jefe que no está trabajando. El resto es el número de personas de la empresa.",
         caption = fun_pie("Train"))
)
print(
  fun_tasa_factor("cat_ubicacion", "Pobreza según la ciudad o la zona") +
    labs(subtitle = "Bogotá entra en este gráfico y no está en el test. La línea es la pobreza de todo el train.",
         caption = fun_pie("Train. Rural y resto urbano son categorías del Dominio, no ciudades"))
)

# 10. Balance train-test
bal <- smd_muestra |>
  mutate(variable = reorder(variable, smd))
print(
  ggplot(bal, aes(variable, smd)) +
    geom_col(fill = col_barra, width = 0.72) +
    geom_hline(yintercept = c(-0.1, 0.1), linetype = "dashed",
               colour = col_pobre, linewidth = 0.3) +
    coord_flip() +
    labs(title = "Train y test, ¿traen los mismos predictores?",
         subtitle = paste0(
           "SMD = (media train − media test) / desviación pooled. Líneas en ±0,1. Variables que pasan de 0,1: ",
           sum(abs(smd_muestra$smd) > 0.1), "."
         ),
         x = NULL, y = "Diferencia estandarizada train − test",
         caption = fun_pie("No usa Pobre: el test no la tiene. Un SMD chico no es una prueba de igualdad")) +
    tema +
    theme(axis.text.y = element_text(size = 7))
)

comp_ubi <- bind_rows(
  train |> count(cat_ubicacion, .drop = FALSE) |>
    mutate(muestra = "Train", prop = n / sum(n)),
  test |> count(cat_ubicacion, .drop = FALSE) |>
    mutate(muestra = "Test", prop = n / sum(n))
)
orden_ubi <- comp_ubi |>
  filter(muestra == "Train") |>
  arrange(prop) |>
  pull(cat_ubicacion) |>
  as.character()
comp_ubi <- comp_ubi |>
  mutate(
    cat_ubicacion = factor(as.character(cat_ubicacion), levels = orden_ubi),
    muestra = factor(muestra, levels = c("Train", "Test"))
  )
print(
  ggplot(comp_ubi, aes(cat_ubicacion, prop, fill = muestra)) +
    geom_col(position = "dodge", width = 0.75) +
    coord_flip() +
    scale_y_continuous(labels = percent_format(accuracy = 1),
                       expand = expansion(mult = c(0, 0.05))) +
    scale_fill_manual(values = c("Train" = col_train, "Test" = col_test)) +
    labs(title = "Dónde están los hogares",
         subtitle = "Proporción de hogares por Dominio. Bogotá queda en cero en el test porque el factor existe, pero no hay hogares.",
         x = NULL, y = "Proporción de hogares", fill = NULL,
         caption = fun_pie("Los niveles del factor son los mismos en train y en test")) +
    tema +
    theme(axis.text.y = element_text(size = 7))
)

tam <- bind_rows(
  train |> transmute(muestra = "Train",
                     `Personas del hogar` = num_personas_hogar,
                     `Personas de la UG` = num_personas_ug,
                     Cuartos = num_cuartos),
  test |> transmute(muestra = "Test",
                    `Personas del hogar` = num_personas_hogar,
                    `Personas de la UG` = num_personas_ug,
                    Cuartos = num_cuartos)
) |>
  pivot_longer(-muestra, names_to = "variable", values_to = "valor") |>
  mutate(muestra = factor(muestra, levels = c("Train", "Test")))
print(
  ggplot(tam, aes(valor, colour = muestra)) +
    geom_density(linewidth = 0.7) +
    facet_wrap(~variable, scales = "free", ncol = 1) +
    scale_colour_manual(values = c("Train" = col_train, "Test" = col_test)) +
    labs(title = "Tamaño del hogar en train y en test",
         subtitle = "Densidad. Personas del hogar es Nper. La unidad de gasto excluye servicio doméstico, pensionistas y trabajadores.",
         x = NULL, y = "Densidad", colour = NULL,
         caption = fun_pie("Misma definición en las dos muestras")) +
    tema
)

# 11. Redundancia de construcción
if (nrow(pares_altos) == 0) {
  tabla_cor <- tibble(A = "Ningún par", B = "", r = "")
} else {
  tabla_cor <- pares_altos |>
    slice_head(n = 18) |>
    transmute(A = variable_a, B = variable_b,
              r = fmt_num(correlacion))
}
fun_tabla(
  tabla_cor,
  "Variables numéricas que se parecen mucho",
  "Pares con correlación de Pearson mayor o igual a 0,75 en valor absoluto, calculada en train.",
  "Varias son la misma medida en niveles y en proporción",
  anchos = c(2.6, 2.6, 0.8)
)

tabla_id <- identidades |>
  transmute(
    Relación = relacion,
    `Error máximo` = formatC(error_maximo, format = "e", digits = 1)
  )
fun_tabla(
  tabla_id,
  "Relaciones que se cumplen por construcción",
  "El error máximo es la mayor diferencia absoluta en train. Cero, o ruido de redondeo, quiere decir que la igualdad se cumple.",
  "No son 43 piezas de información independientes",
  anchos = c(6.2, 1.1)
)

top5 <- slice_head(smd_clase, n = 5)
lineas_top <- paste0(
  "• ", top5$variable, ": SMD ", fmt_smd(top5$smd),
  if_else(top5$smd >= 0, ", más alta en pobres.", ", más baja en pobres.")
)
n_fuera <- sum(abs(smd_muestra$smd) > 0.1)
print(fun_texto(
  "Qué sostiene el análisis, y qué no",
  c(
    paste0("La muestra de entrenamiento tiene ", comma(nrow(train)),
           " hogares, sin faltantes después de las reglas de limpieza."),
    paste0("La clase pobre es el ", percent(tasa_global, accuracy = 0.01),
           ". El acierto del clasificador ingenuo es ",
           percent(acc_ingenuo, accuracy = 0.01), " y su F1 es 0."),
    " ",
    "Las cinco variables numéricas que más separan pobres de no pobres:",
    lineas_top[1], lineas_top[2], lineas_top[3], lineas_top[4], lineas_top[5],
    " ",
    if (n_fuera == 0) {
      "Ninguna variable numérica pasa de 0,1 en la diferencia estandarizada entre train y test."
    } else {
      paste0(n_fuera, " variables numéricas pasan de 0,1 en la diferencia estandarizada entre train y test.")
    },
    paste0("El desajuste grueso de composición es Bogotá: ", comma(n_bogota),
           " hogares en train y ninguno en test."),
    " ",
    "Varias columnas son transformaciones de otras (personas por cuarto,",
    "proporción de ocupados, tasas de dependencia, inactividad y desempleo).",
    "El bosque puede usarlas juntas. No hay que leerlas como 43 causas",
    "distintas de la pobreza.",
    " ",
    "Este informe describe la muestra. No dice que una variable cause",
    "la pobreza, ni cómo le irá al modelo en el tablero de Kaggle:",
    "el test no trae la clase verdadera."
  )
))

dev.off()
message("PDF guardado en ", ruta_pdf, " (", pag, " páginas).")
