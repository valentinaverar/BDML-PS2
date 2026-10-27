##########################################################
# 13_compare_algorithms.R
#
# Objetivo: Comparar los algoritmos y separar cuánto del desempeño se debe
# al algoritmo y cuánto a las variables, al ajuste de hiperparámetros y al
# tratamiento del desbalance de clases.
#
# Etapa del Problem Set: Etapa 2: Training and Comparing Algorithms.
# Insumos:  02_outputs/tables/registro_modelos.csv (F1 de validación cruzada
#           de cada variante, todas con las mismas particiones),
#           00_data/kaggle_scores.csv (puntaje público), 03_temp/train_clean.rds
#           y predictores.rds, y las predicciones fuera de muestra de
#           LightGBM, Random Forest y Elastic Net con X_5 (03_temp/).
# Salidas:  02_outputs/figures/ (figuras del deck compare_equipo_03) y
#           02_outputs/tables/ (tablas en .csv y en LaTeX para las slides).
##########################################################
#
# Las comparaciones usan el F1 de validación cruzada y no el puntaje de
# Kaggle: todas las variantes se evaluaron con los mismos hogares (03_temp/
# cv_folds.rds), mientras que en Kaggle no se subieron todas.
#
#   0. Datos: la línea base y por qué los árboles deberían ganar.
#   1. Resumen de los siete algoritmos requeridos y, con las mismas
#      variables (X_5_extendido), el efecto del algoritmo.
#   2. Mismo algoritmo, más variables (X_5, X_6 y X_7): efecto de las
#      variables.
#   3. Ajuste de hiperparámetros y rebalanceo de clases, con el algoritmo y
#      las variables fijos.
#   3b. Comparación a fondo del boosting con sus dos rivales más cercanos
#       (Random Forest y Elastic Net): umbral, errores, grupos de hogares y
#       variables más importantes.
#   4. Respaldo: el F1 de validación cruzada frente al puntaje de Kaggle.
##########################################################

# Preparación -------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, knitr)

registro <- read.csv("02_outputs/tables/registro_modelos.csv")
kaggle   <- read.csv("00_data/kaggle_scores.csv") |>
  select(archivo, puntaje_publico)

# Colores: uno para destacar el boosting, otro para el segundo algoritmo de
# una comparación, y gris para el resto. El texto va siempre en gris
# oscuro, no en el color de la serie.
color_1     <- "#2a78d6"
color_2     <- "#eb6834"
color_resto <- "#a3a29b"
texto_1     <- "#0b0b0b"
texto_2     <- "#52514e"

tema <- theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        axis.text = element_text(color = texto_2),
        axis.title = element_text(color = texto_2),
        # Título, subtítulo y nota centrados sobre toda la figura, con
        # espacio entre ellos y el gráfico.
        plot.title = element_text(color = texto_1, face = "bold", hjust = 0.5,
                                  margin = margin(b = 4)),
        plot.subtitle = element_text(color = texto_2, hjust = 0.5,
                                     margin = margin(b = 14)),
        plot.caption = element_text(color = texto_2, hjust = 0.5,
                                    margin = margin(t = 14)),
        plot.margin = margin(12, 14, 10, 14),
        plot.title.position = "plot",
        plot.caption.position = "plot",
        legend.position = "none")

# Números con coma decimal, para las slides en español.
fun_coma <- function(.x, .digits = 3) {
  ifelse(is.na(.x), "--",
         formatC(.x, format = "f", digits = .digits, decimal.mark = ","))
}

dir.create("02_outputs/figures", recursive = TRUE, showWarnings = FALSE)

# Variantes ---------------------------------------------------------------
# Nombre corto del algoritmo y tipo de rebalanceo de cada variante. Los dos
# Random Forest se distinguen por cómo se eligieron sus hiperparámetros:
# fijos (RF_trees_...) o con una grilla en validación cruzada (rf_...).
variantes <- registro |>
  left_join(kaggle, by = "archivo") |>
  mutate(
    algoritmo_corto = case_when(
      algoritmo == "Boosting (XGBoost)"  ~ "XGBoost",
      algoritmo == "Boosting (LightGBM)" ~ "LightGBM",
      startsWith(archivo, "RF_trees") ~ "Random Forest (fijo)",
      startsWith(archivo, "rf_") ~ "Random Forest (grilla CV)",
      TRUE ~ algoritmo),
    rebalanceo = case_when(
      grepl("x4", hiperparametros) ~ "Pobres x4",
      grepl("rebalanceo = up", hiperparametros) ~ "Upsampling",
      grepl("rebalanceo = down", hiperparametros) ~ "Downsampling",
      TRUE ~ "Sin rebalanceo"),
    familia = if_else(algoritmo_corto %in% c("XGBoost", "LightGBM"),
                      "Boosting de gradiente", "Otros")
  )

# 0. Datos ---------------------------------------------------------------------
train <- readRDS("03_temp/train_clean.rds")
predictores_x5 <- readRDS("03_temp/predictores.rds")$X_5_extendido

# Línea base: clasificar a todos los hogares como no pobres. Con 20% de
# pobres acierta cerca del 80%, pero no encuentra ningún hogar pobre: su F1
# es 0. Por eso la métrica es el F1 y el umbral se elige para maximizarlo.
prop_pobres <- mean(train$Pobre == "Yes")
linea_base <- tibble(
  `Métrica` = c("Accuracy", "Recall (pobres encontrados)", "F1"),
  `Todos no pobres` = c(paste0(fun_coma(100 * (1 - prop_pobres), 1), "%"),
                        "0%", "0")
)
writeLines(kable(linea_base, format = "latex", booktabs = TRUE, linesep = "",
                 align = c("l", "r")),
           "02_outputs/tables/linea_base.tex")

# Interacción: el efecto de un ocupado más depende del tamaño del hogar.
# Un modelo lineal en estas dos variables suma efectos que no cambian entre
# celdas; los árboles pueden aprender que el efecto cambia.
pobreza_cruzada <- train |>
  mutate(ocupados = factor(pmin(num_ocupados, 3), levels = 0:3,
                           labels = c("0", "1", "2", "3 o más")),
         personas = factor(pmin(num_personas_ug, 6), levels = 1:6,
                           labels = c("1", "2", "3", "4", "5", "6 o más"))) |>
  group_by(personas, ocupados) |>
  summarize(pobreza = mean(Pobre == "Yes"), hogares = n(), .groups = "drop") |>
  # Celdas con muy pocos hogares no se muestran: su porcentaje es inestable.
  filter(hogares >= 100)

grafico_datos <- pobreza_cruzada |>
  ggplot(aes(x = ocupados, y = personas, fill = pobreza)) +
  geom_tile(color = "white", linewidth = 1.5) +
  geom_text(aes(label = paste0(round(100 * pobreza), "%"),
                color = pobreza > 0.45), size = 3.6, fontface = "bold") +
  scale_fill_gradient(low = "#cde2fb", high = "#104281",
                      labels = function(x) paste0(round(100 * x), "%"),
                      name = "Hogares\npobres") +
  scale_color_manual(values = c("TRUE" = "white", "FALSE" = texto_1),
                     guide = "none") +
  labs(title = "Un ocupado más pesa distinto según el tamaño del hogar",
       subtitle = "Porcentaje de hogares pobres según personas y ocupados de la unidad de gasto",
       x = "Ocupados en el hogar", y = "Personas en el hogar",
       caption = "Train (164.960 hogares). Se omiten celdas con menos de 100 hogares.") +
  tema +
  theme(legend.position = "right",
        legend.title = element_text(color = texto_2, size = 9),
        legend.text = element_text(color = texto_2, size = 8),
        panel.grid = element_blank(),
  )
ggsave("02_outputs/figures/pobreza_ocupados_personas.pdf", grafico_datos,
       width = 7, height = 4.6)

# No linealidad: la pobreza cae con el arriendo por persona (un indicador
# del valor de la vivienda), pero no en línea recta. Se excluyen los hogares
# que no saben o no informan el arriendo (bin_arriendo_nd = 1).
pobreza_arriendo <- train |>
  filter(bin_arriendo_nd == 0) |>
  mutate(decil = ntile(num_log_arriendo_pc, 10)) |>
  group_by(decil) |>
  summarize(pobreza = mean(Pobre == "Yes"), .groups = "drop")

grafico_arriendo <- pobreza_arriendo |>
  ggplot(aes(x = factor(decil), y = pobreza)) +
  geom_col(fill = color_1, width = 0.72) +
  geom_text(aes(label = paste0(round(100 * pobreza), "%")), color = texto_1,
            size = 3.3, vjust = -0.5) +
  scale_y_continuous(labels = function(x) paste0(round(100 * x), "%"),
                     limits = c(0, max(pobreza_arriendo$pobreza) * 1.12),
                     expand = expansion(mult = c(0, 0))) +
  labs(title = "La pobreza cae con el arriendo, pero no en línea recta",
       subtitle = "Porcentaje de hogares pobres por decil de arriendo por persona",
       x = "Decil de arriendo por persona (1 = el más bajo)",
       y = "Hogares pobres",
       caption = "Train, sin los hogares que no saben o no informan el arriendo.") +
  tema +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(color = "#eeede8"),
  )
ggsave("02_outputs/figures/pobreza_arriendo_deciles.pdf", grafico_arriendo,
       width = 7, height = 4.6)

# 1. Resumen de los siete algoritmos requeridos ----------------------------------
# Para cada algoritmo de la lista del Problem Set, su mejor F1 de validación
# cruzada (con cualquier conjunto) y su mejor puntaje público en Kaggle.
# XGBoost, LightGBM y AdaBoost son implementaciones de boosting.
fun_categoria <- function(.algoritmo) {
  case_when(.algoritmo %in% c("Boosting (XGBoost)", "Boosting (LightGBM)",
                              "AdaBoost", "XGBoost", "LightGBM") ~ "Boosting",
            TRUE ~ .algoritmo)
}
etiqueta_conjunto <- c(X_1_vivienda = "X_1", X_2_demografia = "X_2",
                       X_3_empleo = "X_3", X_4_completo = "X_4",
                       X_5_extendido = "X_5", X_6_detallado = "X_6",
                       X_7_ampliado = "X_7", LPM_10 = "10 variables")
orden_algoritmos <- c("LPM", "Logit", "Elastic Net", "CART",
                      "Random Forest", "Boosting", "Naive Bayes")

mejor_cv <- registro |>
  mutate(categoria = fun_categoria(algoritmo)) |>
  group_by(categoria) |>
  # El conjunto se toma antes de resumir f1_cv, porque summarize() usa el
  # valor ya resumido en las columnas siguientes.
  summarize(variantes = n(),
            conjunto = etiqueta_conjunto[conjunto[which.max(f1_cv)]],
            f1_cv = max(f1_cv),
            .groups = "drop")
mejor_kaggle <- read.csv("00_data/kaggle_scores.csv") |>
  mutate(categoria = fun_categoria(algoritmo)) |>
  group_by(categoria) |>
  summarize(kaggle = max(puntaje_publico), .groups = "drop")

resumen_algoritmos <- tibble(categoria = orden_algoritmos) |>
  left_join(mejor_cv, by = "categoria") |>
  left_join(mejor_kaggle, by = "categoria")
write.csv(resumen_algoritmos, "02_outputs/tables/resumen_algoritmos.csv",
          row.names = FALSE)
writeLines(kable(resumen_algoritmos |>
                   transmute(Algoritmo = categoria,
                             `Variantes evaluadas` = variantes,
                             `Mejor F1 CV` = fun_coma(f1_cv),
                             Conjunto = conjunto,
                             `Mejor puntaje Kaggle` = fun_coma(kaggle, 2)),
                 format = "latex", booktabs = TRUE, linesep = "",
                 align = c("l", "r", "r", "l", "r")),
           "02_outputs/tables/resumen_algoritmos.tex")

# 1b. Mismas variables, distintos algoritmos -------------------------------------
# Mejor variante sin rebalanceo de cada algoritmo con X_5_extendido. Los dos
# Random Forest se tratan como un algoritmo: se conserva el de mayor F1.
comparacion_x5 <- variantes |>
  filter(conjunto == "X_5_extendido", rebalanceo == "Sin rebalanceo") |>
  mutate(algoritmo_base = if_else(startsWith(algoritmo_corto, "Random Forest"),
                                  "Random Forest", algoritmo_corto)) |>
  group_by(algoritmo_base) |>
  slice_max(f1_cv, n = 1, with_ties = FALSE) |>
  ungroup() |>
  arrange(desc(f1_cv))

# Figura: F1 de cada algoritmo, agrupados por familia (cómo aprende cada
# uno) y ordenados de mayor a menor. El boosting va en azul.
datos_x5 <- comparacion_x5 |>
  mutate(
    familia_grafico = case_when(
      algoritmo_base %in% c("XGBoost", "LightGBM") ~ "Boosting",
      algoritmo_base %in% c("Random Forest", "AdaBoost") ~ "Ensambles\nde árboles",
      algoritmo_base %in% c("Logit", "Elastic Net", "LPM") ~ "Modelos\nlineales",
      algoritmo_base == "CART" ~ "Un solo\nárbol",
      TRUE ~ "Otros"),
    familia_grafico = fct_reorder(familia_grafico, f1_cv, .fun = max,
                                  .desc = TRUE),
    algoritmo_base = fct_reorder(algoritmo_base, f1_cv),
    destacado = familia == "Boosting de gradiente"
  )

grafico_x5 <- datos_x5 |>
  ggplot(aes(x = f1_cv, y = algoritmo_base, color = destacado)) +
  geom_point(size = 4) +
  geom_text(aes(label = fun_coma(f1_cv)), color = texto_1, size = 3.3,
            hjust = -0.45) +
  facet_grid(familia_grafico ~ ., scales = "free_y", space = "free_y",
             switch = "y") +
  scale_color_manual(values = c("TRUE" = color_1, "FALSE" = color_resto)) +
  scale_x_continuous(limits = c(0.635, 0.745),
                     breaks = seq(0.64, 0.74, by = 0.02),
                     labels = function(x) fun_coma(x, 2)) +
  labs(title = "Con las mismas variables, el boosting obtiene el mayor F1",
       subtitle = "F1 de validación cruzada con las 43 variables de X_5_extendido",
       x = "F1 de validación cruzada", y = NULL,
       caption = "Mejor variante sin rebalanceo de cada algoritmo; mismos 5 folds para todos.") +
  tema +
  theme(        strip.placement = "outside",
                strip.text.y.left = element_text(angle = 0, hjust = 1,
                                                 color = texto_2, face = "bold",
                                                 size = 9),
                panel.spacing.y = unit(6, "pt"),
                panel.grid.major.x = element_line(color = "#eeede8"),
                axis.text.y = element_text(color = texto_1, size = 10))
ggsave("02_outputs/figures/comparacion_algoritmos_x5.pdf", grafico_x5,
       width = 8, height = 4.8)

tabla_x5 <- comparacion_x5 |>
  transmute(Algoritmo = algoritmo_base,
            `F1 CV` = fun_coma(f1_cv),
            `Precisión` = fun_coma(precision_cv),
            Recall = fun_coma(recall_cv),
            AUC = fun_coma(auc_cv),
            Umbral = fun_coma(umbral, 2),
            Kaggle = fun_coma(puntaje_publico, 2))
write.csv(comparacion_x5, "02_outputs/tables/comparacion_algoritmos_x5.csv",
          row.names = FALSE)
writeLines(kable(tabla_x5, format = "latex", booktabs = TRUE, linesep = "",
                 align = c("l", rep("r", 6))),
           "02_outputs/tables/comparacion_algoritmos_x5.tex")

# 2. Mismo algoritmo, más variables ------------------------------------------
# XGBoost y LightGBM con cada conjunto, con la misma configuración en todos.
efecto_variables <- variantes |>
  filter(algoritmo_corto %in% c("XGBoost", "LightGBM"),
         conjunto %in% c("X_5_extendido", "X_6_detallado", "X_7_ampliado")) |>
  mutate(conjunto = factor(conjunto,
                           levels = c("X_5_extendido", "X_6_detallado",
                                      "X_7_ampliado"),
                           labels = c("X_5\n(43)", "X_6\n(148)", "X_7\n(1.296)")))

grafico_variables <- efecto_variables |>
  ggplot(aes(x = conjunto, y = f1_cv, color = algoritmo_corto,
             group = algoritmo_corto)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 3.5) +
  # Etiquetas de valor: LightGBM arriba de su punto y XGBoost abajo, para
  # que no se crucen en X_5, donde casi empatan.
  geom_text(aes(label = fun_coma(f1_cv),
                vjust = if_else(algoritmo_corto == "LightGBM", -1.1, 2)),
            color = texto_1, size = 3.2) +
  geom_text(data = efecto_variables |> filter(conjunto == "X_7\n(1.296)"),
            aes(label = algoritmo_corto), color = texto_1, size = 3.4,
            hjust = -0.3) +
  scale_color_manual(values = c("XGBoost" = color_1, "LightGBM" = color_2)) +
  scale_y_continuous(limits = c(0.72, 0.775),
                     labels = function(x) fun_coma(x, 2)) +
  scale_x_discrete(expand = expansion(add = c(0.3, 0.8))) +
  labs(title = "Con boosting, más variables suman 3 a 4 puntos de F1",
       subtitle = "F1 de validación cruzada según el conjunto de variables (número de variables)",
       x = NULL, y = "F1 de validación cruzada",
       caption = "Misma configuración de cada algoritmo en todos los conjuntos.") +
  tema +
  theme(panel.grid.major.y = element_line(color = "#e8e7e2"),
        panel.grid.major.x = element_blank())
ggsave("02_outputs/figures/efecto_variables_boosting.pdf", grafico_variables,
       width = 7, height = 4.2)

tabla_variables <- efecto_variables |>
  arrange(conjunto) |>
  select(algoritmo_corto, conjunto, f1_cv) |>
  mutate(conjunto = gsub("\n", " ", conjunto), f1_cv = fun_coma(f1_cv)) |>
  pivot_wider(names_from = conjunto, values_from = f1_cv,
              values_fill = "--") |>
  rename(Algoritmo = algoritmo_corto)
writeLines(kable(tabla_variables, format = "latex", booktabs = TRUE, linesep = ""),
           "02_outputs/tables/efecto_variables_boosting.tex")

# 3. Ajuste de hiperparámetros y rebalanceo -----------------------------------
# Ajuste: el mismo algoritmo y las mismas variables, con hiperparámetros
# fijos o elegidos en validación cruzada (Random Forest), y un logit con y
# sin penalización (Elastic Net).
ajuste <- variantes |>
  filter(conjunto == "X_5_extendido", rebalanceo == "Sin rebalanceo",
         algoritmo_corto %in% c("Random Forest (fijo)",
                                "Random Forest (grilla CV)",
                                "Logit", "Elastic Net")) |>
  transmute(Variante = algoritmo_corto,
            `Hiperparámetros` = case_when(
              algoritmo_corto == "Logit" ~ "Sin penalización",
              algoritmo_corto == "Elastic Net" ~
                paste0("alpha ", fun_coma(as.numeric(str_extract(hiperparametros, "(?<=alpha = )[0-9.e-]+")), 2),
                       "; lambda ", fun_coma(as.numeric(str_extract(hiperparametros, "(?<=lambda = )[0-9.e-]+")), 4),
                       " (elegidos por CV)"),
              algoritmo_corto == "Random Forest (fijo)" ~
                "mtry 6, nodo mínimo 10, 300 árboles (fijos)",
              TRUE ~ "mtry 31, nodo mínimo 20, 300 árboles (elegidos por CV)"),
            `F1 CV` = fun_coma(f1_cv)) |>
  arrange(Variante)
writeLines(kable(ajuste, format = "latex", booktabs = TRUE, linesep = ""),
           "02_outputs/tables/efecto_ajuste.tex")

# Rebalanceo: cada algoritmo con X_5, sin y con rebalanceo. Con el umbral
# elegido por F1, rebalancear casi no cambia el resultado: mover el umbral
# ya corrige el desbalance.
rebalanceo <- variantes |>
  filter(conjunto == "X_5_extendido",
         algoritmo_corto %in% c("CART", "Random Forest (grilla CV)",
                                "AdaBoost", "Naive Bayes")) |>
  transmute(Algoritmo = algoritmo_corto, Rebalanceo = rebalanceo,
            `F1 CV` = fun_coma(f1_cv), Umbral = fun_coma(umbral, 2)) |>
  arrange(Algoritmo, Rebalanceo != "Sin rebalanceo")
writeLines(kable(rebalanceo, format = "latex", booktabs = TRUE, linesep = ""),
           "02_outputs/tables/efecto_rebalanceo.tex")

# 3b. Comparación a fondo: el boosting frente a sus dos rivales más cercanos ----
# Boosting (LightGBM), Random Forest y Elastic Net, los tres con X_5_extendido
# y los mismos folds. Se usan las predicciones fuera de muestra que cada
# script de modelo guarda en 03_temp/ con el nombre de su envío.
color_3 <- "#1baf7a"
colores_top3 <- c("LightGBM" = color_1, "Random Forest" = color_2,
                  "Elastic Net" = color_3)

top3 <- comparacion_x5 |>
  filter(algoritmo_base %in% names(colores_top3)) |>
  select(algoritmo_base, archivo, umbral)
rutas_top3 <- file.path("03_temp", paste0(top3$archivo, ".rds"))
if (!all(file.exists(rutas_top3))) {
  stop("Faltan predicciones fuera de muestra en 03_temp/: ",
       paste(top3$archivo[!file.exists(rutas_top3)], collapse = ", "),
       ". Corra los scripts de esos modelos.", call. = FALSE)
}
resultados_top3 <- setNames(lapply(rutas_top3, readRDS), top3$algoritmo_base)

# Probabilidades fuera de muestra, en el orden de las filas de train.
prob_top3 <- map_dfc(resultados_top3, function(.r) {
  oof <- .r$oof |> arrange(rowIndex)
  stopifnot(nrow(oof) == nrow(train), all(oof$rowIndex == seq_len(nrow(train))),
            all(as.character(oof$obs) == as.character(train$Pobre)))
  oof$Yes
})
pobre <- train$Pobre == "Yes"
umbrales_top3 <- setNames(top3$umbral, top3$algoritmo_base)

fun_f1 <- function(.y, .pred) {
  tp <- sum(.y & .pred)
  2 * tp / (2 * tp + sum(!.y & .pred) + sum(.y & !.pred))
}

# (a) F1 según el umbral: si la ventaja del boosting depende del umbral
# elegido o se mantiene en todo el rango.
curvas <- expand_grid(umbral = seq(0.10, 0.70, by = 0.01),
                      modelo = names(colores_top3)) |>
  mutate(F1 = map2_dbl(umbral, modelo,
                       \(u, m) fun_f1(pobre, prob_top3[[m]] >= u)),
         modelo = factor(modelo, levels = names(colores_top3)))
elegidos <- tibble(modelo = factor(names(umbrales_top3), levels = names(colores_top3)),
                   umbral = umbrales_top3) |>
  mutate(F1 = map2_dbl(umbral, as.character(modelo),
                       \(u, m) fun_f1(pobre, prob_top3[[m]] >= u)))

grafico_curvas <- curvas |>
  ggplot(aes(x = umbral, y = F1, color = modelo)) +
  geom_line(linewidth = 0.8) +
  geom_point(data = elegidos, size = 3.2) +
  geom_text(data = curvas |> filter(umbral == max(umbral)),
            aes(label = modelo), color = texto_1, size = 3.3, hjust = -0.1) +
  scale_color_manual(values = colores_top3) +
  scale_x_continuous(labels = function(x) fun_coma(x, 1),
                     expand = expansion(add = c(0.01, 0.13))) +
  scale_y_continuous(labels = function(x) fun_coma(x, 2)) +
  labs(title = "El boosting tiene el mayor F1 con cualquier umbral",
       subtitle = "F1 de validación cruzada según el umbral; el punto marca el umbral elegido",
       x = "Umbral de clasificación", y = "F1 de validación cruzada",
       caption = "Los tres modelos con las 43 variables de X_5_extendido y los mismos 5 folds.") +
  tema +
  theme(panel.grid.major.y = element_line(color = "#eeede8"))
ggsave("02_outputs/figures/f1_umbral_top3.pdf", grafico_curvas,
       width = 7, height = 4.4)

# (b) Errores con el umbral elegido: falsos positivos (hogares no pobres que
# recibirían el beneficio, leakage) y falsos negativos (hogares pobres que
# quedarían por fuera, undercoverage).
errores_top3 <- map_dfr(names(colores_top3), function(.m) {
  pred <- prob_top3[[.m]] >= umbrales_top3[[.m]]
  tibble(Modelo = .m,
         Umbral = fun_coma(umbrales_top3[[.m]], 2),
         `Precisión` = fun_coma(sum(pobre & pred) / sum(pred)),
         Recall = fun_coma(sum(pobre & pred) / sum(pobre)),
         F1 = fun_coma(fun_f1(pobre, pred)),
         `Falsos positivos` = format(sum(!pobre & pred), big.mark = ".", decimal.mark = ","),
         `Falsos negativos` = format(sum(pobre & !pred), big.mark = ".", decimal.mark = ","))
})
writeLines(kable(errores_top3, format = "latex", booktabs = TRUE, linesep = "",
                 align = c("l", rep("r", 6))),
           "02_outputs/tables/errores_top3.tex")
print(errores_top3)

# (c) Dónde gana: F1 de cada modelo por zona y por tamaño del hogar.
subgrupos <- train |>
  transmute(Zona = case_when(cat_ubicacion == "RURAL" ~ "Rural",
                             cat_ubicacion == "RESTO URBANO" ~ "Resto urbano",
                             TRUE ~ "Ciudades"),
            tamano = cut(num_personas_ug, c(0, 2, 4, Inf),
                         labels = c("1 a 2 personas", "3 a 4 personas",
                                    "5 o más personas"))) |>
  bind_cols(prob_top3) |>
  mutate(pobre = pobre)
f1_subgrupos <- bind_rows(
  subgrupos |> mutate(grupo = Zona, dimension = "Zona"),
  subgrupos |> mutate(grupo = as.character(tamano), dimension = "Tamaño del hogar")
) |>
  pivot_longer(all_of(names(colores_top3)), names_to = "modelo",
               values_to = "prob") |>
  group_by(dimension, grupo, modelo) |>
  summarize(F1 = fun_f1(pobre, prob >= umbrales_top3[modelo[1]]),
            .groups = "drop") |>
  mutate(modelo = factor(modelo, levels = names(colores_top3)),
         grupo = factor(grupo, levels = rev(c("Ciudades", "Resto urbano", "Rural",
                                              "1 a 2 personas", "3 a 4 personas",
                                              "5 o más personas"))))

grafico_subgrupos <- f1_subgrupos |>
  ggplot(aes(x = F1, y = grupo, color = modelo)) +
  geom_line(aes(group = grupo), color = "#d6d5cf", linewidth = 1) +
  geom_point(size = 3.4) +
  facet_grid(dimension ~ ., scales = "free_y", space = "free_y",
             switch = "y") +
  scale_color_manual(values = colores_top3, name = NULL) +
  scale_x_continuous(labels = function(x) fun_coma(x, 2)) +
  labs(title = "La ventaja del boosting es mayor en hogares pequeños y rurales",
       subtitle = "F1 de validación cruzada por grupo de hogares, con X_5_extendido",
       x = "F1 de validación cruzada", y = NULL,
       caption = "Cada modelo con su umbral elegido en validación cruzada.") +
  tema +
  theme(legend.position = "top", legend.justification = "left",
        legend.text = element_text(color = texto_2),
        strip.placement = "outside",
        strip.text.y.left = element_text(angle = 0, hjust = 1, color = texto_2,
                                         face = "bold", size = 9),
        panel.grid.major.x = element_line(color = "#eeede8"))
ggsave("02_outputs/figures/f1_subgrupos_top3.pdf", grafico_subgrupos,
       width = 7, height = 4.4)

# (d) Qué variables usa cada modelo: las 8 más importantes, en escala de 0 a
# 100 dentro de cada modelo. LightGBM: ganancia; Random Forest: importancia
# por permutación; Elastic Net: valor absoluto del coeficiente por la
# desviación estándar de la variable (efecto de una desviación estándar).
matriz_x5 <- model.matrix(~ ., data = train[, predictores_x5])[, -1]
coef_en <- resultados_top3[["Elastic Net"]]$coeficientes
coef_en <- coef_en[rownames(coef_en) != "(Intercept)", 1]
importancias <- bind_rows(
  resultados_top3[["LightGBM"]]$importancia |>
    transmute(variable = Feature, valor = Gain, modelo = "LightGBM"),
  resultados_top3[["Random Forest"]]$importancia |>
    transmute(variable = Variable, valor = Importancia, modelo = "Random Forest"),
  tibble(variable = names(coef_en),
         valor = abs(coef_en) * apply(matriz_x5[, names(coef_en)], 2, sd),
         modelo = "Elastic Net")
) |>
  group_by(modelo) |>
  mutate(importancia = 100 * valor / max(valor)) |>
  slice_max(importancia, n = 8, with_ties = FALSE) |>
  ungroup() |>
  mutate(modelo = factor(modelo, levels = names(colores_top3)),
         # Nombre único por panel, para ordenar las barras dentro de cada uno.
         variable = paste(variable, modelo, sep = "___"))

# Nombres legibles para las variables que aparecen en la figura; las demás
# conservan su nombre en la base.
nombres_variables <- c(
  num_log_arriendo_pc = "Arriendo por persona (log)",
  bin_arriendo_nd = "Arriendo no informado",
  prop_ocupados = "Proporción de ocupados",
  num_ocupados = "Ocupados",
  num_cotizantes = "Cotizantes a pensión",
  prop_cotizantes_ocupados_ug = "Cotizantes entre ocupados",
  bin_hogar_pensionado = "Hogar con pensionado",
  num_horas_ug = "Horas trabajadas",
  bin_prima_servicios_ug = "Recibe prima de servicios",
  cat_tenenciaArriendo = "Vive en arriendo",
  num_minors = "Menores de 18 años",
  num_edad_promedio_ug = "Edad promedio",
  num_personas_ug = "Personas en el hogar",
  num_personas_cuarto = "Personas por cuarto",
  num_asalariados = "Asalariados",
  num_cuentapropia = "Cuenta propia",
  prop_dependencia = "Tasa de dependencia",
  num_educ_jefe_conyuge = "Educación jefe y cónyuge",
  cat_ubicacionRURAL = "Zona rural"
)
fun_nombre <- function(.x) {
  base <- sub("___.*$", "", .x)
  if_else(base %in% names(nombres_variables), nombres_variables[base], base)
}

grafico_importancia <- importancias |>
  mutate(variable = fct_reorder(variable, importancia)) |>
  ggplot(aes(x = importancia, y = variable, fill = modelo)) +
  geom_col(width = 0.7) +
  facet_wrap(~ modelo, scales = "free_y", ncol = 3) +
  scale_y_discrete(labels = fun_nombre) +
  scale_fill_manual(values = colores_top3) +
  scale_x_continuous(breaks = c(0, 50, 100)) +
  labs(title = "Los tres modelos usan la misma información, con otro peso",
       subtitle = "Las 8 variables más importantes de cada modelo (100 = la más importante)",
       x = "Importancia relativa", y = NULL,
       caption = paste("LightGBM: ganancia. Random Forest: permutación.",
                       "Elastic Net: |coeficiente| × desviación estándar.")) +
  tema +
  theme(strip.text = element_text(color = texto_1, face = "bold"),
        axis.text.y = element_text(size = 7.5),
        panel.grid.major.x = element_line(color = "#eeede8"))
ggsave("02_outputs/figures/importancia_top3.pdf", grafico_importancia,
       width = 8.5, height = 4.4)

# 4. Respaldo: validación cruzada frente a Kaggle ---------------------------------
# Para cada envío con puntaje público, su F1 de validación cruzada. Si los dos
# se parecen, comparar con validación cruzada (que existe para todas las
# variantes) es equivalente a comparar con Kaggle.
cv_kaggle <- variantes |>
  filter(!is.na(puntaje_publico)) |>
  arrange(desc(puntaje_publico), desc(f1_cv)) |>
  transmute(Modelo = algoritmo_corto,
            Conjunto = etiqueta_conjunto[conjunto],
            Rebalanceo = rebalanceo,
            `F1 CV` = fun_coma(f1_cv),
            `Kaggle` = fun_coma(puntaje_publico, 2),
            `Diferencia` = fun_coma(puntaje_publico - f1_cv, 3))
writeLines(kable(cv_kaggle, format = "latex", booktabs = TRUE, linesep = "",
                 align = c("l", "l", "l", "r", "r", "r")),
           "02_outputs/tables/cv_frente_kaggle.tex")

# Síntesis ---------------------------------------------------------------------
# Tamaño de cada efecto, en puntos de F1 (centésimas).
f1 <- function(.archivo_patron) {
  variantes$f1_cv[grepl(.archivo_patron, variantes$archivo)][1]
}
sin_nb <- comparacion_x5 |> filter(algoritmo_base != "Naive Bayes")
sintesis <- tibble(
  Efecto = c("Algoritmo (con X_5, sin Naive Bayes)",
             "Variables: XGBoost de X_5 a X_7",
             "Variables: LightGBM de X_5 a X_7",
             "Algoritmo con X_7: LightGBM frente a XGBoost",
             "Ajuste: Random Forest fijo frente a grilla"),
  Puntos_F1 = 100 * c(max(sin_nb$f1_cv) - min(sin_nb$f1_cv),
                      f1("^XGB_X_7") - f1("^XGB_depth"),
                      f1("^LGBM_X_7") - f1("^LGBM_X_5"),
                      f1("^LGBM_X_7") - f1("^XGB_X_7"),
                      f1("^rf_X_5_extendido_mtry") - f1("^RF_trees"))
)
print(sintesis)
write.csv(sintesis, "02_outputs/tables/sintesis_comparacion.csv",
          row.names = FALSE)
writeLines(kable(sintesis |> mutate(Puntos_F1 = fun_coma(Puntos_F1, 1)) |>
                   rename(`Puntos de F1` = Puntos_F1),
                 format = "latex", booktabs = TRUE, linesep = "", align = c("l", "r")),
           "02_outputs/tables/sintesis_comparacion.tex")

print(tabla_x5)
message("Figuras en 02_outputs/figures/ y tablas en 02_outputs/tables/.")
