##########################################################
# 13_compare_algorithms.R
#
# Objetivo: Comparar los algoritmos y separar cuánto del desempeño se debe
# al algoritmo y cuánto a las variables, al ajuste de hiperparámetros y al
# tratamiento del desbalance de clases.
#
# Etapa del Problem Set: Etapa 2: Training and Comparing Algorithms.
# Insumos:  02_outputs/tables/registro_modelos.csv (F1 de validación cruzada
#           de cada variante, todas con las mismas particiones) y
#           00_data/kaggle_scores.csv (puntaje público).
# Salidas:  02_outputs/figures/ (figuras del deck compare_equipo_03) y
#           02_outputs/tables/ (tablas en .csv y en LaTeX para las slides).
##########################################################
#
# Las comparaciones usan el F1 de validación cruzada y no el puntaje de
# Kaggle: todas las variantes se evaluaron con los mismos hogares (03_temp/
# cv_folds.rds), mientras que en Kaggle no se subieron todas.
#
#   1. Mismas variables (X_5_extendido), distintos algoritmos: efecto del
#      algoritmo.
#   2. Mismo algoritmo, más variables (X_5, X_6 y X_7): efecto de las
#      variables.
#   3. Ajuste de hiperparámetros y rebalanceo de clases, con el algoritmo y
#      las variables fijos.
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
        plot.title = element_text(color = texto_1, face = "bold"),
        plot.subtitle = element_text(color = texto_2),
        plot.caption = element_text(color = texto_2, hjust = 0),
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

# 1. Mismas variables, distintos algoritmos ----------------------------------
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

# Figura: cada algoritmo frente al logit, la referencia natural para un
# economista. Los algoritmos se agrupan por familia y la línea une cada
# punto con el F1 del logit, de modo que se lee de inmediato cuánto gana o
# pierde cada uno.
f1_logit <- comparacion_x5$f1_cv[comparacion_x5$algoritmo_base == "Logit"]

datos_x5 <- comparacion_x5 |>
  mutate(
    familia_grafico = case_when(
      algoritmo_base %in% c("XGBoost", "LightGBM") ~ "Boosting de\ngradiente",
      algoritmo_base %in% c("Random Forest", "AdaBoost") ~ "Ensambles\nde árboles",
      algoritmo_base %in% c("Logit", "Elastic Net", "LPM") ~ "Modelos\nlineales",
      algoritmo_base == "CART" ~ "Un solo\nárbol",
      TRUE ~ "Naive\nBayes"),
    familia_grafico = fct_reorder(familia_grafico, f1_cv, .fun = max,
                                  .desc = TRUE),
    algoritmo_base = fct_reorder(algoritmo_base, f1_cv),
    diferencia = 100 * (f1_cv - f1_logit),
    etiqueta = if_else(algoritmo_base == "Logit",
                       paste0(fun_coma(f1_cv), "  (referencia)"),
                       paste0(fun_coma(f1_cv), "  (",
                              if_else(diferencia >= 0, "+", "-"),
                              fun_coma(abs(diferencia), 1), ")")),
    destacado = familia == "Boosting de gradiente"
  )

grafico_x5 <- datos_x5 |>
  ggplot(aes(y = algoritmo_base)) +
  geom_vline(xintercept = f1_logit, color = color_resto, linetype = "22",
             linewidth = 0.5) +
  geom_segment(aes(x = f1_logit, xend = f1_cv, yend = algoritmo_base,
                   color = destacado), linewidth = 1.2, alpha = 0.6) +
  geom_point(aes(x = f1_cv, color = destacado), size = 4) +
  geom_text(aes(x = pmax(f1_cv, f1_logit), label = etiqueta), color = texto_1,
            size = 3.2, hjust = -0.12) +
  facet_grid(familia_grafico ~ ., scales = "free_y", space = "free_y",
             switch = "y") +
  scale_color_manual(values = c("TRUE" = color_1, "FALSE" = color_resto)) +
  scale_x_continuous(limits = c(0.635, 0.765),
                     breaks = seq(0.64, 0.76, by = 0.02),
                     labels = function(x) fun_coma(x, 2)) +
  labs(title = "Con las mismas variables, el boosting le gana al logit por 2 puntos",
       subtitle = "F1 de validación cruzada con las 43 variables de X_5_extendido",
       x = "F1 de validación cruzada", y = NULL,
       caption = paste("Entre paréntesis, diferencia con el logit en puntos de F1.",
                       "Mejor variante sin rebalanceo de cada algoritmo; mismos 5 folds para todos.")) +
  tema +
  theme(plot.title.position = "plot",
        plot.caption.position = "plot",
        strip.placement = "outside",
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
