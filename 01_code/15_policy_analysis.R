##########################################################
# 15_policy_analysis.R
#
# Objetivo: Traducir los errores del mejor modelo a costos de política:
# filtración (leakage, falsos positivos) frente a subcobertura
# (undercoverage, falsos negativos), y evaluar cuánto se pierde si el modelo
# se aplica con una encuesta más corta.
#
# Etapa del Problem Set: Etapa 4: From Prediction to Policy.
# Insumos:  02_outputs/tables/registro_modelos.csv, 03_temp/predictores.rds y
#           las predicciones fuera de muestra del mejor modelo, que su script
#           guarda en 03_temp/ con el nombre de su envío.
# Salidas:  02_outputs/figures/ y 02_outputs/tables/ (figuras y tablas del
#           deck best_equipo_03).
##########################################################
#
# Todo se calcula con las predicciones de validación cruzada: cada hogar de
# train recibe la probabilidad de un modelo que no lo usó para entrenar. Así
# los errores que se reportan son fuera de muestra. No se reentrena ningún
# modelo: mover el umbral solo cambia qué hogares quedan en la lista.
##########################################################

# Preparación -------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, knitr)

registro <- read.csv("02_outputs/tables/registro_modelos.csv")
predictores <- readRDS("03_temp/predictores.rds")

# Mismo estilo de las figuras del script 13.
color_1 <- "#2a78d6"
color_2 <- "#eb6834"
texto_1 <- "#0b0b0b"
texto_2 <- "#52514e"
tema <- theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        axis.text = element_text(color = texto_2),
        axis.title = element_text(color = texto_2),
        plot.title = element_text(color = texto_1, face = "bold", hjust = 0.5,
                                  margin = margin(b = 4)),
        plot.subtitle = element_text(color = texto_2, hjust = 0.5,
                                     margin = margin(b = 14)),
        plot.caption = element_text(color = texto_2, hjust = 0.5,
                                    margin = margin(t = 14)),
        plot.margin = margin(12, 14, 10, 14),
        plot.title.position = "plot",
        plot.caption.position = "plot",
        legend.position = "top",
        legend.text = element_text(color = texto_2))
fun_coma <- function(.x, .digits = 3) {
  ifelse(is.na(.x), "--",
         formatC(.x, format = "f", digits = .digits, decimal.mark = ","))
}
fun_entero <- function(.x) trimws(format(.x, big.mark = ".", decimal.mark = ","))
fun_pct <- function(.x, .digits = 1) paste0(fun_coma(100 * .x, .digits), "\\%")
dir.create("02_outputs/figures", recursive = TRUE, showWarnings = FALSE)

# Mejor modelo: la variante con mayor F1 de validación cruzada, que también
# tuvo el mayor puntaje público en Kaggle (LightGBM con X_7).
mejor <- registro |> slice_max(f1_cv, n = 1, with_ties = FALSE)
ruta_mejor <- file.path("03_temp", paste0(mejor$archivo, ".rds"))
if (!file.exists(ruta_mejor)) {
  stop("Faltan las predicciones fuera de muestra de ", mejor$archivo,
       " en 03_temp/. Corra su script de modelo.", call. = FALSE)
}
resultado <- readRDS(ruta_mejor)
oof <- resultado$oof |> arrange(rowIndex)
pobre <- oof$obs == "Yes"
prob <- oof$Yes
umbral <- resultado$umbral

# Errores con un umbral: lista de hogares seleccionados, pobres encontrados,
# leakage (proporción de la lista que no es pobre) y undercoverage
# (proporción de los pobres que queda por fuera).
fun_errores <- function(.u) {
  sel <- prob >= .u
  tp <- sum(pobre & sel); fp <- sum(!pobre & sel); fn <- sum(pobre & !sel)
  tibble(umbral = .u, lista = sum(sel), tp = tp, fp = fp, fn = fn,
         tn = sum(!pobre & !sel),
         leakage = fp / (tp + fp), undercoverage = fn / (tp + fn),
         f1 = 2 * tp / (2 * tp + fp + fn))
}

# 1. Matriz de confusión del mejor modelo ----------------------------------
# Conteos exactos con las predicciones guardadas y el umbral elegido.
elegido <- fun_errores(umbral)
stopifnot(abs(elegido$f1 - mejor$f1_cv) < 1e-8)
matriz <- tibble(` ` = c("Predicho pobre", "Predicho no pobre"),
                 `Pobre` = fun_entero(c(elegido$tp, elegido$fn)),
                 `No pobre` = fun_entero(c(elegido$fp, elegido$tn)))
writeLines(kable(matriz, format = "latex", booktabs = TRUE, linesep = "",
                 align = c("l", "r", "r")),
           "02_outputs/tables/matriz_confusion_mejor.tex")
metricas <- tibble(
  `Métrica` = c("Precisión", "Recall", "F1", "AUC", "Accuracy"),
  Valor = c(fun_pct(mejor$precision_cv), fun_pct(mejor$recall_cv),
            fun_coma(mejor$f1_cv), fun_coma(mejor$auc_cv),
            fun_pct((elegido$tp + elegido$tn) / length(pobre))))
writeLines(kable(metricas, format = "latex", booktabs = TRUE, linesep = "",
                 align = c("l", "r"), escape = FALSE),
           "02_outputs/tables/metricas_mejor.tex")
print(elegido)

# 2. Leakage y undercoverage según el umbral ------------------------------
# El F1 pesa igual los dos errores. Una agencia puede preferir otro balance:
# bajar el umbral deja menos pobres por fuera, a costa de una lista más larga
# con más hogares no pobres.
curva <- map_dfr(seq(0.10, 0.70, by = 0.01), fun_errores) |>
  select(umbral, Leakage = leakage, Undercoverage = undercoverage) |>
  pivot_longer(-umbral, names_to = "error", values_to = "valor")

grafico_umbral <- curva |>
  ggplot(aes(x = umbral, y = valor, color = error)) +
  geom_vline(xintercept = umbral, color = "#d6d5cf", linewidth = 0.8) +
  geom_line(linewidth = 0.9) +
  annotate("text", x = umbral + 0.01, y = 0.62, hjust = 0, size = 3.2,
           color = texto_2,
           label = paste0("Umbral del F1 (", fun_coma(umbral, 2), ")")) +
  scale_color_manual(values = c("Leakage" = color_2,
                                "Undercoverage" = color_1),
                     labels = c("Leakage: no pobres en la lista",
                                "Undercoverage: pobres por fuera"),
                     name = NULL) +
  scale_x_continuous(labels = function(x) fun_coma(x, 1)) +
  scale_y_continuous(labels = function(x) paste0(round(100 * x), "%"),
                     limits = c(0, 0.65)) +
  labs(title = "Bajar el umbral deja menos pobres por fuera, pero incluye más no pobres",
       subtitle = "Leakage y undercoverage de LightGBM con X_7 según el umbral",
       x = "Umbral de clasificación", y = NULL,
       caption = "Predicciones de validación cruzada en train (164.960 hogares).") +
  tema +
  theme(panel.grid.major.x = element_blank())
ggsave("02_outputs/figures/leakage_undercoverage_umbral.pdf", grafico_umbral,
       width = 7, height = 4.4)

# Tabla con algunos umbrales: qué cambia en hogares si la agencia se aparta
# del umbral que maximiza el F1.
tabla_umbrales <- map_dfr(sort(c(0.2, 0.3, umbral, 0.5, 0.6)), fun_errores) |>
  transmute(Umbral = fun_coma(umbral, 2),
            `Hogares en la lista` = fun_entero(lista),
            `Pobres por fuera` = fun_entero(fn),
            `No pobres en la lista` = fun_entero(fp),
            Undercoverage = fun_pct(undercoverage),
            Leakage = fun_pct(leakage),
            F1 = fun_coma(f1))
writeLines(kable(tabla_umbrales, format = "latex", booktabs = TRUE,
                 linesep = "", align = c("l", rep("r", 6)), escape = FALSE),
           "02_outputs/tables/errores_umbrales_mejor.tex")
print(tabla_umbrales)

# 3. ¿Cabe en una encuesta corta? -----------------------------------------
# F1 según el número de variables que hay que recoger. Se muestran los dos
# algoritmos evaluados con más de un conjunto: el logit (X_1 a X_5) y
# LightGBM (X_5 a X_7). Con X_5 los dos se pueden comparar directamente.
etiqueta_conjunto <- c(X_1_vivienda = "X_1", X_2_demografia = "X_2",
                       X_3_empleo = "X_3", X_4_completo = "X_4",
                       X_5_extendido = "X_5", X_6_detallado = "X_6",
                       X_7_ampliado = "X_7")
escalera <- registro |>
  filter(algoritmo %in% c("Logit", "Boosting (LightGBM)"),
         conjunto %in% names(etiqueta_conjunto)) |>
  group_by(algoritmo, conjunto) |>
  slice_max(f1_cv, n = 1, with_ties = FALSE) |>
  ungroup() |>
  mutate(variables = lengths(predictores[conjunto]),
         etiqueta = etiqueta_conjunto[conjunto],
         modelo = if_else(algoritmo == "Logit", "Logit", "LightGBM"))
print(escalera |> select(modelo, etiqueta, variables, f1_cv))

grafico_encuesta <- escalera |>
  ggplot(aes(x = variables, y = f1_cv, color = modelo)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 3.2) +
  # Con X_5 los dos puntos casi se tocan: la etiqueta del logit va abajo.
  geom_text(aes(label = paste0(etiqueta, ": ", fun_coma(f1_cv)),
                vjust = if_else(modelo == "Logit" & etiqueta == "X_5", 2, -1.1)),
            size = 3.1, color = texto_1, show.legend = FALSE) +
  scale_color_manual(values = c("LightGBM" = color_1, "Logit" = color_2),
                     name = NULL) +
  scale_x_log10(breaks = c(5, 11, 21, 43, 148, 1296),
                labels = function(x) fun_entero(x),
                expand = expansion(mult = c(0.05, 0.08))) +
  scale_y_continuous(labels = function(x) fun_coma(x, 2),
                     limits = c(0.5, 0.8)) +
  labs(title = "Con 43 variables, el boosting queda a 3,7 puntos del mejor modelo",
       subtitle = "F1 de validación cruzada según el número de variables que hay que recoger",
       x = "Variables del conjunto (escala logarítmica)", y = "F1 de validación cruzada",
       caption = "Cada conjunto agrega un bloque al anterior: vivienda, demografía, empleo, formalidad y más detalle.") +
  tema +
  theme(panel.grid.major.x = element_blank())
ggsave("02_outputs/figures/f1_tamano_encuesta.pdf", grafico_encuesta,
       width = 7, height = 4.4)

# 4. Qué se pierde con la encuesta corta -----------------------------------
# Matriz de confusión de LightGBM con X_5 (43 variables) frente a X_7 (1.296),
# cada uno con su propio umbral que maximiza el F1.
lgbm_x5 <- registro |>
  filter(algoritmo == "Boosting (LightGBM)", conjunto == "X_5_extendido")
resultado_x5 <- readRDS(file.path("03_temp", paste0(lgbm_x5$archivo, ".rds")))
oof_x5 <- resultado_x5$oof |> arrange(rowIndex)
stopifnot(all(oof_x5$obs == oof$obs))

fun_conteos <- function(.prob, .u) {
  sel <- .prob >= .u
  tp <- sum(pobre & sel); fp <- sum(!pobre & sel); fn <- sum(pobre & !sel)
  c(tp = tp, fn = fn, fp = fp, lista = tp + fp,
    precision = tp / (tp + fp), recall = tp / (tp + fn),
    f1 = 2 * tp / (2 * tp + fp + fn))
}
conteos <- list(x7 = fun_conteos(prob, umbral),
                x5 = fun_conteos(oof_x5$Yes, resultado_x5$umbral))
fila <- function(.nombre, .campo, .formato) {
  tibble(` ` = .nombre,
         `X\\_7 (1.296 variables)` = .formato(conteos$x7[[.campo]]),
         `X\\_5 (43 variables)` = .formato(conteos$x5[[.campo]]))
}
comparacion_encuesta <- bind_rows(
  fila("Pobres encontrados", "tp", fun_entero),
  fila("Pobres por fuera (undercoverage)", "fn", fun_entero),
  fila("No pobres en la lista (leakage)", "fp", fun_entero),
  fila("Hogares en la lista", "lista", fun_entero),
  fila("Precisión", "precision", fun_pct),
  fila("Recall", "recall", fun_pct),
  fila("F1", "f1", fun_coma))
writeLines(kable(comparacion_encuesta, format = "latex", booktabs = TRUE,
                 linesep = "", align = c("l", "r", "r"), escape = FALSE),
           "02_outputs/tables/comparacion_x5_x7.tex")
print(comparacion_encuesta)

message("Figuras y tablas de política en 02_outputs/.")
