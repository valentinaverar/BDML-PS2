##########################################################
# 14_best_model_analysis.R
#
# Objetivo: Analizar el modelo con mayor puntaje público en Kaggle
# (LightGBM con X_7): sus cifras principales, la importancia de sus
# variables, la forma de la relación de las principales con la pobreza y el
# balance entre train y test con y sin Bogotá.
#
# Etapa del Problem Set: Etapa 3: The Best Model.
# Insumos:  02_outputs/tables/registro_modelos.csv, 00_data/kaggle_scores.csv,
#           02_outputs/artifacts/importancia_permutacion_lgbm.rds (importancia
#           por permutación, curvas y balance, calculados en el script 04),
#           03_temp/train_clean.rds y las predicciones fuera de muestra del
#           mejor modelo en 03_temp/.
# Salidas:  02_outputs/figures/ (figuras best_*) y
#           02_outputs/tables/best_datos.tex (cifras que usa el deck
#           best_equipo_03, como comandos de LaTeX).
##########################################################
#
# Este script no entrena modelos: solo lee resultados guardados. Las cifras
# se escriben como comandos de LaTeX (por ejemplo, \FoneCV) para que el deck
# siempre muestre los valores de la última corrida.
##########################################################

# Preparación -------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse)

registro <- read.csv("02_outputs/tables/registro_modelos.csv")
kaggle <- read.csv("00_data/kaggle_scores.csv")
artifact <- readRDS("02_outputs/artifacts/importancia_permutacion_lgbm.rds")
train <- readRDS("03_temp/train_clean.rds")

# Mismo estilo de las figuras de los scripts 13 y 15.
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
  formatC(.x, format = "f", digits = .digits, decimal.mark = ",")
}
fun_entero <- function(.x) {
  formatC(as.integer(.x), format = "d", big.mark = ".", decimal.mark = ",")
}
fun_pct <- function(.x, .digits = 1) paste0(fun_coma(100 * .x, .digits), "\\%")
dir.create("02_outputs/figures", recursive = TRUE, showWarnings = FALSE)

# Mejor modelo y sus predicciones de validación cruzada.
mejor <- registro |>
  filter(algoritmo == "Boosting (LightGBM)", conjunto == "X_7_ampliado")
stopifnot(nrow(mejor) == 1)
oof <- readRDS(file.path("03_temp", paste0(mejor$archivo, ".rds")))$oof |>
  arrange(rowIndex)
pobre <- oof$obs == "Yes"
seleccionado <- oof$Yes >= mejor$umbral
tp <- sum(pobre & seleccionado)
fp <- sum(!pobre & seleccionado)
fn <- sum(pobre & !seleccionado)

# Variación del F1 entre folds, con la configuración elegida y X_5 (la
# misma de la búsqueda de hiperparámetros del script 11). Sirve para leer las
# diferencias de la búsqueda: una diferencia menor que lo que cambia el F1
# de un fold a otro no se distingue del azar.
lgbm_x5 <- registro |>
  filter(algoritmo == "Boosting (LightGBM)", conjunto == "X_5_extendido")
resultado_x5 <- readRDS(file.path("03_temp", paste0(lgbm_x5$archivo, ".rds")))
oof_x5 <- resultado_x5$oof |> arrange(rowIndex)
folds <- readRDS("03_temp/cv_folds.rds")
f1_folds <- map_dbl(folds, function(.entrenamiento) {
  i <- setdiff(seq_len(nrow(oof_x5)), .entrenamiento)
  pobre_i <- oof_x5$obs[i] == "Yes"
  sel_i <- oof_x5$Yes[i] >= resultado_x5$umbral
  tp_i <- sum(pobre_i & sel_i)
  2 * tp_i / (2 * tp_i + sum(!pobre_i & sel_i) + sum(pobre_i & !sel_i))
})
print(round(f1_folds, 4))

# 1. Cifras del deck ------------------------------------------------------
cifras <- c(
  NTrain = fun_entero(nrow(train)),
  NBogota = fun_entero(sum(train$cat_ubicacion == "BOGOTA")),
  PorcBogota = fun_pct(mean(train$cat_ubicacion == "BOGOTA")),
  PublicScore = fun_coma(kaggle$puntaje_publico[kaggle$archivo == mejor$archivo], 2),
  FoneCV = fun_coma(mejor$f1_cv),
  UmbralBest = fun_coma(mejor$umbral, 2),
  PrecisionBest = fun_pct(mejor$precision_cv),
  RecallBest = fun_pct(mejor$recall_cv),
  FNBest = fun_entero(fn),
  FPBest = fun_entero(fp),
  Leakage = fun_pct(fp / (tp + fp)),
  Undercoverage = fun_pct(fn / (tp + fn)),
  FoneXgb = fun_coma(registro$f1_cv[registro$algoritmo == "Boosting (XGBoost)" &
                                      registro$conjunto == "X_7_ampliado"]),
  FoneLgbmXcinco = fun_coma(registro$f1_cv[registro$algoritmo == "Boosting (LightGBM)" &
                                             registro$conjunto == "X_5_extendido"]),
  FoneFoldMin = fun_coma(min(f1_folds)),
  FoneFoldMax = fun_coma(max(f1_folds)),
  FoneFoldRango = fun_coma(100 * (max(f1_folds) - min(f1_folds)), 1)
)
writeLines(paste0("\\newcommand{\\", names(cifras), "}{", cifras, "}"),
           "02_outputs/tables/best_datos.tex")

# Nombres legibles de las variables que aparecen en las figuras.
etiquetas <- c(
  hh_rent_ug_povertyline = "Arriendo relativo a la línea de pobreza",
  hh_P5130 = "Arriendo estimado (vivienda propia)",
  demo_all_persons_per_formal_proxy = "Miembros por ocupado formal",
  category_P6240_share_1 = "Proporción que trabaja",
  hh_Dominio = "Ciudad o zona",
  category_P7495_share_1 = "Recibe rentas o pensiones",
  hh_P5090 = "Tenencia de la vivienda",
  category_P7090_share_2 = "No quiere trabajar más horas",
  category_P6585s2_share_1 = "Recibe subsidio de transporte",
  hh_P5140 = "Arriendo pagado",
  "category_P6920_share_-1" = "Sin dato: cotización a pensión",
  category_P7422_share_1 = "Desocupados con ingresos",
  hh_Depto = "Departamento",
  work_education_years_max = "Educación máxima de los ocupados",
  "category_P6510_share_-1" = "Sin dato: horas extra"
)

# Desbalance de clases --------------------------------------------------------
# Proporción de hogares pobres y lo que logra un clasificador ingenuo que
# declara a todos no pobres: acierta mucho y no encuentra a ningún pobre.
clases <- tibble(clase = factor(c("Pobre", "No pobre"),
                                levels = c("Pobre", "No pobre")),
                 proporcion = c(mean(train$Pobre == "Yes"),
                                mean(train$Pobre == "No")))
grafico_clases <- clases |>
  ggplot(aes(x = clase, y = proporcion, fill = clase)) +
  geom_col(width = 0.55) +
  geom_text(aes(label = paste0(fun_coma(100 * proporcion, 1), "%")),
            vjust = -0.5, size = 3.6, color = texto_1) +
  scale_fill_manual(values = c("Pobre" = color_2, "No pobre" = color_1),
                    guide = "none") +
  scale_y_continuous(labels = function(x) paste0(round(100 * x), "%"),
                     limits = c(0, 1), expand = expansion(mult = c(0, 0))) +
  labs(title = "Solo 1 de cada 5 hogares es pobre",
       subtitle = "Distribución de la variable Pobre en train",
       x = NULL, y = "Hogares",
       caption = paste0("Train: ", fun_entero(nrow(train)), " hogares.")) +
  tema +
  theme(panel.grid.major.x = element_blank())
ggsave("02_outputs/figures/best_distribucion_clase.pdf", grafico_clases,
       width = 5, height = 4.4)

ingenuo <- tibble(
  `Métrica` = c("Accuracy", "F1", "Sensibilidad (recall)", "Especificidad"),
  `Todos no pobres` = c(fun_pct(mean(train$Pobre == "No"), 1), "0",
                        "0\\%", "100\\%"))
writeLines(knitr::kable(ingenuo, format = "latex", booktabs = TRUE,
                        linesep = "", align = c("l", "r"), escape = FALSE),
           "02_outputs/tables/best_linea_base.tex")

# 2. Importancia por permutación -------------------------------------------
# Caída del F1 (en puntos) al desordenar cada variable: cuánto depende la
# predicción de ella. No son efectos causales.
importancia <- head(artifact$importancia, 15) |>
  mutate(nombre = fct_reorder(etiquetas[Variable], Importancia_F1),
         puntos = 100 * Importancia_F1,
         grupo = case_when(
           Variable %in% c("hh_rent_ug_povertyline", "hh_P5130", "hh_P5090",
                           "hh_P5140") ~ "Vivienda y arriendo",
           Variable %in% c("demo_all_persons_per_formal_proxy",
                           "category_P6240_share_1", "category_P7090_share_2",
                           "category_P6920_share_-1", "work_education_years_max",
                           "category_P6510_share_-1") ~ "Trabajo",
           Variable %in% c("category_P7495_share_1", "category_P6585s2_share_1",
                           "category_P7422_share_1") ~ "Otros ingresos",
           TRUE ~ "Territorio"),
         grupo = factor(grupo, levels = c("Vivienda y arriendo", "Trabajo",
                                          "Otros ingresos", "Territorio")))

grafico_importancia <- importancia |>
  ggplot(aes(x = puntos, y = nombre, fill = grupo)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = fun_coma(puntos, 1)), hjust = -0.2, size = 3,
            color = texto_1) +
  scale_fill_manual(values = c("Vivienda y arriendo" = color_1,
                               "Trabajo" = color_2,
                               "Otros ingresos" = "#1baf7a",
                               "Territorio" = "#a3a29b"),
                    name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(title = "El arriendo es la variable que más pesa en la predicción",
       subtitle = "Caída del F1 al desordenar cada variable (puntos), 15 principales",
       x = "Caída del F1 (puntos)", y = NULL,
       caption = "Importancia por permutación: 3 repeticiones en 5 folds. No son efectos causales.") +
  tema +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(color = texto_1, size = 9))
ggsave("02_outputs/figures/best_importancia_top15.pdf", grafico_importancia,
       width = 7.5, height = 4.6)

# 3. Forma de la relación de las cuatro principales con la pobreza ----------
# Tasa de pobreza observada por grupos de la variable, con su intervalo del
# 95%. Es descriptiva: muestra que la relación no es una línea recta.
titulos <- c(hh_rent_ug_povertyline = "Arriendo / (miembros × línea de pobreza)",
             hh_P5130 = "Arriendo estimado (millones de pesos)",
             demo_all_persons_per_formal_proxy = "Miembros por ocupado formal",
             category_P6240_share_1 = "Proporción de miembros que trabaja")
curvas <- map_dfr(names(titulos), function(.v) {
  artifact$formas[[.v]]$observado |>
    mutate(variable = titulos[[.v]],
           Valor = if (.v == "hh_P5130") Valor / 1e6 else Valor)
}) |>
  mutate(variable = factor(variable, levels = titulos))

grafico_curvas <- curvas |>
  ggplot(aes(x = Valor, y = Tasa)) +
  geom_ribbon(aes(ymin = Inferior, ymax = Superior), fill = color_1,
              alpha = 0.2) +
  geom_line(color = color_1, linewidth = 0.8) +
  geom_point(color = color_1, size = 1.4) +
  facet_wrap(~ variable, scales = "free_x", ncol = 2) +
  scale_x_continuous(labels = scales::label_number(big.mark = ".",
                                                   decimal.mark = ",")) +
  scale_y_continuous(labels = function(x) paste0(round(100 * x), "%"),
                     limits = c(0, 0.7)) +
  labs(title = "Las variables principales se relacionan con la pobreza por tramos",
       subtitle = "Porcentaje de hogares pobres por grupos de cada variable",
       x = NULL, y = "Hogares pobres",
       caption = "Train, grupos de igual tamaño entre los percentiles 1 y 99; banda: intervalo del 95%.") +
  tema +
  theme(strip.text = element_text(color = texto_1, face = "bold"),
        panel.grid.major.x = element_blank())
ggsave("02_outputs/figures/best_datos_curvas.pdf", grafico_curvas,
       width = 7.5, height = 4.6)

# 4. Balance entre train y test, con y sin Bogotá --------------------------
# Bogotá está en train pero no en test. Para las 15 variables principales:
# diferencia estandarizada de medias (SMD) entre train y test, con su
# intervalo del 95%, incluyendo o excluyendo Bogotá de train.
balance <- artifact$balance |>
  filter(Variable %in% importancia$Variable, is.finite(Estimacion)) |>
  mutate(nombre = factor(etiquetas[Variable],
                         levels = levels(importancia$nombre)),
         Grupo = if_else(startsWith(Grupo, "Con"), "Con Bogotá", "Sin Bogotá"))

grafico_balance <- balance |>
  ggplot(aes(x = Estimacion, y = nombre, color = Grupo)) +
  geom_vline(xintercept = 0, color = "#a3a29b", linetype = "dashed") +
  geom_errorbarh(aes(xmin = Inferior, xmax = Superior), height = 0,
                 position = position_dodge(width = 0.6), linewidth = 0.6) +
  geom_point(position = position_dodge(width = 0.6), size = 1.8) +
  scale_color_manual(values = c("Con Bogotá" = color_2,
                                "Sin Bogotá" = color_1), name = NULL) +
  scale_x_continuous(labels = function(x) fun_coma(x, 2)) +
  labs(title = "Salvo el departamento, train y test se parecen en las variables principales",
       subtitle = "Diferencia estandarizada de medias entre train y test (SMD)",
       x = "SMD (train - test)", y = NULL,
       caption = "Las 15 variables más importantes del mejor modelo; intervalo del 95%.") +
  tema +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(color = texto_1, size = 9))
ggsave("02_outputs/figures/best_balance_bogota.pdf", grafico_balance,
       width = 7.5, height = 4.6)

message("Cifras y figuras del mejor modelo en 02_outputs/.")
