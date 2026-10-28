# Figuras y cifras del deck Best Model. Ejecutar desde la raíz o desde 04_slides.
# Solo lee resultados existentes. No entrena modelos ni escribe fuera de 04_slides.
dir_slides <- if (dir.exists("04_slides")) "04_slides" else "."
raiz <- file.path(dir_slides, "..")
salida <- file.path(dir_slides, "figures")
dir.create(salida, showWarnings = FALSE)

registro <- read.csv(file.path(raiz, "02_outputs/tables/registro_modelos.csv"))
kaggle <- read.csv(file.path(raiz, "00_data/kaggle_scores.csv"))
artifact <- readRDS(file.path(raiz, "02_outputs/artifacts/importancia_permutacion_lgbm.rds"))
train <- readRDS(file.path(raiz, "03_temp/train_clean.rds"))
test <- readRDS(file.path(raiz, "03_temp/test_clean.rds"))
ganador <- registro[registro$algoritmo == "Boosting (LightGBM)" &
                      registro$conjunto == "X_7_ampliado", ]
stopifnot(nrow(ganador) == 1, length(artifact$formas) >= 4)
publico <- kaggle$puntaje_publico[match(ganador$archivo, kaggle$archivo)]

# Conteos reconstruidos de las métricas OOF registradas, no de predicciones nuevas.
positivos <- sum(train$Pobre == "Yes")
negativos <- sum(train$Pobre == "No")
TP <- round(ganador$recall_cv * positivos)
FN <- positivos - TP
FP <- round(TP / ganador$precision_cv - TP)
TN <- negativos - FP
stopifnot(abs(2 * TP / (2 * TP + FP + FN) - ganador$f1_cv) < 1e-10,
          abs(TP / (TP + FP) - ganador$precision_cv) < 1e-10)
write.csv(data.frame(Prediccion = c("Pobre", "No pobre"),
                     Pobre = c(TP, FN), No_pobre = c(FP, TN)),
          file.path(dir_slides, "matriz_oof_reconstruida.csv"), row.names = FALSE)

decimal <- function(x, d = 3) formatC(x, format = "f", digits = d, decimal.mark = ",")
entero <- function(x) formatC(as.integer(x), format = "d", big.mark = ".", decimal.mark = ",")
porcentaje <- function(x, d = 1) paste0(decimal(100 * x, d), "\\%")
macros <- c(
  NTrain = entero(nrow(train)), NTest = entero(nrow(test)), NPobres = entero(positivos),
  TasaPobreza = porcentaje(positivos / nrow(train)),
  OcupadosPobres = porcentaje(median(train$prop_ocupados[train$Pobre == "Yes"])),
  OcupadosNoPobres = porcentaje(median(train$prop_ocupados[train$Pobre == "No"])),
  NBogota = entero(sum(train$cat_ubicacion == "BOGOTA")),
  PorcBogota = porcentaje(mean(train$cat_ubicacion == "BOGOTA")),
  FoneCV = decimal(ganador$f1_cv), PublicScore = decimal(publico, 2),
  UmbralBest = decimal(ganador$umbral), PrecisionBest = porcentaje(ganador$precision_cv),
  RecallBest = porcentaje(ganador$recall_cv), AucBest = decimal(ganador$auc_cv),
  TPBest = entero(TP), FNBest = entero(FN), FPBest = entero(FP), TNBest = entero(TN),
  Leakage = porcentaje(FP / (TP + FP)), Undercoverage = porcentaje(FN / positivos),
  AccBase = porcentaje(negativos / nrow(train), 2), FoneBase = decimal(0, 2),
  SensBase = porcentaje(0, 0), EspBase = porcentaje(1, 0),
  FoneXgb = decimal(registro$f1_cv[registro$algoritmo == "Boosting (XGBoost)" &
                                  registro$conjunto == "X_7_ampliado"])
)
writeLines(paste0("\\newcommand{\\", names(macros), "}{", macros, "}"),
           file.path(dir_slides, "best_datos.tex"))

# Etiquetas concisas para los predictores mostrados.
etiquetas <- c(
  hh_rent_ug_povertyline = "Arriendo relativo",
  hh_P5130 = "Arriendo estimado",
  demo_all_persons_per_formal_proxy = "Miembros UG / formal",
  category_P6240_share_1 = "Actividad: trabajo",
  hh_Dominio = "Ciudad o zona",
  category_P7495_share_1 = "Rentas y pensiones",
  hh_P5090 = "Tenencia",
  category_P7090_share_2 = "Sin deseo de más horas",
  category_P6585s2_share_1 = "Subsidio de transporte",
  hh_P5140 = "Arriendo pagado",
  "category_P6920_share_-1" = "Sin dato: cotización",
  category_P7422_share_1 = "Ingreso de desocupados",
  hh_Depto = "Departamento",
  work_education_years_max = "Educación máxima",
  "category_P6510_share_-1" = "Sin dato: horas extra"
)
etiquetas_original <- c(
  num_si_P7510s6 = "Cesantías", num_posicion_1 = "Empleo privado",
  num_asalariados = "Asalariados", num_si_P6630s1 = "Prima de servicios",
  num_si_P7510s7 = "Otros ingresos", bin_prima_servicios_ug = "Hogar con prima",
  num_si_P6585s2 = "Subsidio de transporte",
  prop_cotizantes_ocupados_ug = "Cotizantes / ocupados",
  num_cotizantes = "Cotizantes (agregado)", num_pension_1 = "Cotizantes (conteo)",
  num_actividad_4 = "Oficios del hogar", num_tam_emp_1 = "Trabajan solos",
  num_cuentapropia = "Cuenta propia (agr.)", num_posicion_4 = "Cuenta propia (cnt.)",
  num_pension_2 = "No cotizan"
)
azul <- "#3285BA"
naranja <- "#D97736"
abrir_pdf <- function(archivo, ancho, alto, tamano = 11) {
  grDevices::cairo_pdf(archivo, width = ancho, height = alto,
                      pointsize = tamano, family = "Arial")
  par(col = "#222222", col.axis = "#444444", col.lab = "#222222",
      fg = "#BBBBBB", las = 1, bty = "n", tcl = -0.25)
}

# 15 barras en formato vertical para la mitad izquierda de una diapositiva.
imp <- head(artifact$importancia, 15)
abrir_pdf(file.path(salida, "best_importancia_top15.pdf"), 4.4, 4.0, 10.5)
par(mai = c(0.65, 1.70, 0.10, 0.18), mgp = c(1.65, 0.5, 0))
valores <- rev(100 * imp$Importancia_F1)
pos <- barplot(valores, horiz = TRUE, names.arg = rev(unname(etiquetas[imp$Variable])),
               col = rev(c(naranja, rep(azul, 14))), border = NA, axes = FALSE,
               xlim = c(0, max(valores) * 1.19), cex.names = 0.90,
               xlab = "Caída del F1 (puntos)", cex.lab = 0.88)
axis(1, at = c(0, 2, 4, 6), cex.axis = 0.9)
text(valores + 0.06, pos,
     labels = formatC(valores, digits = 2, format = "f", decimal.mark = ","),
     pos = 4, cex = 0.78, col = "#444444")
dev.off()
write.csv(imp, file.path(dir_slides, "importancia_top15.csv"), row.names = FALSE)

# IC de Welch para las diferencias estandarizadas train - test.
calcular_ic <- function(x, y) {
  x <- x[is.finite(x)]; y <- y[is.finite(y)]
  vacio <- c(Estimacion = NA_real_, Inferior = NA_real_, Superior = NA_real_)
  if (length(x) < 2 || length(y) < 2) return(vacio)
  escala <- sqrt((var(x) + var(y)) / 2)
  if (!is.finite(escala) || escala == 0) return(vacio)
  prueba <- t.test(x, y)
  c(Estimacion = unname(prueba$estimate[1] - prueba$estimate[2]) / escala,
    Inferior = prueba$conf.int[1] / escala, Superior = prueba$conf.int[2] / escala)
}
sin_bogota <- train$cat_ubicacion != "BOGOTA"
variables <- intersect(names(train), names(test))
variables <- variables[vapply(train[variables], is.numeric, logical(1)) &
                         vapply(test[variables], is.numeric, logical(1))]
balance_original <- do.call(rbind, lapply(variables, function(v) {
  rbind(data.frame(Variable = v, Grupo = "Con Bogotá",
                   t(calcular_ic(train[[v]], test[[v]]))),
        data.frame(Variable = v, Grupo = "Sin Bogotá",
                   t(calcular_ic(train[[v]][sin_bogota], test[[v]]))))
}))
con <- balance_original[balance_original$Grupo == "Con Bogotá", ]
sin <- balance_original[balance_original$Grupo == "Sin Bogotá", ]
cambio <- abs(abs(con$Estimacion) - abs(sin$Estimacion))
names(cambio) <- con$Variable
top_original <- head(names(sort(cambio[is.finite(cambio)], decreasing = TRUE)), 15)

dibujar_balance <- function(datos, vars, nombres, archivo, ancho = 4.4, alto = 4.0) {
  d <- datos[datos$Variable %in% vars, ]
  limites <- pretty(range(c(d$Inferior, d$Superior, 0), na.rm = TRUE), n = 3)
  abrir_pdf(archivo, ancho, alto, 10.5)
  par(mai = c(0.65, 1.80, 0.10, 0.12), mgp = c(1.65, 0.5, 0))
  n <- length(vars)
  plot(0, 0, type = "n", xlim = range(limites), ylim = c(0.4, n + 2.1),
       axes = FALSE, xlab = "SMD (train - test)", ylab = "", cex.lab = 0.88)
  abline(v = limites, col = "#EEEEEE", lwd = 0.7)
  abline(v = 0, col = "#888888", lty = 2)
  axis(1, at = limites, labels = formatC(limites, digits = 2, format = "f",
                                        decimal.mark = ","), cex.axis = 0.85)
  axis(2, at = n:1, labels = unname(nombres[vars]), tick = FALSE, cex.axis = 0.88)
  for (g in c("Con Bogotá", "Sin Bogotá")) {
    dg <- d[d$Grupo == g, ]
    dg <- dg[match(vars, dg$Variable), ]
    yy <- n:1 + if (g == "Con Bogotá") 0.14 else -0.14
    color <- if (g == "Con Bogotá") naranja else azul
    segments(dg$Inferior, yy, dg$Superior, yy, col = color, lwd = 1.1)
    points(dg$Estimacion, yy, pch = 16, col = color, cex = 0.60)
  }
  legend("topright", c("Con Bogotá", "Sin Bogotá"), col = c(naranja, azul),
         pch = 16, horiz = TRUE, bty = "n", text.col = "#444444", cex = 0.72)
  dev.off()
}
dibujar_balance(balance_original, top_original, etiquetas_original,
                file.path(salida, "best_balance_original.pdf"))
write.csv(balance_original[balance_original$Variable %in% top_original, ],
          file.path(dir_slides, "balance_original_top15.csv"), row.names = FALSE)

# Balance de las quince primeras variables del ganador, usando el artifact del 04.
vars_ganador <- head(artifact$importancia$Variable, 15)
nombres_ganador <- etiquetas
nombres_ganador["hh_Dominio"] <- "Resto urbano"
nombres_ganador["hh_P5090"] <- "Vivienda en arriendo"
nombres_ganador["hh_Depto"] <- "Departamento (moda)"
dibujar_balance(artifact$balance, vars_ganador, nombres_ganador,
                file.path(salida, "best_balance_bogota.pdf"), 4.4, 4.0)

# Desbalance de la variable objetivo.
abrir_pdf(file.path(salida, "best_distribucion_clase.pdf"), 3.5, 3.25, 12)
par(mai = c(0.55, 0.55, 0.18, 0.08), mgp = c(1.8, 0.55, 0))
tasas <- c(positivos, negativos) / nrow(train)
barras <- barplot(tasas, names.arg = c("Pobre", "No pobre"),
                  col = c(naranja, azul), border = NA, ylim = c(0, 1), axes = FALSE,
                  width = 0.65, space = 0.6)
axis(2, at = seq(0, 1, 0.2), labels = paste0(seq(0, 100, 20), "%"), cex.axis = 0.85)
text(barras, tasas + 0.055, paste0(round(100 * tasas, 1), "%"), cex = 1.0, col = "#333333")
dev.off()

# Las cuatro curvas usan solo tasas observadas e IC. No se ajusta ni dibuja logit.
dibujar_formas <- function(archivo, ancho = 6.6, alto = 3.8, tamano = 11) {
  vars <- head(artifact$importancia$Variable, 4)
  titulos <- c("Arriendo relativo", "Arriendo estimado",
               "Miembros por ocupado formal", "Actividad principal: trabajar")
  ejes <- c("Arriendo / (miembros UG × LP)", "Millones de pesos",
            "Miembros UG / ocupado formal", "Proporción de miembros")
  abrir_pdf(archivo, ancho, alto, tamano)
  par(mfrow = c(2, 2), mar = c(3.3, 3.5, 1.7, 0.6), mgp = c(2.3, 0.55, 0))
  for (j in seq_along(vars)) {
    d <- artifact$formas[[vars[j]]]$observado
    x <- if (j == 2) d$Valor / 1e6 else d$Valor
    plot(x, d$Tasa, type = "n", ylim = c(0, 0.7), axes = FALSE,
         xlab = ejes[j], ylab = if (j %in% c(1, 3)) "Hogares pobres" else "",
         main = titulos[j], cex.main = 0.97, cex.lab = 0.82)
    abline(h = c(0, 0.3, 0.6), col = "#EEEEEE", lwd = 0.7)
    axis(2, at = c(0, 0.3, 0.6), labels = c("0%", "30%", "60%"), cex.axis = 0.85)
    if (j == 4) {
      axis(1, at = c(0, 0.5, 1), labels = c("0%", "50%", "100%"), cex.axis = 0.85)
    } else axis(1, cex.axis = 0.85)
    segments(x, d$Inferior, x, d$Superior, col = azul, lwd = 0.8)
    lines(x, d$Tasa, col = azul, lwd = 1.4)
    points(x, d$Tasa, col = azul, pch = 16, cex = 0.65)
  }
  dev.off()
}
dibujar_formas(file.path(salida, "best_datos_curvas.pdf"))
message("Figuras actualizadas sin reentrenar modelos.")
