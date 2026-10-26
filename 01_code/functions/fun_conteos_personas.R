##########################################################
# fun_conteos_personas.R
#
# Función para describir la composición detallada de la unidad de gasto:
# cuántas personas hay en cada categoría de las principales preguntas de la
# encuesta. Complementa a fun_colapse_personas(), que resume cada pregunta en
# una sola variable. Requiere que el paquete tidyverse esté cargado.
##########################################################

#integracion de la base de datos
fun_conteos_personas <- function(.db) {
  #' Contar personas de la unidad de gasto por categoría
  #'
  #' @description \code{fun_conteos_personas()} toma la base cruda de
  #' personas y, para cada hogar, cuenta cuántos miembros de la unidad de
  #' gasto hay en cada categoría de un conjunto de preguntas (educación,
  #' posición ocupacional, actividad principal, tamaño de la empresa, edad,
  #' oficio, salud, pensión, segundo trabajo, subempleo e ingresos no
  #' laborales), cuántos respondieron "sí" a las preguntas de primas,
  #' subsidios, pagos en especie y otros ingresos, y algunos máximos del
  #' hogar (horas, antigüedad, grado escolar y edad). Los modelos de árboles
  #' pueden combinar estas piezas: no es lo mismo un hogar con un empleado
  #' del gobierno y un trabajador sin remuneración que uno con dos
  #' trabajadores por cuenta propia, aunque ambos tengan dos ocupados.
  #'
  #' @param .db \code{data.frame}. Base de datos de Kaggle a nivel de personas
  #' (train_personas.csv o test_personas.csv), sin procesar.
  #'
  #' @returns \code{tibble}. Una fila por hogar con el id del hogar,
  #' \code{id}, y las variables de conteo (prefijo \code{num_}). Las
  #' categorías de cada pregunta están fijas, así que train y test tienen
  #' siempre las mismas columnas.
  
  # Unidad de gasto: se excluyen el servicio doméstico (6), los pensionistas
  # (7) y los trabajadores (8), como en las demás variables del hogar.
  db_ug <- .db |>
    filter(!(P6050 %in% c(6, 7, 8))) |>
    mutate(
      # Grupos de edad, para describir el ciclo de vida del hogar.
      cat_edad = cut(P6040,
                     breaks = c(-Inf, 5, 11, 17, 24, 34, 44, 54, 64, Inf),
                     labels = c("0_5", "6_11", "12_17", "18_24", "25_34",
                                "35_44", "45_54", "55_64", "65_mas")),
      # Oficio: código de dos dígitos. Se agrupa por su primer dígito para
      # no crear 83 columnas.
      cat_oficio = Oficio %/% 10
    )
  
  # Preguntas categóricas y sus categorías posibles (según el diccionario
  # del DANE). "na" agrupa a quienes no responden porque la pregunta no les
  # aplica (por ejemplo, la posición ocupacional de quien no trabaja).
  preguntas <- list(
    # P6210: nivel educativo (1 ninguno ... 6 superior, 9 no sabe).
    educ      = list(var = "P6210", niveles = c(1:6, 9)),
    # P6430: posición ocupacional (1 empleado particular, 2 del gobierno,
    # 3 doméstico, 4 cuenta propia, 5 patrón, 6 y 7 sin remuneración,
    # 8 jornalero, 9 otro).
    posicion  = list(var = "P6430", niveles = 1:9),
    # P6240: actividad principal la semana pasada (1 trabajando, 2 buscando
    # trabajo, 3 estudiando, 4 oficios del hogar, 5 incapacitado, 6 otra).
    actividad = list(var = "P6240", niveles = 1:6),
    # P6870: personas en la empresa donde trabaja (1 solo ... 9 más de 100).
    tam_emp   = list(var = "P6870", niveles = 1:9),
    edad      = list(var = "cat_edad",
                     niveles = c("0_5", "6_11", "12_17", "18_24", "25_34",
                                 "35_44", "45_54", "55_64", "65_mas")),
    oficio    = list(var = "cat_oficio", niveles = 0:9),
    # P6090: afiliado a seguridad social en salud (1 sí, 2 no, 9 no sabe).
    salud     = list(var = "P6090", niveles = c(1, 2, 9)),
    # P6920: cotiza a pensión (1 sí, 2 no, 3 ya es pensionado).
    pension   = list(var = "P6920", niveles = 1:3),
    # P7040: tiene otro trabajo o negocio (1 sí, 2 no).
    segundo   = list(var = "P7040", niveles = 1:2),
    # P7090: quiere trabajar más horas (1 sí, 2 no).
    mas_horas = list(var = "P7090", niveles = 1:2),
    # P7495: recibió pagos por arriendos y/o pensiones (1 sí, 2 no).
    arr_pen   = list(var = "P7495", niveles = 1:2),
    # P7505: recibió dinero de otros hogares, intereses, dividendos o
    # cesantías (1 sí, 2 no).
    otros_ing = list(var = "P7505", niveles = 1:2)
  )
  
  # Si apareciera una categoría que no está en la lista, factor() la
  # convertiría en NA sin avisar. Mejor detenerse con un mensaje claro.
  for (nombre in names(preguntas)) {
    p <- preguntas[[nombre]]
    nuevos <- setdiff(na.omit(unique(as.character(db_ug[[p$var]]))),
                      as.character(p$niveles))
    if (length(nuevos) > 0) {
      stop(p$var, " tiene categorías no reconocidas: ",
           paste(nuevos, collapse = ", "), call. = FALSE)
    }
  }
  
  # Conteo por categoría: una tabla de hogares por categorías. Como las
  # categorías del factor son fijas, todos los hogares tienen todas las
  # columnas, aunque alguna categoría no aparezca.
  conteos <- lapply(names(preguntas), function(nombre) {
    p <- preguntas[[nombre]]
    categoria <- factor(replace_na(as.character(db_ug[[p$var]]), "na"),
                        levels = c(as.character(p$niveles), "na"))
    tabla <- table(db_ug$id, categoria)
    colnames(tabla) <- paste0("num_", nombre, "_", colnames(tabla))
    as_tibble(as.data.frame.matrix(tabla), rownames = "id")
  })
  
  # Preguntas de sí o no: cuántos respondieron "sí" (1). Todas son de
  # seguimiento y solo se hacen a quien aplica: NA significa "no".
  # P6510 horas extras; P6545 primas; P6580 bonificaciones; P6585s1 a s4
  # subsidios de alimentación, transporte, familiar y educativo; P6590,
  # P6600, P6610 y P6620 pagos en especie (alimentos, vivienda, transporte,
  # otros); P6630s1 a s4 y s6 primas de servicios, navidad, vacaciones,
  # viáticos y bonificaciones anuales; P7500s2 y s3 pensiones y pensión
  # alimentaria; P7510s1, s2, s5, s6 y s7 dinero de otros hogares del país,
  # del exterior, intereses, cesantías y otras fuentes; P7472 y P7422
  # ingresos laborales de desocupados. No se usa P7510s3 (ayudas de
  # instituciones): sería circular para un instrumento de focalización.
  preguntas_si <- c("P6510", "P6545", "P6580", "P6585s1", "P6585s2",
                    "P6585s3", "P6585s4", "P6590", "P6600", "P6610",
                    "P6620", "P6630s1", "P6630s2", "P6630s3", "P6630s4",
                    "P6630s6", "P7500s2", "P7500s3", "P7510s1", "P7510s2",
                    "P7510s5", "P7510s6", "P7510s7", "P7472", "P7422")
  
  respuestas_si <- db_ug |>
    group_by(id) |>
    summarize(across(all_of(preguntas_si), \(x) sum(x %in% 1),
                     .names = "num_si_{.col}"),
              # Máximos del hogar. NA significa que la pregunta no aplica
              # (no trabaja) y se toma como 0. En P6210s1, 99 es "no sabe".
              num_horas_max = max(replace_na(P6800, 0)),
              num_antiguedad_max = max(replace_na(P6426, 0)),
              num_grado_max = max(if_else(is.na(P6210s1) | P6210s1 == 99,
                                          0, P6210s1)),
              num_edad_max = max(P6040),
              .groups = "drop")
  
  # Unión de todas las piezas: una fila por hogar.
  db_result <- reduce(c(conteos, list(respuestas_si)), left_join, by = "id")
  
  return(db_result)
}
