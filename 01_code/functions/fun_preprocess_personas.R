##########################################################
# fun_preprocess_personas.R
#
# Función para limpiar la base de personas: renombra las variables con
# nombres descriptivos y crea indicadores a nivel persona. Requiere que el
# paquete tidyverse esté cargado.
##########################################################

fun_preprocess_personas <- function(.db) {
  #' Preprocesar base de datos de personas
  #'
  #' @description \code{fun_preprocess_personas()} retorna una base de datos con
  #' nombres descriptivos y etiquetas en sus variables categóricas.
  #'
  #' @param .db \code{data.frame}. Base de datos de Kaggle a nivel de personas
  #' (train_personas.csv o test_personas.csv).
  #'
  #' @returns \code{tibble}. Una fila por persona con el id del hogar, \code{id},
  #' el id de la persona dentro del hogar, \code{id_persona}, y los indicadores:
  #' hombre, \code{bin_male}; jefe del hogar, \code{bin_head}; cónyuge del
  #' jefe, \code{bin_conyuge}; ocupado, \code{bin_occup}; inactivo,
  #' \code{bin_inactivo}; en edad de trabajar, \code{bin_edad_trabajar}; menor
  #' de edad, \code{bin_minor}; mayor de 65, \code{bin_old}; su máximo nivel
  #' educativo, \code{cat_educ}; si pertenece a la unidad de gasto,
  #' \code{bin_ug}; si cotiza a pensión, \code{bin_cotiza}; si recibió pensión
  #' el mes pasado, \code{bin_pensionado}; su posición ocupacional,
  #' \code{cat_posicion}; si es asalariado, \code{bin_asalariado}; si trabaja
  #' por cuenta propia, \code{bin_cuentapropia}; y las horas que trabaja a la
  #' semana, \code{num_horas}.
  
  # Variables que usamos y sus nombres descriptivos. Todas existen tanto en
  # train como en test.
  text_vars <- c('id', 'id_persona' = 'Orden', 'num_age' = 'P6040',
                 'cat_gender' = 'P6020', 'bin_employed' = 'Oc',
                 'bin_inactivo' = 'Ina', 'cat_educ' = 'P6210',
                 'cat_relationship' = 'P6050',
                 'cat_cotiza' = 'P6920', 'cat_recibe_pension' = 'P7500s2',
                 'cat_posicion' = 'P6430', 'num_horas' = 'P6800',
                 # Nuevas fuentes, presentes en train y test de personas.
                 'bin_des' = 'Des', 'cat_actividad' = 'P6240',
                 'cat_salud_afiliacion' = 'P6090',
                 'cat_mas_horas' = 'P7090', 'cat_tam_empresa' = 'P6870',
                 'cat_prima_servicios' = 'P6630s1',
                 'cat_dinero_hogares' = 'P7510s1')
  
  db_result <- .db |>
    select(all_of(text_vars)) |>
    mutate(
      # P6020: 1 = hombre, 2 = mujer.
      bin_male = ifelse(test = cat_gender == 1, yes = 1, no = 0),
      # P6050: 1 = jefe(a) del hogar; 2 = pareja, esposo(a), cónyuge.
      bin_head = ifelse(test = cat_relationship == 1, yes = 1, no = 0),
      bin_conyuge = ifelse(test = cat_relationship == 2, yes = 1, no = 0),
      # Oc e Ina valen 1 (sí) o NA (no), según el diccionario del DANE.
      bin_occup = ifelse(test = is.na(bin_employed), yes = 0, no = 1),
      bin_inactivo = ifelse(test = is.na(bin_inactivo), yes = 0, no = 1),
      # Grupos de edad. Se considera en edad de trabajar a las personas de
      # 18 a 65 años.
      bin_minor = ifelse(test = num_age < 18, yes = 1, no = 0),
      bin_old = ifelse(test = num_age > 65, yes = 1, no = 0),
      bin_edad_trabajar = as.integer(between(num_age, 18, 65)),
      # P6210: 1 = ninguno, ..., 6 = superior, 9 = no sabe. Los NA son niños
      # menores de 3 años, a quienes la encuesta no pregunta.
      cat_educ = case_when(is.na(cat_educ) ~ 1, # No reportado: 'ninguno'.
                           cat_educ == 9 ~ 1,   # No sabe: 'ninguno'.
                           TRUE ~ cat_educ),    # Resto: valor original.
      # Factor ordenado: sin el orden no se puede tomar el máximo por hogar.
      cat_educ = factor(x = cat_educ,
                        levels = 1:6,
                        ordered = TRUE,
                        labels = c("Ninguno", "Preescolar", "Primaria",
                                   "Secundaria", "Media", "Universitaria")),
      # Unidad de gasto: personas que comparten el presupuesto del hogar. Según
      # P6050, quedan por fuera el empleado(a) del servicio (6), el
      # pensionista (7) y el trabajador (8). Así lo define el DANE: en los
      # datos, Nper - Npersug es exactamente el número de estas personas.
      bin_ug = ifelse(test = cat_relationship %in% c(6, 7, 8), yes = 0, no = 1),
      # P6920: ¿Está cotizando a un fondo de pensiones? 1 = sí, 2 = no,
      # 3 = ya es pensionado. Solo se pregunta a los ocupados: NA significa
      # que la persona no trabaja y, por lo tanto, no cotiza. Quien ya es
      # pensionado (3) tampoco cotiza; su pensión se captura con P7500s2.
      bin_cotiza = ifelse(test = cat_cotiza %in% 1, yes = 1, no = 0),
      # P7500s2: ¿El mes pasado recibió pagos por pensiones o jubilaciones?
      # 1 = sí, 2 = no, 9 = no sabe. Solo se pregunta a quien respondió "sí"
      # en P7495 (¿recibió pagos por arriendos y/o pensiones?): NA significa
      # que no recibió pensión. "No sabe" (9) se trata como no.
      bin_pensionado = ifelse(test = cat_recibe_pension %in% 1, yes = 1, no = 0),
      # P6430: posición ocupacional en el trabajo principal. 1 = obrero o
      # empleado de empresa particular, 2 = obrero o empleado del gobierno,
      # 3 = empleado doméstico, 4 = cuenta propia, 5 = patrón o empleador,
      # 6 = trabajador familiar sin remuneración, 7 = trabajador sin
      # remuneración en empresas de otros hogares, 8 = jornalero o peón,
      # 9 = otro. Solo se pregunta a los ocupados: NA significa que la
      # persona no trabaja.
      # Asalariados: trabajan para un empleador, privado o público (1 y 2).
      bin_asalariado = ifelse(test = cat_posicion %in% c(1, 2), yes = 1, no = 0),
      # Cuenta propia: trabajan por su cuenta, sin empleador ni empleados (4).
      bin_cuentapropia = ifelse(test = cat_posicion %in% 4, yes = 1, no = 0),
      # Como factor: NA pasa a "No ocupado", y 6 y 7 se unen en "Sin
      # remuneracion", porque ambos describen trabajo sin pago y el 7 es muy
      # poco frecuente. Los indicadores de arriba se crean antes de este
      # cambio, a partir del código original.
      cat_posicion = case_when(is.na(cat_posicion) ~ 0,
                               cat_posicion == 7 ~ 6,
                               TRUE ~ cat_posicion),
      cat_posicion = factor(x = cat_posicion,
                            levels = c(0, 1, 2, 3, 4, 5, 6, 8, 9),
                            labels = c("No ocupado", "Empleado particular",
                                       "Empleado gobierno",
                                       "Empleado domestico", "Cuenta propia",
                                       "Patron", "Sin remuneracion",
                                       "Jornalero", "Otro")),
      # P6800: horas que trabaja normalmente a la semana en el trabajo
      # principal. Solo se pregunta a los ocupados: NA significa que la
      # persona no trabaja, es decir, 0 horas.
      num_horas = replace_na(num_horas, 0),
      # PREDICTORES NUEVOS: solo añadidos; ningún indicador previo cambia.
      # La edad original se conserva para la agregación posterior.
      bin_female = as.integer(cat_gender == 2),
      bin_menor_6 = as.integer(num_age < 6),
      bin_adulto_universitario = as.integer(
        num_age >= 18 & cat_educ == "Universitaria"),
      # Des = 1 si está desocupado; NA para los demás.
      bin_desocupado = as.integer(bin_des %in% 1),
      # P6240 = 3: estudió la mayor parte del tiempo la semana anterior.
      bin_estudiante = as.integer(cat_actividad %in% 3),
      # P6090 = 2: declara no estar afiliado al sistema de salud.
      bin_no_afiliado_salud = as.integer(cat_salud_afiliacion %in% 2),
      # P7090: además de las horas que trabaja, ¿quiere trabajar más horas?
      # 1 = sí, 2 = no. Solo se pregunta a los ocupados: NA = no ocupado.
      # Mide subempleo por insuficiencia de horas.
      bin_subempleo = as.integer(cat_mas_horas %in% 1),
      # P6630s1: en los últimos 12 meses, ¿recibió prima de servicios?
      # 1 = sí, 2 = no. Es una prestación del empleo formal asalariado; NA
      # (no se le pregunta) significa que no la recibió.
      bin_prima_servicios = as.integer(cat_prima_servicios %in% 1),
      # P7510s1: en los últimos 12 meses, ¿recibió dinero de otros hogares o
      # personas residentes en el país? 1 = sí, 2 = no, 9 = no sabe. Solo se
      # pregunta a quien declaró otros ingresos en P7505: NA = no recibió.
      bin_dinero_hogares = as.integer(cat_dinero_hogares %in% 1),
      # P6870: personas que tiene la empresa donde trabaja. Solo se pregunta
      # a los ocupados: NA pasa a "No ocupado".
      cat_tam_empresa = factor(x = replace_na(cat_tam_empresa, 0),
                               levels = 0:9,
                               labels = c("No ocupado", "Solo", "2 a 3",
                                          "4 a 5", "6 a 10", "11 a 19",
                                          "20 a 30", "31 a 50", "51 a 100",
                                          "101 o mas"))
    ) |>
    select(id, id_persona, bin_male, bin_head, bin_conyuge, bin_occup,
           bin_inactivo, bin_edad_trabajar, cat_educ, bin_minor, bin_old,
           bin_ug, bin_cotiza, bin_pensionado, cat_posicion, bin_asalariado,
           bin_cuentapropia, num_horas,
           # Variables auxiliares nuevas para agregación al hogar.
           num_age, bin_female, bin_menor_6,
           bin_adulto_universitario, bin_desocupado, bin_estudiante,
           bin_no_afiliado_salud, bin_subempleo, bin_prima_servicios,
           bin_dinero_hogares, cat_tam_empresa)
  
  return(db_result)
}
