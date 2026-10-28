##########################################################
# compile_slides.R
#
# Objetivo: Compilar los dos decks de la presentación a PDF a partir de
# las figuras y tablas que generan los scripts de 01_code/.
#
# Insumos:  04_slides/compare_equipo_03.tex (lee 02_outputs/figures/ y
#           02_outputs/tables/) y 04_slides/best_equipo_03.tex.
# Salidas:  04_slides/compare_equipo_03.pdf y 04_slides/best_equipo_03.pdf.
##########################################################
#
# Requiere el paquete tinytex y una instalación de LaTeX (por ejemplo,
# tinytex::install_tinytex()). Si no están disponibles, se avisa y se sigue:
# las diapositivas son el único paso del proyecto que depende de LaTeX, y su
# ausencia no debe impedir reproducir los datos, los modelos y los resultados.
#
# Cada deck se compila con su propio motor: compare usa XeLaTeX, porque su
# plantilla carga la fuente Carlito con fontspec; best usa pdfLaTeX. Con
# TinyTeX, tinytex::latexmk() instala solo los paquetes de LaTeX que falten.
##########################################################

decks <- c(compare_equipo_03 = "xelatex", best_equipo_03 = "pdflatex")

if (!requireNamespace("tinytex", quietly = TRUE) ||
    Sys.which("xelatex") == "" || Sys.which("pdflatex") == "") {
  message("No se compilan las diapositivas: no se encontró LaTeX. ",
          "Instálelo con tinytex::install_tinytex() para compilarlas.")
} else {
  # La fuente Carlito no es un paquete de estilo, así que tinytex::latexmk()
  # no la instala por su cuenta: con TinyTeX se instala aquí si falta.
  carlito <- suppressWarnings(system2("kpsewhich", "Carlito-Regular.ttf",
                                      stdout = TRUE))
  if (length(carlito) == 0 && tinytex::is_tinytex()) {
    tinytex::tlmgr_install("carlito")
  }
  
  # LaTeX resuelve las rutas de los .tex (../02_outputs/...) desde la
  # carpeta en la que se compila; por eso se compila dentro de 04_slides/.
  # El bloque finally garantiza que se borren los auxiliares y se vuelva a la
  # raíz del proyecto aunque la compilación falle. Si un deck falla, se avisa
  # y se sigue con el otro.
  extensiones <- c("aux", "log", "nav", "snm", "toc", "out", "synctex.gz",
                   "fls", "fdb_latexmk", "vrb", "xdv")
  dir_raiz <- getwd()
  setwd("04_slides")
  tryCatch(
    for (deck in names(decks)) {
      tryCatch(
        tinytex::latexmk(paste0(deck, ".tex"), engine = decks[[deck]]),
        error = function(e) message("No se pudo compilar ", deck, ": ",
                                    conditionMessage(e))
      )
      unlink(paste0(deck, ".", extensiones))
    },
    finally = {
      unlink("texput.log")
      setwd(dir_raiz)
    }
  )
  
  rm(decks, deck, carlito, dir_raiz, extensiones)
}
