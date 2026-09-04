# ##########################################################################
# ARCHIVO DE CONFIGURACIÓN
# Cada parametro indica de donde sale: PAPER / CEO / DECISION PROPIA.
# ##########################################################################

# ==========================================================================
# BLOQUE 1 - QUE EJECUTAR
# ==========================================================================

# Carpeta con los archivos scp*.txt de OR-Library.
DATA_DIR = "./instances"

# Variante del problema:
#   "SCP"  -> 22 instancias con costos (Tabla 6, p. 13). Referencia: Tablas 8-9.
#   "USCP" -> 17 instancias unicost (Tabla 7, p. 13). Referencia: Tabla 12.
#             Fuerza c_j = 1 (Seccion 2.3, p. 5) y requiere los archivos CLR/CYC
#             (ejecuta antes: python descargar_instancias.py).
PROBLEM = "USCP"

# Instancias a resolver. Tres formas de escribirlo:
#
#   INSTANCES = "all"                 -> todas las del PROBLEM elegido
#                                        (22 para SCP, 17 para USCP).
#   INSTANCES = ["61", "b1"]          -> solo esas dos. Prueba corta.
#   INSTANCES = ["nrg1"]              -> una sola instancia grande, para medir
#                                        el tiempo antes de lanzar todo.
#
# SCP  (minusculas): 41, 42, 51, 52, 61, 62, a1, a2, b1, b2, c1, c2, d1, d2,
#                    nre1, nre2, nrf1, nrf2, nrg1, nrg2, nrh1, nrh2
# USCP (mayusculas): U41, U51, U61, UA1, UB1, UC1, UD1, UNRE1, UNRF1, UNRG1,
#                    UNRH1, CLR10, CLR11, CLR12, CYC06, CYC07, CYC08
# (constantes SCP_22 y USCP_17 en paper_results.py).
INSTANCES = "all"

# Corridas independientes por instancia.
# PAPER: Tabla 5, p. 12 -> "Independent runs 31".
RUNS = 31


# ==========================================================================
# BLOQUE 2 - PARAMETROS FIJADOS POR EL PAPER (Mathematics 2025, 13, 2482)
# Cambiarlos rompe la comparabilidad con las tablas publicadas.
# ==========================================================================

# Tamano de poblacion. PAPER: Tabla 5, p. 12 -> "Population size 10".
# CEO exige que sea PAR (CEO.py lanza ValueError si Np es impar).
POP_SIZE = 10

# Iteraciones. PAPER: Tabla 5, p. 12 -> "Iterations 600".
ITERATIONS = 600

# Funcion de transferencia. PAPER: Ec. (16), p. 10 -> V3.
# Valores validos: "s1","s2","s3","s4","v1","v2","v3","v4"
#                  "ceo_v1","ceo_v2"  (nomenclatura invertida de su CEO.py)
TRANSFER = "v3"

# Regla de binarizacion.
#   "elit"          -> PAPER, Ec. (17), p. 10.  <-- configuracion del paper
#   "ceo"           -> llama literalmente a CEO.binarizar de SU archivo,
#                      sin pasar por binarization.py. Es la regla Standard.
#   "std"           -> Standard, Ec. (8), p. 6. Equivalente a "ceo".
#   "com"           -> Complement, Ec. (10), p. 6.
#   "static"        -> Static Probability (Crawford 2017; no esta en el paper SCP).
#   "elit_roulette" -> Elitist Roulette (interpretacion operativa, ver binarization.py).
RULE = "elit"


# ==========================================================================
# BLOQUE 3 - PARAMETROS PROPIOS DE CEO (no existen en el paper de SCP)
# ==========================================================================

# N = numero de muestras caoticas por par de individuos (parametro de CEO.py).
# CEO consume POP_SIZE * CHAOS_SAMPLES evaluaciones por iteracion.
CHAOS_SAMPLES = 2

# Dominio continuo [VARMIN, VARMAX] en el que vive la poblacion de CEO.
# DECISION PROPIA: el paper no fija este dominio. Es critico porque
# determina la escala de entrada de la funcion de transferencia y, con ello,
# cuantos bits se activan. Sensible: conviene reportarlo en el informe.
VARMIN = -5.0
VARMAX = 5.0

# Criterio de termino de cada corrida.
#   "paper"  -> solo "FEvals < MaxFES", tal como el Algoritmo 1, linea 5 del
#               paper de CEO (Dong et al. 2025, p. 6). El pseudocodigo NO
#               contiene corte por estancamiento. RECOMENDADO: agota el
#               presupuesto y hace comparable el esfuerzo con el paper de SCP.
#   "ceo_py" -> comportamiento tal cual del archivo CEO.py: corta tras 51
#               iteraciones consecutivas sin mejora. Es un agregado de la
#               implementacion, no aparece en ninguno de los dos papers.
# Ninguna de las dos opciones edita CEO.py: "paper" se logra sobrescribiendo
# handle_stagnation en la subclase CEO_SCP.
STAGNATION = "paper"

# Como igualar el esfuerzo computacional con el paper.
#   "fes"   -> igualar EVALUACIONES: MaxFES = POP_SIZE * ITERATIONS = 6000.
#              Comparacion justa. RECOMENDADO.
#   "iters" -> igualar ITERACIONES: CEO recibe CHAOS_SAMPLES veces mas
#              evaluaciones que el paper. NO es comparacion justa.
BUDGET = "fes"


# ==========================================================================
# BLOQUE 4 - DECISIONES QUE EL PAPER NO ESPECIFICA
# Verificado: los terminos "feasible", "infeasible", "repair" y "penalty"
# tienen CERO ocurrencias en las 28 paginas del paper.
# ==========================================================================

# Manejo de soluciones que violan la restriccion de cobertura, Ec. (2), p. 3.
#   "repair"  -> heuristica greedy costo/cobertura + eliminacion de columnas
#                redundantes. Garantiza factibilidad. DECISION PROPIA.
#   "penalty" -> fitness = costo + (sum(c)+1) * filas_descubiertas. DECISION PROPIA.
#   "raw"     -> sin manejo. Produce fitness por debajo del optimo. Solo sirve
#                para demostrar en el informe que el vacio del paper es real.
INFEASIBLE = "repair"

# UNICOST se deriva de PROBLEM y no se edita aqui.


# ==========================================================================
# BLOQUE 5 - CONTROL EXPERIMENTAL (recomendado, no obligatorio)
# ==========================================================================

# Ejecuta dos baselines sin CEO para medir cuanto aporta la metaheuristica
# por encima del reparador:
#   greedy        -> x=0 reparado (determinista, 1 evaluacion).
#   random+repair -> mismo presupuesto de evaluaciones, soluciones aleatorias.
# Si el baseline iguala a CEO, el merito no es de CEO. Debe reportarse.
RUN_BASELINE = True


# ==========================================================================
# BLOQUE 6 - SALIDA
# ==========================================================================

# Reanudacion: si existe <OUT_PREFIX>_parcial.csv con la MISMA configuracion,
# las corridas ya hechas se reutilizan y no se repiten. Ponga False para
# empezar de cero (conviene borrar tambien el archivo parcial).
RESUME = False

# Guarda la curva de convergencia de cada corrida en <OUT_PREFIX>_convergencia.csv
# (instancia, corrida, iteracion, mejor fitness). Es lo que necesitan las
# Figuras 2-5 del paper (pp. 15-16). Sin esto, las curvas se pierden.
SAVE_CONVERGENCE = True

SEED = 2025               # semilla base; la corrida r usa SEED + (r-1)
OUT_PREFIX = "ceo_scp"    # genera ceo_scp_raw.csv, ceo_scp_tabla.csv/.md/.tex
SHOW_PROGRESS = True      # una sola linea por instancia, sobrescrita en sitio