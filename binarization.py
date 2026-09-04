import math
import numpy as np

# --------------------------------------------------------------------------
# FUENTE: Crawford, B. et al. "Binary Secretary Bird Optimization Algorithm
# for the Set Covering Problem". Mathematics 2025, 13, 2482.
#   - Tabla 2 (pag. 6 de 28): 8 funciones de transferencia S1-S4 / V1-V4.
#   - Ecs. (8), (9), (10) (pag. 6 de 28): reglas STD, ELIT, COM.
#   - Ecs. (16) y (17) (pag. 10 de 28): configuracion adoptada = V3 + ELIT.
#
# ADVERTENCIA DE CONVENCION (no es error del codigo, es divergencia de fuentes):
#   La Tabla 2 del paper define  V1 = |erf(...)|  y  V2 = |tanh(d)|.
#   El archivo CEO.py del usuario define lo contrario: v1 = |tanh|, v2 = |erf|.
#   Este modulo sigue la convencion del PAPER. Si se desea replicar exactamente
#   los nombres de CEO.py, usar los alias "ceo_v1"/"ceo_v2" definidos abajo.
#   V3 y V4 coinciden en ambas fuentes, por lo que la configuracion del paper
#   (V3) no se ve afectada por esta divergencia.
#
# ADVERTENCIA DE ALCANCE:
#   Las reglas STATIC y ELIT_ROULETTE NO estan escritas en forma cerrada en el
#   paper de SCP (solo se menciona que existen cinco reglas). Sus expresiones
#   provienen de Crawford et al., Complexity 2017, Art. 8404231, Ecs. (17) y
#   descripcion verbal de Elitist Roulette. ELIT_ROULETTE es una INTERPRETACION
#   OPERATIVA: la fuente no fija como se calcula la probabilidad proporcional
#   bajo minimizacion. Marcado explicitamente en el codigo.
# --------------------------------------------------------------------------

_ERF = np.vectorize(math.erf)


def _clip(d):
    return np.clip(np.asarray(d, dtype=float), -700.0, 700.0)


TRANSFER = {
    # S-shaped, Tabla 2 (p. 6)
    "s1": lambda d: 1.0 / (1.0 + np.exp(-2.0 * _clip(d))),
    "s2": lambda d: 1.0 / (1.0 + np.exp(-_clip(d))),
    "s3": lambda d: 1.0 / (1.0 + np.exp(-_clip(d) / 2.0)),
    "s4": lambda d: 1.0 / (1.0 + np.exp(-_clip(d) / 3.0)),
    # V-shaped, Tabla 2 (p. 6)
    "v1": lambda d: np.abs(_ERF((math.sqrt(math.pi) / 2.0) * np.asarray(d, dtype=float))),
    "v2": lambda d: np.abs(np.tanh(np.asarray(d, dtype=float))),
    "v3": lambda d: np.abs(np.asarray(d, dtype=float) /
                           np.sqrt(1.0 + np.asarray(d, dtype=float) ** 2)),
    "v4": lambda d: np.abs((2.0 / math.pi) * np.arctan((math.pi / 2.0) * np.asarray(d, dtype=float))),
}

# Alias con la nomenclatura del archivo CEO.py del usuario (v1/v2 intercambiadas)
TRANSFER["ceo_v1"] = TRANSFER["v2"]
TRANSFER["ceo_v2"] = TRANSFER["v1"]

RULES = ("std", "elit", "com", "static", "elit_roulette")


def binarize(d, tf="v3", rule="elit", x_current=None, x_best=None,
             alpha=0.5, rng=None, pop_bin=None, pop_fit=None):
    """
    d        : array (filas, dim) de valores continuos producidos por la MH.
    tf       : clave de TRANSFER.
    rule     : una de RULES.
    x_current: array (dim,) o (filas, dim). Requerido por "com" y "static".
    x_best   : array (dim,). Requerido por "elit".
    pop_bin  : array (Np, dim). Requerido por "elit_roulette".
    pop_fit  : array (Np,). Requerido por "elit_roulette" (minimizacion).
    Retorna array int8 (filas, dim) con valores en {0,1}.
    """
    if tf not in TRANSFER:
        raise ValueError("tf desconocida: %s. Opciones: %s" % (tf, sorted(TRANSFER)))
    if rule not in RULES:
        raise ValueError("rule desconocida: %s. Opciones: %s" % (rule, RULES))

    d = np.atleast_2d(np.asarray(d, dtype=float))
    T = TRANSFER[tf](d)
    rng = np.random.default_rng() if rng is None else rng
    r = rng.random(d.shape)

    if rule == "std":
        # Ec. (8), p. 6
        out = (r <= T)

    elif rule == "elit":
        # Ec. (9) / Ec. (17), p. 6 y p. 10
        if x_best is None:
            raise ValueError("rule='elit' requiere x_best")
        xb = np.broadcast_to(np.asarray(x_best).reshape(1, -1), d.shape)
        out = np.where(r < T, xb, 0)

    elif rule == "com":
        # Ec. (10), p. 6
        if x_current is None:
            raise ValueError("rule='com' requiere x_current")
        xc = np.broadcast_to(np.asarray(x_current).reshape(-1, d.shape[1]), d.shape)
        out = np.where(r <= T, 1 - xc, 0)

    elif rule == "static":
        # Crawford et al. 2017, Ec. (17). No escrita en el paper de SCP.
        if x_current is None:
            raise ValueError("rule='static' requiere x_current")
        xc = np.broadcast_to(np.asarray(x_current).reshape(-1, d.shape[1]), d.shape)
        hi = 0.5 * (1.0 + alpha)
        out = np.where(T <= alpha, 0, np.where(T <= hi, xc, 1))

    else:  # elit_roulette
        # INTERPRETACION OPERATIVA (no especificada en forma cerrada en ninguna
        # de las dos fuentes): se elige un individuo de la poblacion por ruleta
        # con peso proporcional a (peor_fitness - fitness_i), es decir mejor
        # fitness => mayor probabilidad bajo minimizacion; luego se aplica la
        # misma estructura que ELIT usando los bits del individuo elegido.
        if pop_bin is None or pop_fit is None:
            raise ValueError("rule='elit_roulette' requiere pop_bin y pop_fit")
        f = np.asarray(pop_fit, dtype=float)
        w = (f.max() - f) + 1e-12
        w = w / w.sum()
        idx = rng.choice(len(f), size=d.shape[0], p=w)
        xb = np.asarray(pop_bin)[idx, :]
        out = np.where(r < T, xb, 0)

    return out.astype(np.int8)