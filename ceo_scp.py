import contextlib
import io
import numpy as np

from CEO import CEO                 # SU archivo original, NO modificado
from binarization import binarize

# --------------------------------------------------------------------------
# Adaptador por HERENCIA: CEO.py no se edita en ningun punto.
# Se sobrescriben SOLO dos metodos:
#   binarizar() -> permite elegir la regla. Con RULE="ceo" se llama
#                  literalmente a CEO.binarizar de su archivo via super().
#   fitness()   -> binariza, maneja infactibilidad y evalua la Ec. (1), p. 3.
#
# La poblacion permanece CONTINUA en self.Population. La binarizacion actua
# solo al evaluar, nunca modifica los vectores almacenados.
#
# LIMITACION DECLARADA: "com", "static" y "elit_roulette" requieren el estado
# binario previo de cada solucion, que CEO.optimize() no entrega a fitness().
# Sin editar CEO.py se usa el ultimo bloque binarizado como x_current. Es una
# aproximacion operativa. "ceo", "std" y "elit" (la del paper) no la necesitan.
# --------------------------------------------------------------------------


class CEO_SCP(CEO):
    def __init__(self, inst, pop_size=10, chaos_samples=2, maxfes=6000,
                 varmin=-5.0, varmax=5.0, transfer="v3", rule="elit",
                 infeasible="repair", alpha=0.5, seed=None,
                 stagnation="paper"):
        # asignados ANTES de super() porque CEO.__init__ llama a self.fitness()
        self.inst = inst
        self._tf = transfer
        self._rule = rule
        self._mode = infeasible
        self._alpha = alpha
        self._rng = np.random.default_rng(seed)
        self._stagnation = stagnation
        self._last_bin = None
        self.n_evals = 0

        dim = inst.n
        super().__init__(func=self._cost, Np=pop_size, Dim=dim,
                         Varmin=varmin * np.ones((1, dim)),
                         Varmax=varmax * np.ones((1, dim)),
                         N=chaos_samples, MaxFES=maxfes,
                         tf=transfer, binario=True)

    def _cost(self, xb):
        return self.inst.cost(xb)

    # ---- override 3: criterio de termino ----
    # "paper": el Algoritmo 1 de Dong et al. (p. 6) solo corta con
    # FEvals >= MaxFES; devolver siempre False reproduce ese pseudocodigo.
    # "ceo_py": se delega en el metodo original de CEO.py sin cambios.
    def handle_stagnation(self, old, new):
        if self._stagnation == "paper":
            return False
        return super().handle_stagnation(old, new)

    # ---- override 1: regla de binarizacion ----
    def binarizar(self, x, tipo_transferencia=None, umbral_random=None):
        rule = self._rule

        # RULE="ceo": se ejecuta el metodo original de SU CEO.py, sin cambios.
        if rule == "ceo":
            xb = super().binarizar(np.atleast_2d(x), self._tf, umbral_random)
            self._last_bin = xb
            return xb

        tf = self._tf if tipo_transferencia is None else tipo_transferencia
        x = np.atleast_2d(np.asarray(x, dtype=float))
        x_best = getattr(self, "Best_bin", None)

        # en la primera llamada (constructor) aun no existe Best_bin
        if rule == "elit" and x_best is None:
            rule = "std"

        x_cur = None
        if rule in ("com", "static"):
            x_cur = (self._last_bin if self._last_bin is not None
                     and self._last_bin.shape == x.shape
                     else np.zeros(x.shape, dtype=np.int8))

        pop_bin = pop_fit = None
        if rule == "elit_roulette":
            pop_bin = getattr(self, "Population_bin", None)
            pop_fit = getattr(self, "fit", None)
            if pop_bin is None or pop_fit is None:
                rule = "std"

        xb = binarize(x, tf=tf, rule=rule, x_current=x_cur, x_best=x_best,
                      alpha=self._alpha, rng=self._rng,
                      pop_bin=pop_bin, pop_fit=pop_fit)
        self._last_bin = xb
        return xb

    # ---- override 2: evaluacion SCP ----
    def fitness(self, x):
        xb = np.asarray(self.binarizar(x, self._tf, None), dtype=np.int32)
        nrows = xb.shape[0]
        fit = np.zeros(nrows, dtype=float)
        sol = np.zeros_like(xb)
        for i in range(nrows):
            fit[i], sol[i, :] = self.inst.evaluate(xb[i, :], mode=self._mode)
        self.n_evals += nrows
        return fit, sol


def maxfes_for_iterations(pop_size, chaos_samples, iterations):
    """CEO gasta pop_size*chaos_samples evaluaciones por iteracion."""
    return int(pop_size + pop_size * chaos_samples * iterations)


def run_once(inst, **kw):
    """Una corrida. CEO.py imprime en cada evaluacion: se silencia stdout sin
    editar el archivo."""
    with contextlib.redirect_stdout(io.StringIO()):
        alg = CEO_SCP(inst, **kw)
        _, fbest, hist, best_bin = alg.optimize()
    x = np.asarray(best_bin, dtype=np.int32)
    return {
        "fitness": float(fbest),
        "cost": inst.cost(x),
        "feasible": bool(inst.is_feasible(x)),
        "n_cols": int(x.sum()),
        "iters": int(alg.t),
        "evals": int(alg.n_evals),
        "history": list(hist),
    }