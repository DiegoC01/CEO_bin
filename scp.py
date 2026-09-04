import os
import numpy as np

try:
    from scipy import sparse as sp
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# --------------------------------------------------------------------------
# Modelo: Mathematics 2025, 13, 2482, Seccion 2.1 (p. 3)
#   Ec. (1)  min  sum_j c_j x_j
#   Ec. (2)  s.a. sum_j a_ij x_j >= 1  para toda fila i
#            x_j en {0,1}
# Unicost: Seccion 2.3 (p. 5), Ecs. (5)-(7), c_j = 1.
#
# Formato de archivo OR-Library (el paper solo cita la libreria, ref. [37],
# p. 11; el formato es conocimiento externo):
#   m n | n costos | por cada fila: cantidad de columnas y sus indices base 1
# --------------------------------------------------------------------------


class SCPInstance:
    def __init__(self, name, m, n, rows, cols, costs, unicost=False):
        self.name = name
        self.m, self.n = int(m), int(n)
        self.unicost = unicost
        self.c = np.ones(self.n) if unicost else np.asarray(costs, dtype=float)
        data = np.ones(len(rows), dtype=np.int32)
        if _HAS_SCIPY:
            self.A = sp.csr_matrix((data, (rows, cols)), shape=(self.m, self.n), dtype=np.int32)
            self.At = self.A.T.tocsr()          # columnas -> filas, para gains
        else:
            self.A = np.zeros((self.m, self.n), dtype=np.int32)
            self.A[np.asarray(rows), np.asarray(cols)] = 1
            self.At = self.A.T
        self.density = 100.0 * len(rows) / float(self.m * self.n)
        self.penalty = float(self.c.sum()) + 1.0

    # ---------------- evaluacion ----------------
    def coverage(self, x):
        return self.A.dot(np.asarray(x, dtype=np.int32))

    def cost(self, x):
        return float(self.c.dot(np.asarray(x, dtype=float)))

    def deficit(self, x):
        return int(np.count_nonzero(self.coverage(x) == 0))

    def is_feasible(self, x):
        return self.deficit(x) == 0

    def evaluate(self, x, mode="repair"):
        x = np.asarray(x, dtype=np.int32).copy()
        if mode == "repair":
            x = self.repair(x)
            return self.cost(x), x
        if mode == "penalty":
            return self.cost(x) + self.penalty * self.deficit(x), x
        if mode == "raw":
            return self.cost(x), x
        raise ValueError("INFEASIBLE invalido: %s" % mode)

    # ---------------- reparacion (EXTERNA AL PAPER) ----------------
    # Greedy de razon costo/cobertura (Chvatal, Math. Oper. Res. 4(3), 1979)
    # con actualizacion incremental de ganancias, mas eliminacion de columnas
    # redundantes. El paper de SCP no describe ningun mecanismo equivalente.
    def repair(self, x):
        x = np.asarray(x, dtype=np.int32).copy()
        cov = self.coverage(x)
        unc = (cov == 0)
        if unc.any():
            gain = self._gain(unc)
            while unc.any():
                pos = gain > 0
                if not pos.any():
                    break                       # fila imposible de cubrir
                ratio = np.where(pos, self.c / np.maximum(gain, 1), np.inf)
                j = int(np.argmin(ratio))
                x[j] = 1
                rows_j = self._col(j)
                cov[rows_j] += 1
                newly = rows_j[unc[rows_j]]
                if newly.size == 0:
                    gain[j] = 0
                    continue
                unc[newly] = False
                idx = np.concatenate([self._row(r) for r in newly])
                gain -= np.bincount(idx, minlength=self.n)
        return self._drop_redundant(x, cov)

    def _gain(self, unc):
        v = unc.astype(np.int32)
        return np.asarray(self.At.dot(v)).ravel().astype(np.int64)

    def _col(self, j):
        if _HAS_SCIPY:
            return self.At.indices[self.At.indptr[j]:self.At.indptr[j + 1]]
        return np.flatnonzero(self.A[:, j] == 1)

    def _row(self, i):
        if _HAS_SCIPY:
            return self.A.indices[self.A.indptr[i]:self.A.indptr[i + 1]]
        return np.flatnonzero(self.A[i, :] == 1)

    def _drop_redundant(self, x, cov=None):
        sel = np.flatnonzero(x == 1)
        if sel.size == 0:
            return x
        if cov is None:
            cov = self.coverage(x)
        for j in sel[np.argsort(-self.c[sel])]:   # primero las columnas caras
            rows = self._col(j)
            if rows.size == 0 or np.all(cov[rows] >= 2):
                x[j] = 0
                cov[rows] -= 1
        return x


# ---------------- lectura ----------------
# El formato estandar de OR-Library incluye la lista de n costos despues de
# "m n". Los archivos unicost aportados por A. Wool (scpclr*, scpcyc*) pueden
# venir SIN esa seccion. load_orlib prueba primero con costos y, si el conteo
# de tokens no cuadra exactamente, reintenta asumiendo costo unitario. Si
# ninguna de las dos variantes cuadra, lanza error en vez de leer basura.
def _parse(tok, con_costos):
    m, n = int(tok[0]), int(tok[1])
    p = 2
    if con_costos:
        if p + n > len(tok):
            raise ValueError("faltan tokens para los %d costos" % n)
        costs = np.array(tok[p:p + n], dtype=float)
        p += n
    else:
        costs = np.ones(n, dtype=float)
    rows, cols = [], []
    for i in range(m):
        if p >= len(tok):
            raise ValueError("archivo truncado en la fila %d de %d" % (i + 1, m))
        k = int(tok[p]); p += 1
        if k < 0 or p + k > len(tok):
            raise ValueError("cardinalidad invalida en la fila %d" % (i + 1))
        idx = [int(t) - 1 for t in tok[p:p + k]]
        if any(j < 0 or j >= n for j in idx):
            raise ValueError("indice de columna fuera de rango en la fila %d" % (i + 1))
        rows.extend([i] * k); cols.extend(idx); p += k
    if p != len(tok):
        raise ValueError("sobran %d tokens al final" % (len(tok) - p))
    return m, n, rows, cols, costs


def load_orlib(path, name=None, unicost=False):
    with open(path, "r") as fh:
        tok = fh.read().split()
    err = None
    for con_costos in (True, False):
        try:
            m, n, rows, cols, costs = _parse(tok, con_costos)
            break
        except (ValueError, IndexError) as e:
            err = e
    else:
        raise ValueError("no se pudo interpretar %s: %s" % (path, err))
    return SCPInstance(name or os.path.basename(path), m, n, rows, cols, costs, unicost)


# ---------------- baselines de control (sin CEO) ----------------
def greedy_solution(inst):
    """x = 0 reparado. Determinista. Cuesta 1 evaluacion."""
    x = inst.repair(np.zeros(inst.n, dtype=np.int32))
    return inst.cost(x), x


def random_repair_best(inst, budget, seed=0, p_max=0.05):
    """Mejor de `budget` soluciones aleatorias reparadas. Mide cuanto del
    resultado se explica solo por el reparador, sin metaheuristica."""
    rng = np.random.default_rng(seed)
    best, bx = np.inf, None
    for _ in range(int(budget)):
        p = rng.uniform(0.0, p_max)
        x = (rng.random(inst.n) < p).astype(np.int32)
        x = inst.repair(x)
        f = inst.cost(x)
        if f < best:
            best, bx = f, x
    return best, bx