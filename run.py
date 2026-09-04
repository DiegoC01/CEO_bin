import csv
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

import config as cfg
import paper_results as pr
import scp as scp_mod
import ceo_scp as ceo_mod
from scp import load_orlib, greedy_solution, random_repair_best

try:
    from ceo_scp import run_once, maxfes_for_iterations
except ImportError as _e:
    print("=" * 70)
    print("ceo_scp.py esta desactualizado: %s" % _e)
    print("Reemplazalo por la version actual y vuelve a ejecutar.")
    print("=" * 70)
    raise SystemExit(1)

# --------------------------------------------------------------------------
# ARCHIVOS QUE SE GENERAN
#   <pre>_tabla.txt      LA TABLA COMPARATIVA, en texto legible.
#                        Se reescribe DESPUES DE CADA CORRIDA.
#   <pre>_tabla.md/.tex/.csv   mismas cifras en otros formatos, tras cada instancia.
#   <pre>_bitacora.csv   una fila por corrida. Sirve para reanudar, no para leer.
# --------------------------------------------------------------------------

BIT_COLS = ["timestamp", "instance", "run", "fitness", "feasible", "n_cols",
            "iters", "evals", "time_s", "problem", "transfer", "rule",
            "infeasible", "varmin", "varmax", "pop", "chaos", "maxfes",
            "stagnation", "seed"]

# Campos que definen una configuracion. Una corrida guardada solo se reutiliza
# si TODOS coinciden con la configuracion actual. Sin esto, un cambio de
# STAGNATION o de MaxFES devolveria resultados viejos con etiqueta nueva.
CFG_COLS = ["problem", "transfer", "rule", "infeasible", "varmin", "varmax",
            "pop", "chaos", "maxfes", "stagnation"]

W = 100


def es_uscp():
    return getattr(cfg, "PROBLEM", "SCP").upper() == "USCP"


def lista_instancias():
    if cfg.INSTANCES == "all":
        return list(pr.USCP_17) if es_uscp() else list(pr.SCP_22)
    return [t if es_uscp() else t.lower() for t in cfg.INSTANCES]


def conv_path():
    return cfg.OUT_PREFIX + "_convergencia.csv"


def verificar_modulos():
    """Avisa si algun archivo del paquete quedo en una version anterior, en vez
    de reventar mas adelante con un TypeError sin contexto."""
    import inspect
    faltan = []
    if "stagnation" not in inspect.signature(ceo_mod.CEO_SCP.__init__).parameters:
        faltan.append("ceo_scp.py  (le falta el parametro stagnation)")
    for attr in ("PROBLEM", "STAGNATION", "SAVE_CONVERGENCE"):
        if not hasattr(cfg, attr):
            faltan.append("config.py   (le falta %s)" % attr)
            break
    for attr in ("info", "ref_table", "source_file", "USCP_17", "PAPER_USCP"):
        if not hasattr(pr, attr):
            faltan.append("paper_results.py  (le falta %s)" % attr)
            break
    if not hasattr(scp_mod, "_parse"):
        faltan.append("scp.py  (lector antiguo, no lee archivos CLR/CYC)")
    if faltan:
        print("=" * 70)
        print("ARCHIVOS DESACTUALIZADOS. Reemplaza estos y vuelve a ejecutar:")
        for f in faltan:
            print("  - %s" % f)
        print("=" * 70)
        sys.exit(1)


def budget():
    if cfg.BUDGET == "fes":
        maxfes = cfg.POP_SIZE * cfg.ITERATIONS
    else:
        maxfes = maxfes_for_iterations(cfg.POP_SIZE, cfg.CHAOS_SAMPLES, cfg.ITERATIONS)
    per_iter = cfg.POP_SIZE * cfg.CHAOS_SAMPLES
    iters_max = int(math.floor((maxfes - cfg.POP_SIZE - 1) / per_iter)) + 1
    return maxfes, iters_max


# ------------------------- bitacora / reanudacion -------------------------
def bitacora_path():
    return cfg.OUT_PREFIX + "_bitacora.csv"


def txt_path():
    return cfg.OUT_PREFIX + "_tabla.txt"


def config_actual(maxfes):
    return {"problem": getattr(cfg, "PROBLEM", "SCP").upper(),
            "transfer": cfg.TRANSFER, "rule": cfg.RULE,
            "infeasible": cfg.INFEASIBLE,
            "varmin": cfg.VARMIN, "varmax": cfg.VARMAX,
            "pop": cfg.POP_SIZE, "chaos": cfg.CHAOS_SAMPLES,
            "maxfes": maxfes,
            "stagnation": getattr(cfg, "STAGNATION", "paper")}


def formato_actual(path):
    with open(path, newline="") as fh:
        cab = fh.readline().strip().replace("\ufeff", "").split(",")
    return [c.strip() for c in cab] == BIT_COLS


def _igual(a, b):
    """Compara valores de configuracion tolerando texto frente a numero."""
    try:
        return abs(float(a) - float(b)) < 1e-12
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def load_bitacora(maxfes):
    done, descartadas = {}, 0
    path = bitacora_path()
    if not os.path.exists(path):
        return done
    if not formato_actual(path):
        print("Bitacora con formato antiguo: NO se reutiliza. Le falta el campo")
        print("stagnation, por lo que su configuracion no se puede verificar.")
        return done
    if not getattr(cfg, "RESUME", True):
        print("RESUME=False: no se reutiliza ninguna corrida guardada.")
        return done
    actual = config_actual(maxfes)
    with open(path, newline="") as fh:
        for rec in csv.DictReader(fh):
            if all(_igual(rec.get(k), actual[k]) for k in CFG_COLS):
                done[(rec["instance"], int(rec["run"]))] = rec
            else:
                descartadas += 1
    print("Bitacora: %d corridas reutilizables, %d descartadas por otra configuracion."
          % (len(done), descartadas))
    return done


def open_convergencia():
    if not getattr(cfg, "SAVE_CONVERGENCE", True):
        return None, None
    path = conv_path()
    nuevo = not os.path.exists(path)
    fh = open(path, "a", newline="")
    w = csv.writer(fh)
    if nuevo:
        w.writerow(["instance", "run", "iter", "fbest"]); fh.flush()
    return fh, w


def save_curve(fh, w, tag, r, history):
    if w is None:
        return
    for i, v in enumerate(history, 1):
        w.writerow([tag, r, i, v])
    fh.flush()


def open_bitacora():
    path = bitacora_path()
    if os.path.exists(path) and not formato_actual(path):
        n, bak = 1, path + ".v1.bak"
        while os.path.exists(bak):
            n += 1
            bak = path + ".v%d.bak" % n
        os.rename(path, bak)
        print("Bitacora antigua movida a %s" % bak)
    nuevo = not os.path.exists(path)
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=BIT_COLS)
    if nuevo:
        w.writeheader(); fh.flush()
    return fh, w


def save_run(fh, w, tag, r, res, dt, maxfes):
    fila = {"timestamp": datetime.now().isoformat(timespec="seconds"),
            "instance": tag, "run": r, "fitness": res["fitness"],
            "feasible": int(res["feasible"]), "n_cols": res["n_cols"],
            "iters": res["iters"], "evals": res["evals"], "time_s": round(dt, 3),
            "seed": cfg.SEED + r - 1}
    fila.update(config_actual(maxfes))
    w.writerow(fila)
    fh.flush()
    os.fsync(fh.fileno())


# ------------------------- construccion de filas -------------------------
def make_row(tag, inst, fits, feas, iters, evals, times, con_baseline, maxfes):
    f = np.asarray(fits, dtype=float)
    inf = pr.info(tag)
    row = {"instance": tag, "m": inst.m, "n": inst.n,
           "opt": inf[3] if inf else None,
           "runs_done": len(fits),
           "ceo_best": f.min(), "ceo_worst": f.max(),
           "ceo_avg": round(float(f.mean()), 3),
           "ceo_std": round(float(f.std(ddof=0)), 3),
           "ceo_pct_feasible": round(100.0 * float(np.mean(feas)), 1),
           "ceo_avg_iters": round(float(np.mean(iters)), 1),
           "ceo_avg_evals": round(float(np.mean(evals)), 1),
           "ceo_avg_time_s": round(float(np.mean(times)), 2)}
    row["ceo_rpd_best"] = (round(pr.rpd(row["ceo_best"], row["opt"]), 3)
                           if row["opt"] else None)
    if con_baseline and cfg.RUN_BASELINE:
        g, _ = greedy_solution(inst)
        rr, _ = random_repair_best(inst, budget=min(maxfes, 500), seed=cfg.SEED)
        row["baseline_greedy"], row["baseline_rand_repair"] = g, rr
    else:
        row["baseline_greedy"] = row["baseline_rand_repair"] = None
    ref = pr.ref_table(tag)
    for mh in ("SBOA", "GWO", "PSO"):
        b, wo, a, s = ref.get(mh, (None,) * 4)
        k = mh.lower()
        row[k + "_best"], row[k + "_worst"] = b, wo
        row[k + "_avg"], row[k + "_std"] = a, s
    return row


# ------------------------- render de la tabla -------------------------
def _f(v, d=0):
    return "-" if v is None else ("%.*f" % (d, v))


def render(rows, maxfes, iters_max, estado=""):
    L = []
    A = L.append
    A("=" * W)
    A("TABLA COMPARATIVA  -  CEO binarizado vs. resultados publicados")
    A("Problema: %s" % getattr(cfg, "PROBLEM", "SCP").upper())
    A("Fuente de SBOA, GWO y PSO: Mathematics 2025, 13, 2482, %s"
      % ("Tabla 12 (pp. 18-19)" if es_uscp() else "Tablas 8 y 9 (pp. 13-15)"))
    A("Actualizado: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if estado:
        A("Estado: %s" % estado)
    A("CEO: tf=%s  regla=%s  infactibles=%s  dominio=[%g,%g]  Np=%d  N=%d  MaxFES=%d  corridas=%d"
      % (cfg.TRANSFER, cfg.RULE, cfg.INFEASIBLE, cfg.VARMIN, cfg.VARMAX,
         cfg.POP_SIZE, cfg.CHAOS_SAMPLES, maxfes, cfg.RUNS))
    A("Criterio de termino: %s" % getattr(cfg, "STAGNATION", "paper"))
    A("=" * W)
    A("")
    A("TABLA 1. Mejor valor y promedio por instancia")
    A("-" * W)
    A("%-6s %5s %4s | %7s %6s %6s %6s | %9s %9s %9s %9s | %7s"
      % ("Inst", "Opt", "n", "CEO", "SBOA", "GWO", "PSO",
         "CEO", "SBOA", "GWO", "PSO", "RPD CEO"))
    A("%-6s %5s %4s | %s | %s | %7s"
      % ("", "", "", "Mejor valor".center(28), "Promedio".center(39), "%"))
    A("-" * W)
    for r in rows:
        mk = "" if r["runs_done"] == cfg.RUNS else "*"
        A("%-5s%1s %5s %4d | %7s %6s %6s %6s | %9s %9s %9s %9s | %7s"
          % (r["instance"].upper(), mk, r["opt"] or "-", r["runs_done"],
             _f(r["ceo_best"]), _f(r["sboa_best"]), _f(r["gwo_best"]), _f(r["pso_best"]),
             _f(r["ceo_avg"], 3), _f(r["sboa_avg"], 3),
             _f(r["gwo_avg"], 3), _f(r["pso_avg"], 3), _f(r["ceo_rpd_best"], 3)))
    A("-" * W)
    A("n = corridas completadas.  * = instancia aun incompleta.")
    A("RPD = 100 (CEO_best - Opt) / Opt.")
    A("")
    A("TABLA 2. Peor valor y desviacion estandar")
    A("-" * W)
    A("%-6s | %7s %6s %6s %6s | %9s %9s %9s %9s"
      % ("Inst", "CEO", "SBOA", "GWO", "PSO", "CEO", "SBOA", "GWO", "PSO"))
    A("%-6s | %s | %s" % ("", "Peor valor".center(28), "Desviacion estandar".center(39)))
    A("-" * W)
    for r in rows:
        A("%-6s | %7s %6s %6s %6s | %9s %9s %9s %9s"
          % (r["instance"].upper(),
             _f(r["ceo_worst"]), _f(r["sboa_worst"]), _f(r["gwo_worst"]), _f(r["pso_worst"]),
             _f(r["ceo_std"], 3), _f(r["sboa_std"], 3),
             _f(r["gwo_std"], 3), _f(r["pso_std"], 3)))
    A("-" * W)
    A("")
    A("TABLA 3. Detalle de la ejecucion de CEO")
    A("-" * W)
    A("%-6s %6s %6s %10s %11s %12s %11s"
      % ("Inst", "m", "n", "%factible", "iter/corr", "evals/corr", "seg/corr"))
    A("-" * W)
    for r in rows:
        A("%-6s %6d %6d %10.1f %11.1f %12.0f %11.2f"
          % (r["instance"].upper(), r["m"], r["n"], r["ceo_pct_feasible"],
             r["ceo_avg_iters"], r["ceo_avg_evals"], r["ceo_avg_time_s"]))
    A("-" * W)

    if cfg.RUN_BASELINE and any(r["baseline_greedy"] is not None for r in rows):
        A("")
        A("TABLA 4. Control sin metaheuristica")
        A("-" * W)
        A("%-6s %9s %13s %19s %9s"
          % ("Inst", "CEO best", "greedy puro", "aleatorio+reparar", "aporte"))
        A("-" * W)
        for r in rows:
            g, rr = r["baseline_greedy"], r["baseline_rand_repair"]
            if g is None or r["ceo_best"] is None:
                continue
            A("%-6s %9.0f %13.0f %19.0f %9.0f"
              % (r["instance"].upper(), r["ceo_best"], g, rr, min(g, rr) - r["ceo_best"]))
        A("-" * W)
        A("aporte = (mejor baseline) - (CEO best). Si es <= 0, CEO no supera al reparador")
        A("y el resultado no puede atribuirse a la metaheuristica.")

    low = [r for r in rows if r["ceo_avg_iters"] < 0.9 * iters_max]
    if low:
        A("")
        A("AVISO DE PRESUPUESTO")
        A("-" * W)
        A("Presupuesto teorico: %d iteraciones (MaxFES=%d)." % (iters_max, maxfes))
        for r in low:
            A("  %-6s %7.1f iteraciones reales, %8.0f evaluaciones de %d"
              % (r["instance"].upper(), r["ceo_avg_iters"], r["ceo_avg_evals"], maxfes))
        if getattr(cfg, "STAGNATION", "paper") == "ceo_py":
            A("Causa: CEO.handle_stagnation corta tras 51 iteraciones consecutivas sin")
            A("mejora. Ese corte NO aparece en el Algoritmo 1 de Dong et al. (p. 6), que")
            A("solo termina con FEvals >= MaxFES. Con STAGNATION=\"paper\" en config.py se")
            A("agota el presupuesto completo.")
        else:
            A("El presupuesto no se agoto pese a STAGNATION=paper. Revise MaxFES y Np.")
    return "\n".join(L) + "\n"


def write_txt(rows, maxfes, iters_max, estado=""):
    if not rows:
        return
    with open(txt_path(), "w") as fh:
        fh.write(render(rows, maxfes, iters_max, estado))
        fh.flush()
        os.fsync(fh.fileno())


# ------------------------- otros formatos -------------------------
def _tex(v, d=0, best=False):
    if v is None:
        return "--"
    s = "%.*f" % (d, v)
    return "\\textbf{%s}" % s if best else s


def _winners(row, suf, d):
    ok = {k: row.get(k + suf) for k in ("ceo", "sboa", "gwo", "pso")}
    ok = {k: v for k, v in ok.items() if v is not None}
    if not ok:
        return set()
    mn = round(min(ok.values()), d)
    return {k for k, v in ok.items() if round(v, d) == mn}


def write_files(rows, maxfes):
    pre = cfg.OUT_PREFIX
    with open(pre + "_tabla.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    with open(pre + "_tabla.md", "w") as fh:
        fh.write("| Inst | Opt | CEO best | SBOA best | GWO best | PSO best | "
                 "CEO avg | SBOA avg | GWO avg | PSO avg | RPD CEO (%) |\n")
        fh.write("|" + "---|" * 11 + "\n")
        for r in rows:
            fh.write("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n"
                     % (r["instance"].upper(), r["opt"],
                        _f(r["ceo_best"]), _f(r["sboa_best"]), _f(r["gwo_best"]), _f(r["pso_best"]),
                        _f(r["ceo_avg"], 3), _f(r["sboa_avg"], 3),
                        _f(r["gwo_avg"], 3), _f(r["pso_avg"], 3), _f(r["ceo_rpd_best"], 3)))
    _write_tex(rows, maxfes, pre)


def _write_tex(rows, maxfes, pre):
    cap = ("Comparacion del algoritmo CEO binarizado con los resultados publicados por "
           "Crawford et al. (2025) sobre instancias SCP de OR-Library. Configuracion de "
           "CEO: funcion de transferencia %s, regla de binarizacion %s, manejo de "
           "infactibilidad %s, dominio continuo $[%g,%g]$, $N_p=%d$, $N=%d$, MaxFES $=%d$, "
           "%d corridas independientes. Los valores de SBOA, GWO y PSO son transcripcion de "
           "las Tablas 8 y 9 de dicho trabajo y provienen de un entorno experimental "
           "distinto, por lo que la comparacion es contra valores reportados en la "
           "literatura y no un experimento controlado."
           % (cfg.TRANSFER.upper(), cfg.RULE.upper(), cfg.INFEASIBLE, cfg.VARMIN,
              cfg.VARMAX, cfg.POP_SIZE, cfg.CHAOS_SAMPLES, maxfes, cfg.RUNS))
    L = ["% Requiere \\usepackage{booktabs}", "\\begin{table}[htbp]", "\\centering",
         "\\footnotesize", "\\caption{%s}" % cap, "\\label{tab:ceo-scp}",
         "\\begin{tabular}{lrrrrrrrrrr}", "\\toprule",
         " & & \\multicolumn{4}{c}{Mejor valor} & \\multicolumn{4}{c}{Promedio} & \\\\",
         "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}",
         "Inst. & Opt. & CEO & SBOA & GWO & PSO & CEO & SBOA & GWO & PSO & RPD (\\%) \\\\",
         "\\midrule"]
    for r in rows:
        wb, wa = _winners(r, "_best", 0), _winners(r, "_avg", 3)
        mk = "" if r["runs_done"] == cfg.RUNS else "$^{*}$"
        L.append("%s%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\"
                 % (r["instance"].upper(), mk, r["opt"],
                    _tex(r["ceo_best"], 0, "ceo" in wb), _tex(r["sboa_best"], 0, "sboa" in wb),
                    _tex(r["gwo_best"], 0, "gwo" in wb), _tex(r["pso_best"], 0, "pso" in wb),
                    _tex(r["ceo_avg"], 3, "ceo" in wa), _tex(r["sboa_avg"], 3, "sboa" in wa),
                    _tex(r["gwo_avg"], 3, "gwo" in wa), _tex(r["pso_avg"], 3, "pso" in wa),
                    _tex(r["ceo_rpd_best"], 3)))
    L += ["\\bottomrule", "\\end{tabular}", "", "\\vspace{2pt}",
          "\\begin{minipage}{\\textwidth}\\scriptsize",
          "En negrita el mejor valor de cada fila. "
          "RPD $=100\\,(\\text{CEO}_{best}-\\text{Opt})/\\text{Opt}$."]
    if any(r["runs_done"] != cfg.RUNS for r in rows):
        L.append("$^{*}$ Instancia con menos de %d corridas: resultado parcial." % cfg.RUNS)
    L += ["\\end{minipage}", "\\end{table}", "",
          "\\begin{table}[htbp]", "\\centering", "\\footnotesize",
          "\\caption{Dispersion de los resultados: peor valor y desviacion estandar, para "
          "la misma configuracion de la Tabla~\\ref{tab:ceo-scp}.}",
          "\\label{tab:ceo-scp-disp}", "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
          " & \\multicolumn{4}{c}{Peor valor} & \\multicolumn{4}{c}{Desv. estandar} \\\\",
          "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}",
          "Inst. & CEO & SBOA & GWO & PSO & CEO & SBOA & GWO & PSO \\\\", "\\midrule"]
    for r in rows:
        L.append("%s & %s & %s & %s & %s & %s & %s & %s & %s \\\\"
                 % (r["instance"].upper(),
                    _tex(r["ceo_worst"]), _tex(r["sboa_worst"]),
                    _tex(r["gwo_worst"]), _tex(r["pso_worst"]),
                    _tex(r["ceo_std"], 3), _tex(r["sboa_std"], 3),
                    _tex(r["gwo_std"], 3), _tex(r["pso_std"], 3)))
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    with open(pre + "_tabla.tex", "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(pre + "_preview.tex", "w") as fh:
        fh.write("\\documentclass[10pt]{article}\n"
                 "\\usepackage[a4paper,margin=2cm]{geometry}\n\\usepackage{booktabs}\n"
                 "\\usepackage[T1]{fontenc}\n\\pagestyle{empty}\n\\begin{document}\n"
                 "\\input{%s_tabla.tex}\n\\end{document}\n" % pre)


# ------------------------- ejecucion -------------------------
def progress(tag, r, total, t0, reused):
    if not cfg.SHOW_PROGRESS:
        return
    el = time.time() - t0
    hechas = max(r - reused, 1)
    sys.stdout.write("\r  %-6s %3d/%-3d  transcurrido %5.0fs  restante ~%6.0fs      "
                     % (tag.upper(), r, total, el, el / hechas * (total - r)))
    sys.stdout.flush()


def solve_instance(tag, maxfes, iters_max, done, fh, w, rows_previas,
                   cfh=None, cw=None, nuevas=None):
    path = os.path.join(cfg.DATA_DIR, "scp%s.txt" % pr.source_file(tag))
    if not os.path.exists(path):
        print("  [FALTA] %s   (ejecuta: python descargar_instancias.py)" % path)
        return None
    inst = load_orlib(path, name=tag, unicost=es_uscp())

    fits, feas, iters, evals, times = [], [], [], [], []
    reused = 0
    t0 = time.time()
    for r in range(1, cfg.RUNS + 1):
        rec = done.get((tag, r))
        if rec is not None:
            fits.append(float(rec["fitness"])); feas.append(bool(int(rec["feasible"])))
            iters.append(float(rec["iters"])); evals.append(float(rec["evals"]))
            times.append(float(rec["time_s"])); reused += 1
        else:
            np.random.seed(cfg.SEED + r - 1)     # CEO.py usa el generador global
            t1 = time.time()
            res = run_once(inst, pop_size=cfg.POP_SIZE, chaos_samples=cfg.CHAOS_SAMPLES,
                           maxfes=maxfes, varmin=cfg.VARMIN, varmax=cfg.VARMAX,
                           transfer=cfg.TRANSFER, rule=cfg.RULE,
                           infeasible=cfg.INFEASIBLE, seed=cfg.SEED + r - 1,
                           stagnation=getattr(cfg, "STAGNATION", "paper"))
            dt = time.time() - t1
            save_run(fh, w, tag, r, res, dt, maxfes)
            save_curve(cfh, cw, tag, r, res["history"])
            if nuevas is not None:
                nuevas[0] += 1
            fits.append(res["fitness"]); feas.append(res["feasible"])
            iters.append(res["iters"]); evals.append(res["evals"]); times.append(dt)
        progress(tag, r, cfg.RUNS, t0, reused)
        # ---- LA TABLA SE REESCRIBE AQUI, TRAS CADA CORRIDA ----
        parcial = make_row(tag, inst, fits, feas, iters, evals, times, False, maxfes)
        write_txt(rows_previas + [parcial], maxfes, iters_max,
                  estado="ejecutando %s, corrida %d de %d" % (tag.upper(), r, cfg.RUNS))
    if cfg.SHOW_PROGRESS:
        sys.stdout.write("\r" + " " * 74 + "\r")
    return make_row(tag, inst, fits, feas, iters, evals, times, True, maxfes)


def main():
    verificar_modulos()
    maxfes, iters_max = budget()
    tags = lista_instancias()
    print("Problema: %s   |   Instancias: %s"
          % (getattr(cfg, "PROBLEM", "SCP").upper(), ", ".join(t.upper() for t in tags)))
    print("Corridas: %d | MaxFES: %d (max %d iteraciones)" % (cfg.RUNS, maxfes, iters_max))
    print("TABLA EN VIVO -> %s   (se reescribe tras cada corrida)" % txt_path())
    print("Bitacora      -> %s" % bitacora_path())
    if getattr(cfg, "SAVE_CONVERGENCE", True):
        print("Convergencia  -> %s" % conv_path())
    print("")

    done = load_bitacora(maxfes)
    fh, w = open_bitacora()
    cfh, cw = open_convergencia()
    rows, nuevas = [], [0]
    try:
        for tag in tags:
            row = solve_instance(tag, maxfes, iters_max, done, fh, w, rows,
                                 cfh, cw, nuevas)
            if row is None:
                continue
            rows.append(row)
            write_txt(rows, maxfes, iters_max, estado="%s completada" % tag.upper())
            write_files(rows, maxfes)
            print("  %-6s listo   best=%.0f   avg=%.3f" % (tag.upper(), row["ceo_best"], row["ceo_avg"]))
    except KeyboardInterrupt:
        print("\nInterrumpido. La tabla con lo completado esta en %s" % txt_path())
    finally:
        fh.close()
        if cfh is not None:
            cfh.close()

    if not rows:
        print("No se resolvio ninguna instancia. Revise DATA_DIR en config.py.")
        return
    if nuevas[0] == 0:
        print("\nATENCION: no se ejecuto ninguna corrida nueva. Toda la tabla proviene")
        print("de %s. Si esperabas una ejecucion nueva, cambia OUT_PREFIX" % bitacora_path())
        print("o pon RESUME = False en config.py.")
    write_txt(rows, maxfes, iters_max, estado="ejecucion finalizada")
    write_files(rows, maxfes)
    print(render(rows, maxfes, iters_max, estado="ejecucion finalizada"))
    print("Archivos: %s_tabla.txt | .md | .tex | .csv | %s_bitacora.csv"
          % (cfg.OUT_PREFIX, cfg.OUT_PREFIX))


if __name__ == "__main__":
    main()