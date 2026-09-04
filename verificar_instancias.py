import os
import paper_results as pr
from scp import load_orlib
import config as cfg

# Comprueba que cada archivo scp*.txt necesario existe, se lee sin error y
# tiene las dimensiones que declara el paper (Tablas 6 y 7, p. 13).

def revisar(tag, m_esp, n_esp, dens_esp, etiqueta):
    path = os.path.join(cfg.DATA_DIR, "scp%s.txt" % tag)
    if not os.path.exists(path):
        return "FALTA", "%s no existe" % path
    try:
        inst = load_orlib(path, name=tag)
    except Exception as e:
        return "ERROR", "%s: %s" % (type(e).__name__, e)
    if inst.m != m_esp or inst.n != n_esp:
        return "ERROR", "m=%d n=%d, el paper declara m=%d n=%d" % (inst.m, inst.n, m_esp, n_esp)
    if abs(inst.density - dens_esp) > 0.5:
        return "AVISO", "densidad %.2f%%, el paper declara %.2f%%" % (inst.density, dens_esp)
    if inst.deficit([1] * inst.n) != 0:
        return "ERROR", "hay filas que ninguna columna cubre"
    return "OK", "m=%d n=%d densidad=%.2f%%" % (inst.m, inst.n, inst.density)


def bloque(titulo, items):
    print("\n" + titulo)
    print("-" * 72)
    est = {"OK": 0, "AVISO": 0, "FALTA": 0, "ERROR": 0}
    for nombre, tag, m, n, d in items:
        s, msg = revisar(tag, m, n, d, nombre)
        est[s] += 1
        print("[%-5s] %-8s %s" % (s, nombre, msg))
    print("-" * 72)
    print("OK=%d  AVISO=%d  FALTA=%d  ERROR=%d" % (est["OK"], est["AVISO"], est["FALTA"], est["ERROR"]))
    return est


print("Carpeta de datos: %s" % os.path.abspath(cfg.DATA_DIR))

scp_items = [(t.upper(), t, pr.INSTANCE_INFO[t][0], pr.INSTANCE_INFO[t][1],
              pr.INSTANCE_INFO[t][2]) for t in pr.SCP_22]
e1 = bloque("SCP - 22 instancias (Tabla 6, p. 13)", scp_items)

uscp_items = [(k, v[0], v[1], v[2], v[3]) for k, v in pr.USCP_INFO.items()]
e2 = bloque("USCP - 17 instancias (Tabla 7, p. 13)", uscp_items)

print("\nRESUMEN")
print("-" * 72)
if e1["FALTA"] == 0 and e1["ERROR"] == 0:
    print("SCP:  listo para ejecutar las 22 instancias del paper.")
else:
    print("SCP:  faltan o fallan %d instancias." % (e1["FALTA"] + e1["ERROR"]))
if e2["FALTA"] == 0 and e2["ERROR"] == 0:
    print("USCP: listo para ejecutar las 17 instancias con UNICOST = True.")
else:
    print("USCP: faltan o fallan %d instancias. Ejecute: python descargar_instancias.py"
          % (e2["FALTA"] + e2["ERROR"]))