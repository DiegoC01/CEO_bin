import os
import sys
import time
import urllib.request

# --------------------------------------------------------------------------
# Descarga instancias SCP de OR-Library (J. E. Beasley, Brunel University).
# Pagina de referencia: people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html
# Los archivos cuelgan del directorio /~mastjjb/jeb/orlib/files/
#
# Uso:
#   python descargar_instancias.py            -> solo lo que falta para el paper
#   python descargar_instancias.py todas      -> las 22 SCP + las 6 CLR/CYC
# --------------------------------------------------------------------------

BASE = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"
DEST = "./instances"

# 22 instancias SCP del paper, Tabla 6 (p. 13)
SCP_22 = ["41", "42", "51", "52", "61", "62", "a1", "a2", "b1", "b2",
          "c1", "c2", "d1", "d2", "nre1", "nre2", "nrf1", "nrf2",
          "nrg1", "nrg2", "nrh1", "nrh2"]

# Instancias CLR y CYC del paper, Tabla 7 (p. 13). Son archivos propios,
# no se derivan de los anteriores.
CLR_CYC = ["clr10", "clr11", "clr12", "cyc06", "cyc07", "cyc08"]


def descargar(tag):
    nombre = "scp%s.txt" % tag
    destino = os.path.join(DEST, nombre)
    if os.path.exists(destino) and os.path.getsize(destino) > 0:
        print("  [ya existe] %s" % nombre)
        return True
    url = BASE + nombre
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as r:
            datos = r.read()
        if len(datos) < 100:
            print("  [SOSPECHOSO] %s solo %d bytes" % (nombre, len(datos)))
            return False
        with open(destino, "wb") as fh:
            fh.write(datos)
        print("  [ok] %-16s %8.1f KB" % (nombre, len(datos) / 1024.0))
        return True
    except Exception as e:
        print("  [ERROR] %s -> %s: %s" % (nombre, url, e))
        return False


def main():
    os.makedirs(DEST, exist_ok=True)
    todas = len(sys.argv) > 1 and sys.argv[1].lower().startswith("tod")
    tags = (SCP_22 + CLR_CYC) if todas else CLR_CYC
    print("Destino: %s" % os.path.abspath(DEST))
    print("Archivos a revisar: %d\n" % len(tags))
    ok = 0
    for t in tags:
        if descargar(t):
            ok += 1
        time.sleep(0.4)          # cortesia con el servidor
    print("\nCompletados: %d de %d" % (ok, len(tags)))
    if ok < len(tags):
        print("Si algo fallo, descargue a mano desde:")
        print("  https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/")


if __name__ == "__main__":
    main()