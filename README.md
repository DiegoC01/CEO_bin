# CEOBIN: CEO binarizado y probado con el Set Covering Problem

Binarización del algoritmo Chaotic Evolution Optimization (CEO) [1] y evaluación
sobre instancias SCP y USCP de OR-Library, comparada contra los resultados
publicados de SBOA, GWO y PSO [2] (Tablas 8 y 9 para SCP, Tabla 12 para USCP).

`CEO.py` es la implementación base y no se modifica. Las extensiones se hacen
por herencia en `ceo_scp.py`.

## Uso

Requiere Python 3.8 o superior, NumPy y SciPy (este último opcional, pero sin él
las instancias grandes van mucho más lentas).

```
pip install numpy scipy
python descargar_instancias.py todas    # instancias desde OR-Library
python verificar_instancias.py          # comprueba dimensiones y densidad
python run.py
```

Todo se configura en `config.py`; es el único archivo que se edita. Los
parámetros están agrupados por origen: los que fija el artículo de referencia
(población 10, 600 iteraciones, 31 corridas, transferencia V3, regla Elitist),
los propios de CEO (dominio continuo, muestras caóticas, criterio de término) y
los que ninguna fuente especifica.

## Archivos

| Archivo | Contenido |
|---|---|
| `config.py` | Parámetros. El único que se edita. |
| `run.py` | Ejecuta los experimentos y genera las tablas. |
| `ceo_scp.py` | Subclase de CEO: binarización, evaluación y criterio de término. |
| `scp.py` | Instancias, evaluación, reparación y baselines de control. |
| `binarization.py` | 8 funciones de transferencia y 5 reglas de binarización. |
| `paper_results.py` | Resultados publicados, transcritos de [2]. |
| `descargar_instancias.py`, `verificar_instancias.py` | Obtención y validación de instancias. |

## Salidas

`<prefijo>_tabla.txt` es la tabla comparativa legible y se reescribe tras cada
corrida. También se generan `.csv`, `.md` y `.tex` (estas requieren
`booktabs`), la bitácora para reanudar y las curvas de convergencia.

Con `RESUME = True` una corrida se reutiliza solo si coincide toda su
configuración. Usa un `OUT_PREFIX` distinto por experimento.


## Referencias

```
[1] Y. Dong, S. Zhang, H. Zhang, X. Zhou, and J. Jiang, "Chaotic evolution
    optimization: A novel metaheuristic algorithm inspired by chaotic
    dynamics," Chaos, Solitons and Fractals, vol. 192, art. 116049, 2025.
[2] B. Crawford et al., "Binary Secretary Bird Optimization Algorithm for the
    Set Covering Problem," Mathematics, vol. 13, art. 2482, 2025.
[3] V. Chvátal, "A greedy heuristic for the set-covering problem," Mathematics
    of Operations Research, vol. 4, no. 3, pp. 233-235, Aug. 1979,
    doi: 10.1287/moor.4.3.233.
[4] J. E. Beasley and P. C. Chu, "A genetic algorithm for the set covering
    problem," European Journal of Operational Research, vol. 94, no. 2,
    pp. 392-404, Oct. 1996, doi: 10.1016/0377-2217(95)00159-X.
[5] T. Grossman and A. Wool, "Computational experience with approximation
    algorithms for the set covering problem," European Journal of Operational
    Research, vol. 101, no. 1, pp. 81-92, 1997.
[6] G. Lan, G. W. DePuy, and G. E. Whitehouse, "An effective and simple
    heuristic for the set covering problem," European Journal of Operational
    Research, vol. 176, no. 3, pp. 1387-1403, 2007,
    doi: 10.1016/j.ejor.2005.09.028.
```

Instancias de OR-Library (J. E. Beasley),
`people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html`. Los archivos unicost CLR
y CYC corresponden a [5].
