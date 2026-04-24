"""
Universalidad δ de Feigenbaum aplicada a INGENIERÍA DE SOFTWARE.

Problema de la vida real:
------------------------
Tenemos un algoritmo iterativo de optimización (p.ej. tuning de
un hiperparámetro λ en un sistema de recomendación). Entre dos
iteraciones consecutivas medimos la razón de mejora:

        x_{n+1} = f(x_n, λ)  con  f(x, λ) = λ · x · (1 - x)

Aquí x_n ∈ (0, 1) representa el *ratio de mejora de la complejidad
computacional* del paso n (cuánto mejora el costo al aplicar el
algoritmo una vez más). λ es la ganancia del ajuste.

Criterio de parada (ingeniería):
    si |x_{n+1} - x_n| < Δ   → el algoritmo CONVERGIÓ.
    si oscila entre 2^k valores → el algoritmo está en régimen
                                   periódico (aceptable si k pequeño).
    si entra en caos → el algoritmo es inútil en producción.

La constante de Feigenbaum  δ = 4.66920160910299...  nos permite
PREDECIR, a partir de dos bifurcaciones observadas, a qué valor
de λ el algoritmo colapsará al caos. Así el ingeniero fija
    λ_max = λ_∞ − margen_seguridad
sin tener que explorar todo el espectro de λ.
"""
import numpy as np

DELTA = 4.669201609102990  # constante universal de Feigenbaum
TOLERANCIA = 1e-6          # Δ constante exigido a cada paso del algoritmo

def paso_algoritmo(x: float, lam: float) -> float:
    """Un paso del algoritmo iterativo (mapa logístico)."""
    return lam * x * (1 - x)

def ejecutar_hasta_tolerancia(
    lam: float, x0: float = 0.5,
    tol: float = TOLERANCIA, max_iter: int = 20000,
) -> dict:
    """
    Ejecuta el algoritmo hasta que |x_{n+1} - x_n| < tol durante
    'warmup' pasos consecutivos, o hasta max_iter.
    Devuelve el diagnóstico: convergió / periódico / caótico.
    """
    x = x0
    historial = [x]
    for n in range(max_iter):
        x_next = paso_algoritmo(x, lam)
        historial.append(x_next)
        if abs(x_next - x) < tol:
            return {"lam": lam, "estado": "CONVERGIDO",
                    "iteraciones": n + 1, "x_final": x_next,
                    "historial_cola": historial[-5:]}
        x = x_next
    # No convergió — clasificamos por el comportamiento de la cola
    cola = historial[-512:]
    periodos_detectados = len({round(v, 5) for v in cola})
    if periodos_detectados <= 16:
        return {"lam": lam, "estado": f"PERIÓDICO (período ~{periodos_detectados})",
                "iteraciones": max_iter, "x_final": None,
                "historial_cola": historial[-5:]}
    return {"lam": lam, "estado": "CAÓTICO (inviable en producción)",
            "iteraciones": max_iter, "x_final": None,
            "historial_cola": historial[-5:]}

def orbita_estacionaria(lam: float, n_iter: int = 20000, n_last: int = 256) -> list:
    """Descarta un transiente largo para evitar falsos positivos cerca
    de cada bifurcación (critical slowing down)."""
    x = 0.5
    for _ in range(n_iter):
        x = paso_algoritmo(x, lam)
    cola = []
    for _ in range(n_last):
        x = paso_algoritmo(x, lam)
        cola.append(x)
    return cola

def _periodo(cola: list, decimales: int = 4) -> int:
    return len({round(v, decimales) for v in cola})

def detectar_bifurcaciones(n_bifurcaciones: int = 4) -> list:
    """
    Barre λ y detecta las duplicaciones de período (1→2→4→8→...).
    Usa un barrido fino y exige que el nuevo período se sostenga
    durante varias muestras consecutivas — filtro anti-ruido.
    """
    puntos_bif = []
    lam_vals = np.linspace(2.95, 3.58, 4000)
    periodo_anterior = 1
    confirmaciones_necesarias = 3
    racha = 0
    lam_candidato = None
    for lam in lam_vals:
        p = _periodo(orbita_estacionaria(lam))
        if p == periodo_anterior * 2 and racha == 0:
            racha = 1
            lam_candidato = float(lam)
        elif p == periodo_anterior * 2 and racha > 0:
            racha += 1
            if racha >= confirmaciones_necesarias:
                puntos_bif.append(lam_candidato)
                periodo_anterior *= 2
                racha = 0
                lam_candidato = None
                if len(puntos_bif) == n_bifurcaciones:
                    break
        else:
            racha = 0
            lam_candidato = None
    return puntos_bif

def calcular_deltas_empiricos(puntos_bif: list) -> list:
    return [
        (puntos_bif[i] - puntos_bif[i-1]) / (puntos_bif[i+1] - puntos_bif[i])
        for i in range(1, len(puntos_bif) - 1)
    ]

def predecir_umbral_caos(puntos_bif: list) -> float:
    """
    λ_∞ ≈ λ_n + (λ_n - λ_{n-1}) / (δ - 1)
    Esta fórmula es la utilidad de ingeniería: con dos bifurcaciones
    consecutivas medidas, Feigenbaum nos regala el umbral de caos.
    """
    lam_n, lam_n_prev = puntos_bif[-1], puntos_bif[-2]
    return lam_n + (lam_n - lam_n_prev) / (DELTA - 1)

if __name__ == "__main__":
    print("=" * 72)
    print("Feigenbaum δ aplicado a un algoritmo iterativo real")
    print(f"Tolerancia exigida por paso:  Δ = {TOLERANCIA}")
    print("=" * 72)

    print("\n[1] Detectando bifurcaciones (ganancias críticas del algoritmo)...")
    puntos = detectar_bifurcaciones(n_bifurcaciones=4)
    for i, p in enumerate(puntos, 1):
        print(f"    λ_{i} = {p:.8f}   (período {2**i})")

    deltas = calcular_deltas_empiricos(puntos)
    print(f"\n[2] δ empíricos (cociente de longitudes entre bifurcaciones):")
    for i, d in enumerate(deltas, 1):
        print(f"    δ_{i} = {d:.6f}")
    print(f"    δ teórico de Feigenbaum = {DELTA}")

    umbral = predecir_umbral_caos(puntos)
    print(f"\n[3] Predicción del umbral de caos vía δ:")
    print(f"    λ_∞ ≈ {umbral:.6f}   (valor aceptado: 3.569946)")

    print("\n[4] Consecuencia de ingeniería — ¿qué pasa al correr el algoritmo")
    print("    con distintas ganancias y la tolerancia Δ fija?")
    print(f"    {'λ':>7} | estado                                 | iter | x_final")
    print("    " + "-" * 68)
    for lam in [2.8, 3.2, 3.45, 3.54, 3.569, 3.7, 3.9]:
        r = ejecutar_hasta_tolerancia(lam)
        x_str = f"{r['x_final']:.6f}" if r["x_final"] is not None else "   -   "
        print(f"    {lam:>7.3f} | {r['estado']:<38} | {r['iteraciones']:>4} | {x_str}")

    margen = 0.02
    lam_max_seguro = umbral - margen
    print(f"\n[5] Recomendación: operar el algoritmo con λ ≤ {lam_max_seguro:.4f}")
    print(f"    (umbral predicho {umbral:.4f} menos margen {margen}). Por encima")
    print(f"    de λ_∞ el algoritmo NO converge a tolerancia Δ — entra en caos.")
    print("=" * 72)
