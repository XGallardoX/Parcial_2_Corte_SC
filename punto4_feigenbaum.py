import numpy as np

DELTA = 4.669201609102990

def mapa_logistico(x: float, lam: float) -> float:
    return lam * x * (1 - x)

def bifurcaciones(lam: float, n_iter: int = 1000, n_last: int = 256) -> list:
    x = 0.5
    for _ in range(n_iter):
        x = mapa_logistico(x, lam)
    result = []
    for _ in range(n_last):
        x = mapa_logistico(x, lam)
        result.append(x)
    return result

def detectar_bifurcaciones(n_bifurcaciones: int = 5) -> list:
    puntos_bif = []
    lam_vals = np.linspace(2.9, 3.9, 200000)
    periodo_anterior = 1
    for lam in lam_vals:
        valores = set(round(v, 5) for v in bifurcaciones(lam, n_iter=2000))
        periodo_actual = len(valores)
        if periodo_actual == periodo_anterior * 2:
            puntos_bif.append(lam)
            periodo_anterior = periodo_actual
            if len(puntos_bif) == n_bifurcaciones:
                break
    return puntos_bif

def predecir_umbral_caos(puntos_bif: list) -> float:
    """mu_inf = mu_n + (mu_n - mu_{n-1}) / (delta - 1)"""
    mu_n = puntos_bif[-1]
    mu_n_prev = puntos_bif[-2]
    return mu_n + (mu_n - mu_n_prev) / (DELTA - 1)

def calcular_deltas_empiricos(puntos_bif: list) -> list:
    return [
        round((puntos_bif[i] - puntos_bif[i-1]) / (puntos_bif[i+1] - puntos_bif[i]), 6)
        for i in range(1, len(puntos_bif) - 1)
    ]

if __name__ == "__main__":
    print("Detectando puntos de bifurcación del sistema de control...")
    puntos = detectar_bifurcaciones(n_bifurcaciones=5)

    print(f"\nBifurcaciones encontradas (λ_n — ganancia crítica):")
    for i, p in enumerate(puntos, 1):
        print(f"  λ_{i} = {p:.8f}  (periodo {2**i})")

    deltas_emp = calcular_deltas_empiricos(puntos)
    print(f"\nδ empíricos:         {deltas_emp}")
    print(f"δ Feigenbaum teórico: {DELTA}")

    umbral = predecir_umbral_caos(puntos)
    print(f"\n→ Umbral de caos predicho: λ_∞ ≈ {umbral:.6f}")
    print(f"  Valor real conocido:      λ_∞ ≈ 3.569946")
    print(f"  El sistema de control es seguro para λ < {umbral:.4f}")
