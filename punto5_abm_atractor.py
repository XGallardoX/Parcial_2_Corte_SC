import random
from functools import reduce

def crear_agente(id_: int, opinion: float, umbral: float, mu: float) -> dict:
    """Cada agente tiene su propio umbral de confianza y su propio mu."""
    return {"id": id_, "opinion": opinion, "umbral": umbral, "mu": mu}

def interactuar(agente_a: dict, agente_b: dict) -> tuple:
    """
    Deffuant heterogéneo: cada agente evalúa con su propio umbral y ajusta
    con su propio mu. La interacción ocurre sólo si ambos se 'aceptan'.
    """
    diff = abs(agente_a["opinion"] - agente_b["opinion"])
    if diff < agente_a["umbral"] and diff < agente_b["umbral"]:
        nueva_a = agente_a["opinion"] + agente_a["mu"] * (agente_b["opinion"] - agente_a["opinion"])
        nueva_b = agente_b["opinion"] + agente_b["mu"] * (agente_a["opinion"] - agente_b["opinion"])
        return (
            {**agente_a, "opinion": round(nueva_a, 6)},
            {**agente_b, "opinion": round(nueva_b, 6)}
        )
    return agente_a, agente_b

def paso_abm(agentes: list) -> list:
    agentes = list(agentes)
    if len(agentes) < 2:
        return agentes
    i, j = random.sample(range(len(agentes)), 2)
    agentes[i], agentes[j] = interactuar(agentes[i], agentes[j])
    return agentes

def varianza_opiniones(agentes: list) -> float:
    opiniones = [a["opinion"] for a in agentes]
    media = reduce(lambda acc, x: acc + x, opiniones, 0.0) / len(opiniones)
    return reduce(lambda acc, x: acc + (x - media) ** 2, opiniones, 0.0) / len(opiniones)

def n_clusters(agentes: list, tolerancia: float = 0.05) -> int:
    return len(set(round(a["opinion"] / tolerancia) * tolerancia for a in agentes))

def inicializar_poblacion(n: int, rng: random.Random) -> list:
    """Cada agente: opinión, umbral y mu aleatorios e independientes."""
    return [
        crear_agente(
            id_=i,
            opinion=rng.uniform(0, 1),
            umbral=rng.uniform(0.1, 0.5),
            mu=rng.uniform(0.1, 0.5),
        )
        for i in range(n)
    ]

def ejecutar_experimento(n_agentes: int, n_pasos: int, semilla: int) -> dict:
    rng = random.Random(semilla)
    random.seed(semilla)
    agentes = inicializar_poblacion(n_agentes, rng)

    estado_inicial = [(a["opinion"], a["umbral"], a["mu"]) for a in agentes]

    for _ in range(n_pasos):
        agentes = paso_abm(agentes)

    return {
        "n_agentes": n_agentes,
        "estado_inicial": estado_inicial,
        "opiniones_finales": [a["opinion"] for a in agentes],
        "varianza": varianza_opiniones(agentes),
        "clusters": n_clusters(agentes),
    }

def describir_atractor(varianza: float, clusters: int) -> str:
    if varianza < 1e-3:
        return "PUNTO FIJO (consenso global)"
    if clusters == 1:
        return "PUNTO FIJO aproximado (una sola región)"
    return f"AGRUPAMIENTO en {clusters} clusters estables"

if __name__ == "__main__":
    CONFIGURACIONES = [
        {"n_agentes":  2, "n_pasos":  2000, "semilla": 42},
        {"n_agentes":  3, "n_pasos":  3000, "semilla": 42},
        {"n_agentes": 20, "n_pasos":  5000, "semilla": 42},
        {"n_agentes": 50, "n_pasos": 10000, "semilla": 42},
    ]

    print("=" * 72)
    print("ABM — Atractores con umbral y mu HETEROGÉNEOS por agente")
    print("=" * 72)

    for cfg in CONFIGURACIONES:
        res = ejecutar_experimento(**cfg)
        print(f"\n--- N = {res['n_agentes']} agentes  (pasos = {cfg['n_pasos']}) ---")
        print("Parámetros iniciales (opinion, umbral, mu):")
        for i, (op, u, m) in enumerate(res["estado_inicial"]):
            print(f"  agente {i:>2}: opinion={op:.4f}  umbral={u:.4f}  mu={m:.4f}")
        print(f"Opiniones finales: {[round(o, 4) for o in res['opiniones_finales']]}")
        print(f"Varianza final:    {res['varianza']:.6f}")
        print(f"Clusters:          {res['clusters']}")
        print(f"→ Atractor: {describir_atractor(res['varianza'], res['clusters'])}")

    print("\n" + "=" * 72)
    print("Interpretación: al dejar que cada agente tenga su propio umbral y mu,")
    print("el sistema revela distintos atractores según N y la heterogeneidad")
    print("inicial. Con pocos agentes tiende al consenso; al crecer N emergen")
    print("clusters estables — el atractor deja de ser un punto fijo y pasa a")
    print("ser un conjunto de opiniones co-existentes.")
    print("=" * 72)
