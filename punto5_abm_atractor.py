import random
from functools import reduce

UMBRAL_CONFIANZA = 0.5
MU = 0.3

def crear_agente(id_: int, opinion: float) -> dict:
    return {"id": id_, "opinion": opinion}

def interactuar(agente_a: dict, agente_b: dict) -> tuple:
    """Modelo de Deffuant: convergencia si |opinion_a - opinion_b| < umbral."""
    diff = abs(agente_a["opinion"] - agente_b["opinion"])
    if diff < UMBRAL_CONFIANZA:
        nueva_a = agente_a["opinion"] + MU * (agente_b["opinion"] - agente_a["opinion"])
        nueva_b = agente_b["opinion"] + MU * (agente_a["opinion"] - agente_b["opinion"])
        return (
            {**agente_a, "opinion": round(nueva_a, 6)},
            {**agente_b, "opinion": round(nueva_b, 6)}
        )
    return agente_a, agente_b

def paso_abm(agentes: list) -> list:
    agentes = list(agentes)
    i, j = random.sample(range(len(agentes)), 2)
    agentes[i], agentes[j] = interactuar(agentes[i], agentes[j])
    return agentes

def varianza_opiniones(agentes: list) -> float:
    opiniones = [a["opinion"] for a in agentes]
    media = reduce(lambda acc, x: acc + x, opiniones, 0.0) / len(opiniones)
    return reduce(lambda acc, x: acc + (x - media) ** 2, opiniones, 0.0) / len(opiniones)

def n_clusters(agentes: list, tolerancia: float = 0.05) -> int:
    return len(set(round(a["opinion"] / tolerancia) * tolerancia for a in agentes))

if __name__ == "__main__":
    random.seed(42)
    N_AGENTES = 20
    N_PASOS   = 5000

    agentes = [crear_agente(i, random.uniform(0, 1)) for i in range(N_AGENTES)]

    print(f"Parámetros: UMBRAL_CONFIANZA={UMBRAL_CONFIANZA}, MU={MU}, N={N_AGENTES}")
    print(f"\n{'Paso':<8} {'Varianza':<14} {'Clusters'}")
    print("-" * 35)

    for paso in range(N_PASOS):
        agentes = paso_abm(agentes)
        if paso % 1000 == 0:
            print(f"{paso:<8} {varianza_opiniones(agentes):<14.6f} {n_clusters(agentes)}")

    var_final = varianza_opiniones(agentes)
    print(f"\nVarianza final: {var_final:.6f}")
    if var_final < 0.001:
        print("→ Atractor de PUNTO FIJO: consenso global alcanzado.")
    else:
        print(f"→ Atractor de AGRUPAMIENTO: {n_clusters(agentes)} clusters estables de opinión.")
