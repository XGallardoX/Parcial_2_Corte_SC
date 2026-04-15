from functools import reduce
from typing import Callable, List, Tuple

Estado = float
FuncionInteraccion = Callable[[Estado, Estado], Estado]

def interaccion_atraccion(emisor: Estado, receptor: Estado) -> Estado:
    """El receptor tiende hacia el estado del emisor."""
    return (emisor - receptor) * 0.1

def interaccion_repulsion(emisor: Estado, receptor: Estado) -> Estado:
    """El receptor se aleja del estado del emisor."""
    return -(emisor - receptor) * 0.05

def net_interaction(
    receptor: Estado,
    emisores: List[Estado],
    f_interaccion: FuncionInteraccion
) -> Estado:
    """
    Calcula la interacción neta sobre una entidad receptor.
    NI(e_i) = Σ_{j≠i} I(x_j, x_i)
    Implementado como fold (reduce) — paradigma funcional puro.
    """
    deltas = map(lambda emisor: f_interaccion(emisor, receptor), emisores)
    return reduce(lambda acc, delta: acc + delta, deltas, 0.0)

def paso_dinamica(
    estados: List[Estado],
    f_interaccion: FuncionInteraccion
) -> List[Estado]:
    """
    x_i(t+1) = x_i(t) + NI(e_i, t)
    Aplicado a todos los agentes con map — sin bucles mutables.
    """
    def actualizar(i_estado: Tuple[int, Estado]) -> Estado:
        i, estado = i_estado
        emisores = [e for j, e in enumerate(estados) if j != i]
        delta = net_interaction(estado, emisores, f_interaccion)
        return estado + delta

    return list(map(actualizar, enumerate(estados)))

if __name__ == "__main__":
    estados = [0.0, 2.0, 5.0, 8.0, 10.0]

    print("Simulación de Net Interactions — Atracción")
    print(f"{'Paso':<6} {'Estados'}")
    print("-" * 50)
    for paso in range(15):
        print(f"{paso:<6} {[round(e, 4) for e in estados]}")
        estados = paso_dinamica(estados, interaccion_atraccion)

    print(f"\nEstado final convergido: {round(estados[0], 4)}")
