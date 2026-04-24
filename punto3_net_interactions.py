"""
Net Interactions aplicado a una app CLIENTE-SERVIDOR.

Modelamos una aplicación distribuida con N clientes y M servidores.
- Estado de un cliente:  al servidor al que apunta actualmente (id).
- Estado de un servidor: su carga = # de clientes que lo apuntan.

La INTERACCIÓN está definida en sentido matemático (net interactions):
  - Cada servidor emite una 'señal de latencia' proporcional a su carga.
  - Cada cliente es receptor: sobre él actúa la interacción neta
        NI(cliente_i) = Σ_{s ∈ servidores} I(servidor_s, cliente_i)
    que, aquí, se resuelve seleccionando el servidor con señal de menor
    latencia — es el balanceo emergente de una app real.

Todo el núcleo usa programación funcional (map, reduce, sin mutación).
"""
from functools import reduce
from typing import Callable, List, NamedTuple, Tuple

class Servidor(NamedTuple):
    id: int
    capacidad: int  # carga "ideal" — define cuán rápido se satura

class Cliente(NamedTuple):
    id: int
    servidor_id: int  # a qué servidor apunta actualmente

Interaccion = Callable[[Servidor, int, Cliente], float]

def latencia(servidor: Servidor, carga: int, _cliente: Cliente) -> float:
    """
    I(servidor, cliente): 'empuje' que el servidor ejerce sobre el cliente.
    Modelo M/M/1-like: la latencia crece cuadráticamente con la saturación.
    """
    saturacion = carga / servidor.capacidad
    return saturacion ** 2

def carga_por_servidor(clientes: List[Cliente], servidores: List[Servidor]) -> dict:
    """Reduce funcional: cuenta cuántos clientes apuntan a cada servidor."""
    inicial = {s.id: 0 for s in servidores}
    return reduce(
        lambda acc, c: {**acc, c.servidor_id: acc[c.servidor_id] + 1},
        clientes,
        inicial,
    )

def net_interaction(
    cliente: Cliente,
    servidores: List[Servidor],
    cargas: dict,
    f_interaccion: Interaccion,
) -> List[Tuple[int, float]]:
    """
    NI(cliente) = vector de señales de latencia percibidas
    desde TODOS los servidores.  (map funcional, sin for.)
    """
    return list(map(
        lambda s: (s.id, f_interaccion(s, cargas[s.id], cliente)),
        servidores,
    ))

def decidir_destino(cliente: Cliente, senales: List[Tuple[int, float]]) -> Cliente:
    """El cliente elige el servidor de menor latencia percibida."""
    mejor_id, _ = reduce(
        lambda mejor, actual: actual if actual[1] < mejor[1] else mejor,
        senales,
    )
    return cliente._replace(servidor_id=mejor_id)

def paso_dinamica(
    clientes: List[Cliente],
    servidores: List[Servidor],
    f_interaccion: Interaccion,
) -> List[Cliente]:
    """
    Barrido asíncrono (Gauss-Seidel): cada cliente, en orden, recalcula
    su interacción neta contra los servidores usando la carga VIGENTE
    en ese momento (los ya migrados + los aún pendientes, descontándose
    a sí mismo) y migra al servidor de menor latencia.

    Implementado con reduce sobre enumerate(clientes) — sin bucles
    mutables, manteniendo el paradigma funcional.
    """
    def reubicar(acc: List[Cliente], idx_c: Tuple[int, Cliente]) -> List[Cliente]:
        idx, cliente = idx_c
        otros = acc + list(clientes[idx + 1:])  # resto del sistema
        cargas = carga_por_servidor(otros, servidores)
        # aseguramos que todos los servidores estén en el dict (otros puede
        # no tocar alguno); net_interaction los recorre a todos.
        cargas = {s.id: cargas.get(s.id, 0) for s in servidores}
        senales = net_interaction(cliente, servidores, cargas, f_interaccion)
        return acc + [decidir_destino(cliente, senales)]

    return reduce(reubicar, enumerate(clientes), [])

def simular(
    clientes: List[Cliente],
    servidores: List[Servidor],
    n_pasos: int,
    f_interaccion: Interaccion = latencia,
) -> List[List[Cliente]]:
    """
    Itera la dinámica n_pasos veces y devuelve la traza completa.
    Implementado con reduce: estado(t+1) = paso_dinamica(estado(t)).
    """
    return reduce(
        lambda traza, _: traza + [paso_dinamica(traza[-1], servidores, f_interaccion)],
        range(n_pasos),
        [clientes],
    )

def imprimir_estado(paso: int, clientes: List[Cliente], servidores: List[Servidor]) -> None:
    cargas = carga_por_servidor(clientes, servidores)
    linea = "  ".join(f"srv{s.id}(cap={s.capacidad}):{cargas[s.id]:>2}" for s in servidores)
    print(f"Paso {paso:>2} | {linea}")

if __name__ == "__main__":
    servidores = [
        Servidor(id=0, capacidad=5),
        Servidor(id=1, capacidad=10),
        Servidor(id=2, capacidad=3),
    ]
    # Escenario inicial pésimo: TODOS los clientes colgados del servidor 0.
    clientes = [Cliente(id=i, servidor_id=0) for i in range(12)]

    print("App cliente-servidor — balanceo emergente vía net interactions")
    print("=" * 60)
    print("Estado inicial (todos en srv0):")
    imprimir_estado(0, clientes, servidores)

    traza = simular(clientes, servidores, n_pasos=8)

    print("\nEvolución de la carga por servidor:")
    for t, estado in enumerate(traza[1:], start=1):
        imprimir_estado(t, estado, servidores)

    final = traza[-1]
    cargas_finales = carga_por_servidor(final, servidores)
    total = sum(cargas_finales.values())
    print("\nDistribución final de clientes:")
    for s in servidores:
        pct = 100 * cargas_finales[s.id] / total
        print(f"  srv{s.id}: {cargas_finales[s.id]}/{total} clientes  ({pct:.1f}%)  "
              f"latencia ≈ {latencia(s, cargas_finales[s.id], final[0]):.3f}")

    print("\nEl equilibrio emerge de interacciones locales (cada cliente")
    print("sólo mira la señal de latencia de cada servidor). No hay un")
    print("planificador central: es exactamente el patrón net interactions.")
