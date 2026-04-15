def S(n):
    return n + 1

def suma(a, b):
    """Suma dos números naturales usando exclusivamente la función sucesor S."""
    if b == 0:
        return a
    return suma(S(a), b - 1)

def multiplicacion(a, b):
    """Multiplica dos números naturales usando exclusivamente la función suma."""
    if b == 0:
        return 0
    return suma(a, multiplicacion(a, b - 1))

if __name__ == "__main__":
    print("=== Función Sucesor ===")
    print(f"S(0) = {S(0)}")
    print(f"S(4) = {S(4)}")

    print("\n=== Suma con Sucesor ===")
    print(f"suma(3, 4)  = {suma(3, 4)}")
    print(f"suma(0, 7)  = {suma(0, 7)}")
    print(f"suma(10, 5) = {suma(10, 5)}")

    print("\n=== Multiplicación con Suma ===")
    print(f"multiplicacion(3, 4) = {multiplicacion(3, 4)}")
    print(f"multiplicacion(5, 6) = {multiplicacion(5, 6)}")
    print(f"multiplicacion(7, 0) = {multiplicacion(7, 0)}")
