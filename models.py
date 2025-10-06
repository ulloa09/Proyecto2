from dataclasses import dataclass

# --- Propósito del archivo ---
# Este archivo define la estructura de datos y funciones relacionadas con las operaciones financieras (posiciones)
# y el cálculo del valor total de un portafolio considerando posiciones largas y cortas.

@dataclass
class Operation:
    '''
    Representación de una operación (posición) financiera.
    Contiene información relevante para cada operación individual.
    '''
    time: str
    price: float
    stop_loss: float
    take_profit: float
    n_shares: float
    type: str

# --- Función: get_portfolio_value ---
# Calcula el valor total actual del portafolio, sumando el efectivo disponible y el valor de las posiciones abiertas.
#
# Parámetros:
# - cash (float): Cantidad de efectivo disponible en el portafolio.
# - long_ops (list[Operation]): Lista de posiciones largas abiertas.
# - short_ops (list[Operation]): Lista de posiciones cortas abiertas.
# - current_price (float): Precio actual del activo subyacente.
# - COM (float): Comisión o costo operacional aplicado a las posiciones cortas.
#
# Retorno:
# - float: Valor total actualizado del portafolio, incluyendo efectivo y valor de posiciones.
#
# Funcionamiento:
# - Se inicia con el valor del efectivo disponible.
# - Se suman los valores de las posiciones largas multiplicando el número de acciones por el precio actual.
# - Para las posiciones cortas, se calcula el valor considerando el precio de apertura, el precio actual,
#   y se ajusta por la comisión (COM) aplicada sobre la ganancia o pérdida.
# - Finalmente, se retorna el valor total calculado.

def get_portfolio_value(cash: float, long_ops: list[Operation],
                        short_ops: list[Operation], current_price: float,
                        COM: float) -> float:
    val = cash

    for position in long_ops:
        # Añadir el valor de las posiciones largas al total
        val += current_price * position.n_shares

    for position in short_ops:
        # Añadir el valor de las posiciones cortas al total,
        # considerando precio de apertura, precio actual y comisión
        val += (position.price * position.n_shares) + (position.price * position.n_shares - current_price * position.n_shares) * (1 - COM)

    return val
