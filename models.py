from dataclasses import dataclass


@dataclass
class Operation:
    '''
    Representation of an operation (position)
    '''
    time: str
    price: float
    stop_loss: float
    take_profit: float
    n_shares: float
    type: str


def get_portfolio_value(cash: float, long_ops: list[Operation],
                        short_ops: list[Operation], current_price: float,
                        COM: float) -> float:
    val = cash

    for position in long_ops:
        # Add long positions value
        val += current_price * position.n_shares

    for position in short_ops:
        # Add short positions value
        val += (position.price * position.n_shares) + (position.price * position.n_shares - current_price * position.n_shares) * (1 - COM)

    return val
