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
    n_shares: int
    type: str


def get_portfolio_value(cash: float, long_ops: list[Operation],
                        short_ops: list[Operation], current_price: float,
                        n_shares: int) -> float:
    val = cash

    # Add long positions value
    val += len(long_ops) * current_price * n_shares

    #Add short positions value

    return val