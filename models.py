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
                        n_shares: int, COM: float) -> float:
    val = cash

    # Add long positions value
    val += len(long_ops) * current_price * n_shares

    #Add short positions value
    for position in short_ops:
    val += ((position.price * position.n_shares)+ (position.price * n_shares - position.close * n_shares)) * (1 - COM)

    return val