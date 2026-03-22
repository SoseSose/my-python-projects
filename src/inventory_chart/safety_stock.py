import math


def get_safety_stock(data, safety_factor: float) -> int:
    values = [int(x) for x in data]
    if not values:
        raise ValueError("data is empty")

    xs = sorted(values)
    n = len(xs)

    index = math.ceil(safety_factor * n) - 1
    return xs[index]
