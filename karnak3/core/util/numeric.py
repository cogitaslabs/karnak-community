from decimal import Decimal


def precise_float_to_decimal(f: float, mantissa_digits: int = 6) -> Decimal:
    d = Decimal(f)
    new_precision = mantissa_digits - d.adjusted()
    if new_precision > 0:
        smartd = round(d, new_precision)
    else:
        smartd = d
    return smartd


def precise_double_to_decimal(f: float, mantissa_digits: int = 12) -> Decimal:
    return precise_float_to_decimal(f, mantissa_digits=mantissa_digits)
