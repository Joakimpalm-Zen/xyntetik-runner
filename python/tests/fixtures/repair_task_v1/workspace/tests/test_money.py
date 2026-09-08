from calc.money import parse_amount


def test_plain() -> None:
    assert parse_amount("12.34") == 1234


def test_thousands_separator() -> None:
    assert parse_amount("1,234.50") == 123450
