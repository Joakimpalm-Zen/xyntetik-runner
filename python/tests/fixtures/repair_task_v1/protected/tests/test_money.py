"""Protected tests: a superset of the visible ones. The attempt never sees
this file; the verifier writes it over the workspace copy before running."""
from calc.money import parse_amount


def test_plain() -> None:
    assert parse_amount("12.34") == 1234


def test_thousands_separator() -> None:
    assert parse_amount("1,234.50") == 123450


def test_float_trap() -> None:
    assert parse_amount("0.29") == 29


def test_negative() -> None:
    assert parse_amount("-12") == -1200


def test_currency_prefix() -> None:
    assert parse_amount("EUR 3.10") == 310
    assert parse_amount("€ 3.10") == 310


def test_whitespace() -> None:
    assert parse_amount(" 7.05 ") == 705
