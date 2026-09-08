# Repair task v1 (frozen)

`calc.money.parse_amount` must return the amount in integer cents for money
strings a ledger export contains: plain decimals ("12.34"), thousands
separators ("1,234.50"), negatives ("-12"), a currency prefix ("EUR 3.10",
"€ 3.10") and surrounding whitespace. Today it only handles plain decimals,
and it rounds through binary floating point. Make `tests/test_money.py`
pass without changing the tests.
