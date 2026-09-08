"""Money parsing for the ledger importer."""


def parse_amount(text: str) -> int:
    """Return the amount in integer cents for a decimal money string."""
    return int(float(text) * 100)
