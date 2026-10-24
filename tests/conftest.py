"""
Shared fixtures: a small synthetic dataset in the IEEE-CIS format.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_transactions() -> pd.DataFrame:
    """
    1,200 transactions with the key IEEE-CIS columns and a learnable
    fraud pattern (large amounts on rare cards from one e-mail domain).
    """

    rng = np.random.default_rng(0)
    n = 1_200

    card1 = rng.choice([1000, 2000, 3000, 4000, 5000], size=n)
    domain = rng.choice(
        ["gmail.com", "yahoo.com", "anonymous.com", None], size=n
    )
    amount = rng.gamma(2.0, 50.0, size=n).round(2)

    fraud = (
        (domain == "anonymous.com") & (amount > 120)
        | (rng.random(n) < 0.02)
    ).astype(int)

    return pd.DataFrame(
        {
            "TransactionID": np.arange(3_000_000, 3_000_000 + n),
            "isFraud": fraud,
            "TransactionDT": np.sort(rng.integers(86_400, 86_400 * 60, n)),
            "TransactionAmt": amount,
            "ProductCD": rng.choice(["W", "C", "H"], size=n),
            "card1": card1,
            "card2": rng.choice([111.0, 222.0, np.nan], size=n),
            "card4": rng.choice(["visa", "mastercard"], size=n),
            "card6": rng.choice(["debit", "credit"], size=n),
            "addr1": rng.choice([100.0, 200.0, np.nan], size=n),
            "P_emaildomain": domain,
            "R_emaildomain": rng.choice(["gmail.com", None], size=n),
            "C1": rng.integers(0, 5, n).astype(float),
            "D1": rng.integers(0, 400, n).astype(float),
            "id_01": np.where(rng.random(n) < 0.3, -5.0, np.nan),
            "id_02": np.where(rng.random(n) < 0.3, 1000.0, np.nan),
            "DeviceType": rng.choice(["desktop", "mobile", None], size=n),
        }
    )
