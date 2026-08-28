import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data as data_mod  # noqa: E402


@pytest.fixture(scope="session")
def prices():
    return data_mod.synthetic_ohlcv(n=600, seed=42)


@pytest.fixture(scope="session")
def choppy():
    return data_mod.synthetic_ohlcv(n=600, seed=99, annual_drift=0.0, annual_vol=0.55)


@pytest.fixture(scope="session")
def benchmark(prices):
    return data_mod.synthetic_benchmark(prices.index, seed=5, regime_flip=380)
