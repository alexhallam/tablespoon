from tablespoon import __version__

# For testing locally 
# poetry run python tests/test_tablespoon.py

# For dev run
# poetry run pytest

import numpy as np
import pandas as pd
from tablespoon import Naive, Mean, Snaive
from tablespoon.data import WALMART_TX, SEAS
import pytest


# small snaive test
df = SEAS
n = Snaive()
df_n = n.predict(
    df,
    horizon=7,
    frequency="D",
    lag=7,
    uncertainty_samples=1000,
).assign(model="snaive").loc[:,['ds', 'y_sim']].groupby('ds', as_index=False).mean()

# the following checks that means are withing 5 percent (0.05) of the last week
def test_snaive_1():
    value = df_n.y_sim[0]
    assert value == pytest.approx(150, 0.05)
def test_snaive_2():
    value = df_n.y_sim[1]
    assert value == pytest.approx(250, 0.05)
def test_snaive_3():
    value = df_n.y_sim[2]
    assert value == pytest.approx(350, 0.05)
def test_snaive_4():
    value = df_n.y_sim[3]
    assert value == pytest.approx(450, 0.05)
def test_snaive_5():
    value = df_n.y_sim[4]
    assert value == pytest.approx(550, 0.05)
def test_snaive_6():
    value = df_n.y_sim[5]
    assert value == pytest.approx(650, 0.01)
def test_snaive_7():
    value = df_n.y_sim[6]
    assert value == pytest.approx(750, 0.01)


# larger snaive test
df = WALMART_TX
n = Snaive()
df_tx = n.predict(
    df,
    horizon=14,
    frequency="D",
    lag=7,
    uncertainty_samples=1000,
).assign(model="snaive").loc[:,['ds', 'y_sim']].groupby('ds', as_index=False).mean()
print(df_tx)

def test_snaive_tx_1():
    value = df_tx.y_sim[0]
    assert value == pytest.approx(12228, 0.20)
def test_snaive_tx_2():
    value = df_tx.y_sim[1]
    assert value == pytest.approx(11370, 0.20)
def test_snaive_tx_3():
    value = df_tx.y_sim[2]
    assert value == pytest.approx(10375, 0.20)
def test_snaive_tx_4():
    value = df_tx.y_sim[3]
    assert value == pytest.approx(9162, 0.20)
def test_snaive_tx_5():
    value = df_tx.y_sim[4]
    assert value == pytest.approx(12303, 0.20)
def test_snaive_tx_6():
    value = df_tx.y_sim[5]
    assert value == pytest.approx(13681, 0.20)
def test_snaive_tx_7():
    value = df_tx.y_sim[6]
    assert value == pytest.approx(14815, 0.20)

# pull and clean data
df = (
    pd.read_csv("https://storage.googleapis.com/data_xvzf/m5_state_sales.csv")
    .query("state_id == 'CA'")
    .rename(columns={"date": "ds", "sales": "y"})
    .assign(y=lambda df: np.log(df.y))
)

# make lag of 1 default
m = Naive()
naive_forecast = m.predict(df, horizon=10, frequency="D")


def test_naive_2016_06_01_lower():
    value = round(
        naive_forecast.query("ds == '2016-06-01'").loc[:, "y_sim"].quantile(q=0.05), 2
    )
    assert value == pytest.approx(7.15, 0.3)


def test_naive_2016_06_01_upper():
    value = round(
        naive_forecast.query("ds == '2016-06-01'").loc[:, "y_sim"].quantile(q=0.95), 2
    )
    assert value == pytest.approx(13.12, 0.3)


m = Mean()
mean_forecast = m.predict(df, horizon=10, frequency="D")
round(mean_forecast.query("ds == '2016-06-01'").loc[:, "y_sim"].quantile(q=0.05), 2)
round(mean_forecast.query("ds == '2016-06-01'").loc[:, "y_sim"].quantile(q=0.95), 2)


def test_mean_2016_06_01_lower():
    value = round(
        mean_forecast.query("ds == '2016-06-01'").loc[:, "y_sim"].quantile(q=0.05), 2
    )
    assert value == pytest.approx(8.85, 0.3)


def test_mean_2016_06_01_upper():
    value = round(
        mean_forecast.query("ds == '2016-06-01'").loc[:, "y_sim"].quantile(q=0.95), 2
    )
    assert value == pytest.approx(10.32, 0.3)


m = Snaive()
snaive_forecast = m.predict(df, horizon=10, frequency="D")
round(snaive_forecast.query("ds == '2016-06-01'").loc[:, "y_sim"].quantile(q=0.05), 2)
round(snaive_forecast.query("ds == '2016-06-01'").loc[:, "y_sim"].quantile(q=0.95), 2)


def test_snaive_2016_06_01_lower():
    value = round(
        snaive_forecast.query("ds == '2016-06-01'").loc[:, "y_sim"].quantile(q=0.05), 2
    )
    assert value == pytest.approx(8.39, 0.3)


def test_snaive_2016_06_01_upper():
    value = round(
        snaive_forecast.query("ds == '2016-06-01'").loc[:, "y_sim"].quantile(q=0.95), 2
    )
    assert value == pytest.approx(10.96, 0.3)
