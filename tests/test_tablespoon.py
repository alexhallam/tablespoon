from tablespoon import __version__


def test_version():
    assert __version__ == '0.1.0'

# pytest -q tests/test_tablespoon.py 

import numpy as np
import pandas as pd
from tablespoon import Naive, Mean, Snaive
import cmdstanpy
import pytest


cmdstanpy.install_cmdstan()


# pull and clean data
df = (
    pd.read_csv("https://storage.googleapis.com/data_xvzf/m5_state_sales.csv")
    .query("state_id == 'CA'")
    .rename(columns={"date": "ds", "sales": "y"})
    .assign(y=lambda df: np.log(df.y))
)

# make lag of 1 default
m = Naive()
naive_forecast = m.predict(df, horizon=10)


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
mean_forecast = m.predict(df, horizon=10)
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
snaive_forecast = m.predict(df, horizon=10)
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
