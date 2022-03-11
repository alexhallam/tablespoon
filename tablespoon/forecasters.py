import logging
import shutil
import tempfile
import os
from pkg_resources import resource_filename

from cmdstanpy import CmdStanModel

import numpy as np
import pandas as pd

# cmdstanpy natrually generates a lot of logs this is to make those logs quiet unless there is an error
# https://discourse.mc-stan.org/t/cmdstanpy-not-compiling-with-uninformative-error/25576/7?u=mitzimorris
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def get_sorted_dates(df_historical):
    """
    get historical dates and sort them
    """
    return pd.to_datetime(pd.Series(df_historical["ds"].unique(), name="ds")).sort_values()


def check_historical_dates_exist(df_historical):
    """
    check that historical dates exist
    """
    if df_historical["ds"] is None:
        raise Exception("Please include historical dates and name the columns 'ds'.")


def send_helpful_frequency_error():
    """
    check that frequency argument exists else provide helpfull guidance
    """
    raise Exception(
        "Please specify frequence of data. \n`D`- daily\n`1H` - hourly\n For more frequency aliases see https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases"
    )


def check_historical_dates_are_contiguous(history_dates, min_date, last_date, frequency):
    """
    check that historical dates are contiguous
    """
    check_dates = pd.date_range(
        start=min_date,
        end=last_date,  # An extra in case we include start
        freq=frequency,
    )  # 'M','D', etc.
    expected_dates = set(check_dates.unique())
    given_dates = set(history_dates.unique())
    remaining_dates = expected_dates.difference(given_dates)
    # print(remaining_dates)
    if len(check_dates) != len(history_dates):
        raise Exception(
            "The series starts on "
            + str(min_date)
            + " and ends on "
            + str(last_date)
            + " the length of the series is expected to be "
            + str(len(check_dates))
            + ", but "
            + str(len(history_dates))
            + " was found "
        )


def fit_stan_model(model_name_string, y, lag: int, uncertainty_samples: int, horizon: int, chain_ids, verbose: bool):
    """
    Fit the stan model
    """
    with tempfile.TemporaryDirectory() as d:
        temp_dir_file = os.path.join(d, model_name_string + ".stan")
        stan_model_file = resource_filename("tablespoon", "stan/" + model_name_string + ".stan")
        shutil.copyfile(stan_model_file, temp_dir_file)
        out_dir = os.path.join(d, "stan", "out")
        if not os.path.exists(out_dir):
            out_dir = os.path.expanduser(out_dir)
            os.makedirs(out_dir, exist_ok=True)
        model_stan = CmdStanModel(stan_file=temp_dir_file)
        cmdstanpy_data = {"horizon": horizon, "T": len(y), "y": y, "lag": lag}
        fit = model_stan.sample(data=cmdstanpy_data, output_dir=out_dir, chains=1, seed=42, chain_ids=chain_ids, iter_sampling=uncertainty_samples)
        df_fit = fit.draws_pd()
        df_fit = df_fit.loc[:, df_fit.columns.str.startswith("forecast")]
        if verbose:
            print(fit.summary())
        return df_fit


class Naive(object):
    """Naive Forecaster

    Args:
        object (None): instantiates a Naive Forecast object
    """

    def __init__(
        self,
        history_dates=None,
        stan_backend=None,
        include_history=False,
        y=None,
    ):
        self.include_history = include_history

    def predict(
        self,
        df_historical,
        horizon=30,
        frequency=None,
        lag=1,
        uncertainty_samples=5000,
        include_history=False,
        chain_ids=None,
        verbose=False,
    ):
        """Predict - forecast method

        Args:
            df_historical (pd.DataFrame): A date sorted dataframe with the columns `ds` and `y`
            horizon (int, optional): Forecast horizon. Defaults to 30.
            frequency (int, optional): number of rows that make a seasonal period. Defaults to None.
            lag (int, optional): number of rows that make a seasonal period. Defaults to 1.
            uncertainty_samples (int, optional): number of uncertainty samples to draw. Defaults to 5000.
            include_history (bool, optional): include history. Defaults to False.
            chain_ids (str, optional): identifiers for chain ids. Defaults to None.
            verbose (bool, optional): verbose. Defaults to False.

        Returns:
            pd.DataFrame: A dataframe of predictions as `y_sim`
        """
        if frequency is None:
            send_helpful_frequency_error()
        self.y = df_historical["y"]
        self.history_dates = get_sorted_dates(df_historical)
        last_date = self.history_dates.max()
        min_date = self.history_dates.min()
        check_historical_dates_are_contiguous(self.history_dates, min_date, last_date, frequency)
        dates = pd.date_range(start=last_date, periods=horizon + 1, freq=frequency)  # An extra in case we include start  # 'M','D', etc.
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:horizon]  # Return correct number of periods
        if include_history:
            dates = np.concatenate((np.array(self.history_dates), dates))
        df_dates = pd.DataFrame({"ds": dates})
        df_samples = pd.DataFrame({"rep": np.arange(uncertainty_samples)})
        df_cross = df_dates.merge(df_samples, how="cross")
        df_fit = fit_stan_model("naive", self.y, lag, uncertainty_samples, horizon, chain_ids, verbose=verbose)
        np_predictions = df_fit.to_numpy().transpose().reshape(uncertainty_samples * horizon, 1)
        df_pred = pd.DataFrame(np_predictions, columns=["y_sim"])
        df_result = pd.concat([df_cross, df_pred], axis=1)
        return df_result


class Mean(object):
    """
    Mean Forecaster
    """

    def __init__(
        self,
        history_dates=None,
        stan_backend=None,
        include_history=False,
        y=None,
    ):
        self.include_history = include_history

    def predict(self, df_historical, horizon=30, frequency=None, lag=1, uncertainty_samples=5000, include_history=False, chain_ids=None, verbose=False):
        """Predict - forecast method

        Args:
            df_historical (pd.DataFrame): A date sorted dataframe with the columns `ds` and `y`
            horizon (int, optional): Forecast horizon. Defaults to 30.
            frequency (int, optional): number of rows that make a seasonal period. Defaults to None.
            lag (int, optional): number of rows that make a seasonal period. Defaults to 1.
            uncertainty_samples (int, optional): number of uncertainty samples to draw. Defaults to 5000.
            include_history (bool, optional): include history. Defaults to False.
            chain_ids (str, optional): identifiers for chain ids. Defaults to None.
            verbose (bool, optional): verbose. Defaults to False.

        Returns:
            pd.DataFrame: A dataframe of predictions as `y_sim`
        """
        self.y = df_historical["y"]
        self.history_dates = get_sorted_dates(df_historical)
        last_date = self.history_dates.max()
        min_date = self.history_dates.min()
        check_historical_dates_are_contiguous(self.history_dates, min_date, last_date, frequency)
        dates = pd.date_range(start=last_date, periods=horizon + 1, freq=frequency)  # An extra in case we include start  # 'M','D', etc.
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:horizon]  # Return correct number of periods
        if include_history:
            dates = np.concatenate((np.array(self.history_dates), dates))
        df_dates = pd.DataFrame({"ds": dates})
        df_samples = pd.DataFrame({"rep": np.arange(uncertainty_samples)})
        df_cross = df_dates.merge(df_samples, how="cross")
        df_fit = fit_stan_model("mean", self.y, lag, uncertainty_samples, horizon, chain_ids, verbose=verbose)
        np_predictions = df_fit.to_numpy().transpose().reshape(uncertainty_samples * horizon, 1)
        df_pred = pd.DataFrame(np_predictions, columns=["y_sim"])
        df_result = pd.concat([df_cross, df_pred], axis=1)
        return df_result


class Snaive(object):
    """
    Seasonal Naive Forecaster
    """

    def __init__(
        self,
        history_dates=None,
        stan_backend=None,
        include_history=False,
        y=None,
    ):
        self.include_history = include_history

    def predict(
        self,
        df_historical,
        horizon=30,
        frequency=None,
        lag=7,
        uncertainty_samples=5000,
        include_history=False,
        chain_ids=None,
        verbose=False,
    ):
        """Predict - forecast method

        Args:
            df_historical (pd.DataFrame): A date sorted dataframe with the columns `ds` and `y`
            horizon (int, optional): Forecast horizon. Defaults to 30.
            frequency (int, optional): number of rows that make a seasonal period. Defaults to None.
            lag (int, optional): number of rows that make a seasonal period. Defaults to 7 (7 days of a week).
            uncertainty_samples (int, optional): number of uncertainty samples to draw. Defaults to 5000.
            include_history (bool, optional): include history. Defaults to False.
            chain_ids (str, optional): identifiers for chain ids. Defaults to None.
            verbose (bool, optional): verbose. Defaults to False.

        Returns:
            pd.DataFrame: A dataframe of predictions as `y_sim`
        """
        self.y = df_historical["y"]
        self.history_dates = get_sorted_dates(df_historical)
        last_date = self.history_dates.max()
        min_date = self.history_dates.min()
        check_historical_dates_are_contiguous(self.history_dates, min_date, last_date, frequency)
        dates = pd.date_range(start=last_date, periods=horizon + 1, freq=frequency)  # An extra in case we include start  # 'M','D', etc.
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:horizon]  # Return correct number of periods
        if include_history:
            dates = np.concatenate((np.array(self.history_dates), dates))
        df_dates = pd.DataFrame({"ds": dates})
        df_samples = pd.DataFrame({"rep": np.arange(uncertainty_samples)})
        df_cross = df_dates.merge(df_samples, how="cross")
        df_fit = fit_stan_model("snaive", self.y, lag, uncertainty_samples, horizon, chain_ids, verbose=verbose)
        np_predictions = df_fit.to_numpy().transpose().reshape(uncertainty_samples * horizon, 1)
        df_pred = pd.DataFrame(np_predictions, columns=["y_sim"])
        df_result = pd.concat([df_cross, df_pred], axis=1)
        return df_result
