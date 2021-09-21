from pkg_resources import resource_filename

from cmdstanpy import CmdStanModel

import numpy as np
import pandas as pd


class Naive(object):
    """
    Naive Forecaster
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
        frequency="D",
        lag=1,
        uncertainty_samples=5000,
        include_history=False,
    ):
        self.history_dates = pd.to_datetime(
            pd.Series(df_historical["ds"].unique(), name="ds")
        ).sort_values()
        self.y = df_historical["y"]
        if self.history_dates is None:
            raise Exception("Please include historical observations.")
        last_date = self.history_dates.max()
        min_date = self.history_dates.min()
        check_dates = pd.date_range(
            start=min_date,
            end=last_date,  # An extra in case we include start
            freq=frequency,
        )  # 'M','D', etc.
        expected_dates = set(check_dates.unique())
        given_dates = set(df_historical["ds"].unique())
        remaining_dates = expected_dates.difference(given_dates)
        #print(remaining_dates)
        if len(check_dates) != len(df_historical["ds"]):
            raise Exception("The series starts on " + str(min_date) + " and ends on " + str(last_date) +
                    " the length of the series is expected to be " + str(len(check_dates)) + ", but " + str(len(df_historical["ds"])) + " was found ")
        dates = pd.date_range(
            start=last_date,
            periods=horizon + 1,  # An extra in case we include start
            freq=frequency,
        )  # 'M','D', etc.
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:horizon]  # Return correct number of periods
        if include_history:
            dates = np.concatenate((np.array(self.history_dates), dates))
        df_dates = pd.DataFrame({"ds": dates})
        df_samples = pd.DataFrame({"rep": np.arange(uncertainty_samples)})
        df_cross = df_dates.merge(df_samples, how="cross")
        # naive model
        stan_model_file = resource_filename("tablespoon", "stan/naive.stan")
        out_dir = resource_filename("tablespoon", "stan/out")
        model_stan = CmdStanModel(stan_file=stan_model_file)
        cmdstanpy_data = {"horizon": horizon, "T": len(self.y), "y": self.y, "lag": lag}
        fit = model_stan.sample(
            data=cmdstanpy_data,
            output_dir=out_dir,
            chains=1,
            seed=42,
            iter_sampling=uncertainty_samples,
        )
        df_fit = fit.draws_pd()
        df_fit = df_fit.loc[:, df_fit.columns.str.startswith("forecast")]
        np_predictions = (
            df_fit.to_numpy().transpose().reshape(uncertainty_samples * horizon, 1)
        )
        print(fit.summary())
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

    def predict(
        self,
        df_historical,
        horizon=30,
        frequency="D",
        uncertainty_samples=5000,
        include_history=False,
    ):
        self.history_dates = pd.to_datetime(
            pd.Series(df_historical["ds"].unique(), name="ds")
        ).sort_values()
        self.y = df_historical["y"]
        if self.history_dates is None:
            raise Exception("Please include historical observations.")
        last_date = self.history_dates.max()
        min_date = self.history_dates.min()
        check_dates = pd.date_range(
            start=min_date,
            end=last_date,  # An extra in case we include start
            freq=frequency,
        )  # 'M','D', etc.
        expected_dates = set(check_dates.unique())
        given_dates = set(df_historical["ds"].unique())
        remaining_dates = expected_dates.difference(given_dates)
        #print(remaining_dates)
        if len(check_dates) != len(df_historical["ds"]):
            raise Exception("The series starts on " + str(min_date) + " and ends on " + str(last_date) +
                    " the length of the series is expected to be " + str(len(check_dates)) + ", but " + str(len(df_historical["ds"])) + " was found ")
        dates = pd.date_range(
            start=last_date,
            periods=horizon + 1,  # An extra in case we include start
            freq=frequency,
        )  # 'M','D', etc.
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:horizon]  # Return correct number of periods
        if include_history:
            dates = np.concatenate((np.array(self.history_dates), dates))
        df_dates = pd.DataFrame({"ds": dates})
        df_samples = pd.DataFrame({"rep": np.arange(uncertainty_samples)})
        df_cross = df_dates.merge(df_samples, how="cross")
        # naive model
        stan_model_file = resource_filename("tablespoon", "stan/mean.stan")
        out_dir = resource_filename("tablespoon", "stan/out")
        model_stan = CmdStanModel(stan_file=stan_model_file)
        cmdstanpy_data = {"horizon": horizon, "T": len(self.y), "y": self.y}
        fit = model_stan.sample(
            data=cmdstanpy_data,
            output_dir=out_dir,
            chains=1,
            seed=42,
            iter_sampling=uncertainty_samples,
        )
        df_fit = fit.draws_pd()
        df_fit = df_fit.loc[:, df_fit.columns.str.startswith("forecast")]
        np_predictions = (
            df_fit.to_numpy().transpose().reshape(uncertainty_samples * horizon, 1)
        )
        print(fit.summary())
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
        frequency="D",
        lag=7,
        uncertainty_samples=5000,
        include_history=False,
    ):
        self.history_dates = pd.to_datetime(
            pd.Series(df_historical["ds"].unique(), name="ds")
        ).sort_values()
        self.y = df_historical["y"]
        if self.history_dates is None:
            raise Exception("Please include historical observations.")
        last_date = self.history_dates.max()
        min_date = self.history_dates.min()
        check_dates = pd.date_range(
            start=min_date,
            end=last_date,  # An extra in case we include start
            freq=frequency,
        )  # 'M','D', etc.
        expected_dates = set(check_dates.unique())
        given_dates = set(df_historical["ds"].unique())
        remaining_dates = expected_dates.difference(given_dates)
        #print(remaining_dates)
        if len(check_dates) != len(df_historical["ds"]):
            raise Exception("The series starts on " + str(min_date) + " and ends on " + str(last_date) +
                    " the length of the series is expected to be " + str(len(check_dates)) + ", but " + str(len(df_historical["ds"])) + " was found ")
        dates = pd.date_range(
            start=last_date,
            periods=horizon + 1,  # An extra in case we include start
            freq=frequency,
        )  # 'M','D', etc.
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:horizon]  # Return correct number of periods
        if include_history:
            dates = np.concatenate((np.array(self.history_dates), dates))
        df_dates = pd.DataFrame({"ds": dates})
        df_samples = pd.DataFrame({"rep": np.arange(uncertainty_samples)})
        df_cross = df_dates.merge(df_samples, how="cross")
        # snaive model
        stan_model_file = resource_filename("tablespoon", "stan/snaive.stan")
        out_dir = resource_filename("tablespoon", "stan/out")
        model_stan = CmdStanModel(stan_file=stan_model_file)
        cmdstanpy_data = {"horizon": horizon, "T": len(self.y), "y": self.y, "lag": lag}
        fit = model_stan.sample(
            data=cmdstanpy_data,
            output_dir=out_dir,
            chains=1,
            seed=42,
            iter_sampling=uncertainty_samples,
        )
        df_fit = fit.draws_pd()
        df_fit = df_fit.loc[:, df_fit.columns.str.startswith("forecast")]
        np_predictions = (
            df_fit.to_numpy().transpose().reshape(uncertainty_samples * horizon, 1)
        )
        print(fit.summary())
        df_pred = pd.DataFrame(np_predictions, columns=["y_sim"])
        df_result = pd.concat([df_cross, df_pred], axis=1)
        return df_result
