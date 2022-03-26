import logging
import shutil
import tempfile
import os
from pkg_resources import resource_filename

from cmdstanpy import CmdStanModel

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from scipy.stats import norm

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
        use_stan_backend=True, 
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
            use_stan_backend (bool, optional): chose to use either standard python (False) or stan backend (True)
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
        if use_stan_backend:
            df_fit = fit_stan_model("naive", self.y, lag, uncertainty_samples, horizon, chain_ids, verbose=verbose)
            #forecast[1] forecast[2]... forecast[28]
            np_predictions = df_fit.to_numpy().transpose().reshape(uncertainty_samples * horizon, 1)
            df_pred = pd.DataFrame(np_predictions, columns=["y_sim"])
            df_result = pd.concat([df_cross, df_pred], axis=1)
        else:
            t = lag+1
            t_lag = t-lag
            y = self.y
            end = len(y)-lag
            yt = y[t:].to_numpy()
            yt_lag = y[t_lag:end].to_numpy()
            y_last = self.y[len(y)-1]
            mod = sm.GLM(yt, yt_lag, family=sm.families.Gaussian())
            sigma = np.sqrt(mod.fit().scale)
            rng = np.random.default_rng()
            forecast = np.empty([uncertainty_samples, horizon])
            for h in range(0, horizon):
                forecast[:,h] = y_last + sigma * np.sqrt(h+1) * rng.standard_normal(uncertainty_samples)
            np_predictions = forecast.transpose().reshape(uncertainty_samples * horizon, 1)
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

    def predict(self, df_historical, horizon=30, frequency=None, lag=1, uncertainty_samples=5000, include_history=False, chain_ids=None,use_stan_backend=True, verbose=False):
        """Predict - forecast method

        Args:
            df_historical (pd.DataFrame): A date sorted dataframe with the columns `ds` and `y`
            horizon (int, optional): Forecast horizon. Defaults to 30.
            frequency (int, optional): number of rows that make a seasonal period. Defaults to None.
            lag (int, optional): number of rows that make a seasonal period. Defaults to 1.
            uncertainty_samples (int, optional): number of uncertainty samples to draw. Defaults to 5000.
            include_history (bool, optional): include history. Defaults to False.
            chain_ids (str, optional): identifiers for chain ids. Defaults to None.
            use_stan_backend (bool, optional): chose to use either standard python (False) or stan backend (True)
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
        if use_stan_backend:
            df_fit = fit_stan_model("mean", self.y, lag, uncertainty_samples, horizon, chain_ids, verbose=verbose)
            np_predictions = df_fit.to_numpy().transpose().reshape(uncertainty_samples * horizon, 1)
            df_pred = pd.DataFrame(np_predictions, columns=["y_sim"])
            df_result = pd.concat([df_cross, df_pred], axis=1)
        else:
            y = self.y
            T = len(y)
            deg_freedom = T - 1
            mu, sigma = norm.fit(y)
            rng = np.random.default_rng()
            forecast = np.empty([uncertainty_samples, horizon])
            for h in range(0, horizon):
                forecast[:,h] = mu + sigma * np.sqrt(1 + (1 / T)) * rng.standard_t(df = deg_freedom, size = uncertainty_samples)
            np_predictions = forecast.transpose().reshape(uncertainty_samples * horizon, 1)
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
        use_stan_backend=True, 
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
            use_stan_backend (bool, optional): chose to use either standard python (False) or stan backend (True)
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
        if use_stan_backend:
            df_fit = fit_stan_model("snaive", self.y, lag, uncertainty_samples, horizon, chain_ids, verbose=verbose)
            np_predictions = df_fit.to_numpy().transpose().reshape(uncertainty_samples * horizon, 1)
            df_pred = pd.DataFrame(np_predictions, columns=["y_sim"])
            df_result = pd.concat([df_cross, df_pred], axis=1)
        else:
            t = lag+1
            t_lag = t-lag
            y = self.y
            end = len(y)-lag
            yt = y[t:].to_numpy()
            yt_lag = y[t_lag:end].to_numpy()
            y_last = self.y[(len(y)-1)-(lag-(horizon%lag))]
            mod = sm.GLM(yt, yt_lag, family=sm.families.Gaussian())
            sigma = np.sqrt(mod.fit().scale)
            rng = np.random.default_rng()
            forecast = np.empty([uncertainty_samples, horizon])
            for h in range(0, horizon):
                forecast[:,h] = y_last + sigma * np.sqrt(np.trunc(((h-1)*1)/(lag)) + 1) * rng.standard_normal(uncertainty_samples)
            np_predictions = forecast.transpose().reshape(uncertainty_samples * horizon, 1)
            df_pred = pd.DataFrame(np_predictions, columns=["y_sim"])
            df_result = pd.concat([df_cross, df_pred], axis=1)
        return df_result

#HEAD###########################################################

from pandas import read_csv
from io import StringIO

APPL = read_csv(
      StringIO(
"""
ds,y
2021-08-01,4.986069344321214
2021-08-02,4.986069344321214
2021-08-03,4.982304387584444
2021-08-04,4.992267665749145
2021-08-05,4.9902964940324095
2021-08-06,4.986001054842818
2021-08-07,4.986001054842818
2021-08-08,4.986001054842818
2021-08-09,4.984975526441821
2021-08-10,4.986615804923752
2021-08-11,4.983949049729045
2021-08-12,4.9849071622237515
2021-08-13,5.00374495158579
2021-08-14,5.00374495158579
2021-08-15,5.00374495158579
2021-08-16,5.000854237042097
2021-08-17,5.012167424634866
2021-08-18,5.009301091455334
2021-08-19,4.976940609155258
2021-08-20,4.99342132992021
2021-08-21,4.99342132992021
2021-08-22,4.99342132992021
2021-08-23,4.999304661292363
2021-08-24,5.006961868310022
2021-08-25,5.009367808232606
2021-08-26,4.999574387879504
2021-08-27,4.993692544396178
2021-08-28,4.993692544396178
2021-08-29,4.993692544396178
2021-08-30,5.003946305945459
2021-08-31,5.028213250358988
2021-09-01,5.029326204520735
2021-09-02,5.036108058335924
2021-09-03,5.035392909496409
2021-09-04,5.035392909496409
2021-09-05,5.035392909496409
2021-09-06,5.035392909496409
2021-09-07,5.043231557676272
2021-09-08,5.056118381482073
2021-09-09,5.04658145619781
2021-09-10,5.043425116919247
2021-09-11,5.043425116919247
2021-09-12,5.043425116919247
2021-09-13,5.01482653113066
2021-09-14,5.012965950029919
2021-09-15,5.000988900610644
2021-09-16,5.000180852639488
2021-09-17,5.002737571184399
2021-09-18,5.002737571184399
2021-09-19,5.002737571184399
2021-09-20,4.968423466509184
2021-09-21,4.969327019387211
2021-09-22,4.972933405785502
2021-09-23,4.9880487538038505
2021-09-24,4.981275163931258
2021-09-25,4.981275163931258
2021-09-26,4.981275163931258
2021-09-27,4.979969888176813
2021-09-28,4.964591355594849
2021-09-29,4.959131459797281
2021-09-30,4.967449422138158
2021-10-01,4.955122541153201
2021-10-02,4.955122541153201
2021-10-03,4.955122541153201
2021-10-04,4.954135448107023
2021-10-05,4.937992953484486
2021-10-06,4.937849533123045
2021-10-07,4.963264105614627
2021-10-08,4.97002160273562
2021-10-09,4.97002160273562
2021-10-10,4.97002160273562
2021-10-11,4.957726690693728
2021-10-12,4.964451699962401
2021-10-13,4.950460609952596
2021-10-14,4.956601409898694
2021-10-15,4.968214830151833
2021-10-16,4.968214830151833
2021-10-17,4.968214830151833
2021-10-18,4.965986521153356
2021-10-19,4.990500574309953
2021-10-20,5.0019308329431915
2021-10-21,5.0026703080357615
2021-10-22,5.008566505236892
2021-10-23,5.008566505236892
2021-10-24,5.008566505236892
2021-10-25,5.001796296167423
2021-10-26,5.006158634330978
2021-10-27,5.0063595033199135
2021-10-28,5.009434622406525
2021-10-29,4.991928074922253
2021-10-30,4.991928074922253
2021-10-31,4.991928074922253
2021-11-01,5.0038792264685945
2021-11-02,5.0016618439312746
2021-11-03,5.013231915885063
2021-11-04,5.0211135504636735
2021-11-05,5.023156570631548
2021-11-06,5.023156570631548
2021-11-07,5.023156570631548
2021-11-08,5.019991413206988
2021-11-09,5.011967719012072
2021-11-10,5.0107686470207655
2021-11-11,5.003677858600367
2021-11-12,5.000113417292074
2021-11-13,5.000113417292074
2021-11-14,5.000113417292074
2021-11-15,5.013098891062242
2021-11-16,5.010235230357471
2021-11-17,5.017279836814924
2021-11-18,5.035067753915632
2021-11-19,5.060377347275461
2021-11-20,5.060377347275461
2021-11-21,5.060377347275461
2021-11-22,5.085619027794655
2021-11-23,5.082149398664812
2021-11-24,5.079850363117729
2021-11-25,5.079850363117729
2021-11-26,5.072482743322057
2021-11-27,5.072482743322057
2021-11-28,5.072482743322057
2021-11-29,5.071228512233326
2021-11-30,5.075111347615041
2021-12-01,5.120863915640666
2021-12-02,5.067267678267517
2021-12-03,5.099988397656679
2021-12-04,5.099988397656679
2021-12-05,5.099988397656679
2021-12-06,5.101633118052293
2021-12-07,5.130371986528106
2021-12-08,5.1482500336504184
2021-12-09,5.164271576856079
2021-12-10,5.165985292817968
2021-12-11,5.165985292817968
2021-12-12,5.165985292817968
2021-12-13,5.199159768055827
2021-12-14,5.166213525914699
2021-12-15,5.165414351369342
2021-12-16,5.188948822683752
2021-12-17,5.135386544444626
2021-12-18,5.135386544444626
2021-12-19,5.135386544444626
2021-12-20,5.12562925146832
2021-12-21,5.144933045418053
2021-12-22,5.1535227428452925
2021-12-23,5.169631393628638
2021-12-24,5.169631393628638
2021-12-25,5.169631393628638
2021-12-26,5.169631393628638
2021-12-27,5.1766580572413385
2021-12-28,5.193845365278309
2021-12-29,5.189227694170865
2021-12-30,5.1900080698428965
2021-12-31,5.182289019924663
2022-01-01,5.182289019924663
""")
)

n = Mean()
df_n1 = n.predict(
    APPL, horizon=7*3, frequency="D", lag=1, uncertainty_samples=1000, use_stan_backend=True
).assign(model="naive")

df_n2 = n.predict(
    APPL, horizon=7*3, frequency="D", lag=1, uncertainty_samples=1000, use_stan_backend=False
).assign(model="naive")

print(df_n1.y_sim.mean())
print(df_n1.y_sim.var())
print(df_n2.y_sim.mean())
print(df_n2.y_sim.var())