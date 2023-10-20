import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
import warnings

def get_sorted_dates(df_historical):
    """
    get historical dates and sort them
    """
    return pd.to_datetime(
        pd.Series(df_historical["ds"].unique(), name="ds")
    ).sort_values()


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
        "Please specify frequency of data. \n`D`- daily\n`1H` - hourly\n For more frequency aliases see https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases"
    )


def check_historical_dates_are_contiguous(
    history_dates, min_date, last_date, frequency
):
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
        warning_message = (
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
        warnings.warn(warning_message, UserWarning)


class Naive(object):
    """Naive Forecaster

    Args:
        object (None): instantiates a Naive Forecast object
    """

    def __init__(
        self,
        history_dates=None,
        include_history=False,
        y=None,
    ):
        self.include_history = include_history
        self.history_dates = history_dates

    def predict(
        self,
        df_historical,
        horizon=30,
        frequency=None,
        lag=1,
        uncertainty_samples=5000,
        include_history=False,
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

        Example:
            ```py
            import pandas as pd
            import tablespoon as tbsp
            from tablespoon.data import APPL
            df_APPLE = APPL
            df_APPLE = df_APPLE.assign(ds = lambda df: pd.to_datetime(df.ds))
            naive = tbsp.Naive()
            df_f = (naive.predict(df_APPLE, horizon=7*4, frequency="D", lag = 1, uncertainty_samples = 500).assign(model = 'naive'))
            df_f.head(10)
            ```
        """
        if frequency is None:
            send_helpful_frequency_error()
        self.y = df_historical["y"]
        self.history_dates = get_sorted_dates(df_historical)
        last_date = self.history_dates.max()
        min_date = self.history_dates.min()
        check_historical_dates_are_contiguous(
            self.history_dates, min_date, last_date, frequency
        )
        dates = pd.date_range(
            start=last_date, periods=horizon + 1, freq=frequency
        )  # An extra in case we include start  # 'M','D', etc.
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:horizon]  # Return correct number of periods
        if include_history:
            dates = np.concatenate((np.array(self.history_dates), dates))
        df_dates = pd.DataFrame({"ds": dates})
        df_samples = pd.DataFrame({"rep": np.arange(uncertainty_samples)})
        df_cross = df_dates.merge(df_samples, how="cross")
        # fit
        t = lag + 1
        t_lag = t - lag
        y = self.y.to_numpy()
        end = len(y) - lag
        yt = y[t:]
        yt_lag = y[t_lag:end]
        y_last = y.take(-1)
        mod = sm.GLM(yt, yt_lag, family=sm.families.Gaussian())
        sigma = np.sqrt(mod.fit().scale)
        rng = np.random.default_rng()
        forecast = np.empty([uncertainty_samples, horizon])
        for h in range(0, horizon):
            forecast[:, h] = y_last + sigma * np.sqrt(h + 1) * rng.standard_normal(
                uncertainty_samples
            )
        np_predictions = forecast.transpose().reshape(
            uncertainty_samples * horizon, 1
        )
        df_pred = pd.DataFrame(np_predictions, columns=["y_sim"])
        df_result = pd.concat([df_cross, df_pred], axis=1)
        return df_result
    
    def fitted_params(
        self,
        df_historical,
        horizon=30,
        frequency=None,
        lag=1,
        uncertainty_samples=5000,
        include_history=False,
    ):
        """Fitted Parameters - forecast method

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

        Example:
            ```py
            import pandas as pd
            import tablespoon as tbsp
            from tablespoon.data import APPL
            df_APPLE = APPL
            df_APPLE = df_APPLE.assign(ds = lambda df: pd.to_datetime(df.ds))
            naive = tbsp.Naive()
            df_f = (naive.predict(df_APPLE, horizon=7*4, frequency="D", lag = 1, uncertainty_samples = 500).assign(model = 'naive'))
            df_f.head(10)
            ```
        """
        if frequency is None:
            send_helpful_frequency_error()
        self.y = df_historical["y"]
        self.history_dates = get_sorted_dates(df_historical)
        last_date = self.history_dates.max()
        min_date = self.history_dates.min()
        check_historical_dates_are_contiguous(
            self.history_dates, min_date, last_date, frequency
        )
        dates = pd.date_range(
            start=last_date, periods=horizon + 1, freq=frequency
        )  # An extra in case we include start  # 'M','D', etc.
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:horizon]  # Return correct number of periods
        if include_history:
            dates = np.concatenate((np.array(self.history_dates), dates))
        df_dates = pd.DataFrame({"ds": dates})
        df_samples = pd.DataFrame({"rep": np.arange(uncertainty_samples)})
        df_cross = df_dates.merge(df_samples, how="cross")
        # fit
        t = lag + 1
        t_lag = t - lag
        y = self.y.to_numpy()
        end = len(y) - lag
        yt = y[t:]
        yt_lag = y[t_lag:end]
        y_last = y.take(-1)
        mod = sm.GLM(yt, yt_lag, family=sm.families.Gaussian())
        sigma = np.sqrt(mod.fit().scale)
        rng = np.random.default_rng()
        forecast = np.empty([uncertainty_samples, horizon])
        # fitted params
        params_list = []
        for h in range(0, horizon):
            params_dict = {}
            # make dictionary of parameters
            params_dict["ds"] = dates[h]
            params_dict["mu"] = np.round(y_last, 2)
            params_dict["sigma"] = np.round(sigma * np.sqrt(h + 1),2)
            mu = params_dict["mu"]; sd = params_dict["sigma"];
            params_dict['dist'] = f"N({mu}, {sd})"
            params_list.append(params_dict)
        return pd.DataFrame(params_list)



class Mean(object):
    """
    Mean Forecaster
    """

    def __init__(
        self,
        history_dates=None,
        include_history=False,
        y=None,
    ):
        self.include_history = include_history
        self.history_dates = history_dates

    def predict(
        self,
        df_historical,
        horizon=30,
        frequency=None,
        uncertainty_samples=5000,
        include_history=False,
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

        Example:
            ```py
            import pandas as pd
            import tablespoon as tbsp
            from tablespoon.data import APPL
            df_APPLE = APPL
            df_APPLE = df_APPLE.assign(ds = lambda df: pd.to_datetime(df.ds))
            mean = tbsp.Mean()
            df_f = (n.predict(df_APPLE, horizon=7*4, frequency="D", lag = 1, uncertainty_samples = 500).assign(model = 'mean'))
            df_f.head(10)
            ```
        """
        self.y = df_historical["y"]
        self.history_dates = get_sorted_dates(df_historical)
        last_date = self.history_dates.max()
        min_date = self.history_dates.min()
        check_historical_dates_are_contiguous(
            self.history_dates, min_date, last_date, frequency
        )
        dates = pd.date_range(
            start=last_date, periods=horizon + 1, freq=frequency
        )  # An extra in case we include start  # 'M','D', etc.
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:horizon]  # Return correct number of periods
        if include_history:
            dates = np.concatenate((np.array(self.history_dates), dates))
        df_dates = pd.DataFrame({"ds": dates})
        df_samples = pd.DataFrame({"rep": np.arange(uncertainty_samples)})
        df_cross = df_dates.merge(df_samples, how="cross")
        # fit
        y = self.y
        T = len(y)
        deg_freedom = T - 1
        mu, sigma = norm.fit(y)
        rng = np.random.default_rng()
        forecast = np.empty([uncertainty_samples, horizon])
        for h in range(0, horizon):
            forecast[:, h] = mu + sigma * np.sqrt(1 + (1 / T)) * rng.standard_t(
                df=deg_freedom, size=uncertainty_samples
            )
        np_predictions = forecast.transpose().reshape(
            uncertainty_samples * horizon, 1
        )
        df_pred = pd.DataFrame(np_predictions, columns=["y_sim"])
        df_result = pd.concat([df_cross, df_pred], axis=1)
        return df_result
    
    def fitted_params(
        self,
        df_historical,
        horizon=30,
        frequency=None,
        uncertainty_samples=5000,
        include_history=False,
    ):
        """Fitted Parameters

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

        Example:
            ```py
            import pandas as pd
            import tablespoon as tbsp
            from tablespoon.data import APPL
            df_APPLE = APPL
            df_APPLE = df_APPLE.assign(ds = lambda df: pd.to_datetime(df.ds))
            mean = tbsp.Mean()
            df_f = (n.predict(df_APPLE, horizon=7*4, frequency="D", lag = 1, uncertainty_samples = 500).assign(model = 'mean'))
            df_f.head(10)
            ```
        """
        self.y = df_historical["y"]
        self.history_dates = get_sorted_dates(df_historical)
        last_date = self.history_dates.max()
        min_date = self.history_dates.min()
        check_historical_dates_are_contiguous(
            self.history_dates, min_date, last_date, frequency
        )
        dates = pd.date_range(
            start=last_date, periods=horizon + 1, freq=frequency
        )  # An extra in case we include start  # 'M','D', etc.
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:horizon]  # Return correct number of periods
        if include_history:
            dates = np.concatenate((np.array(self.history_dates), dates))
        df_dates = pd.DataFrame({"ds": dates})
        df_samples = pd.DataFrame({"rep": np.arange(uncertainty_samples)})
        df_cross = df_dates.merge(df_samples, how="cross")
        # fit
        y = self.y
        T = len(y)
        deg_freedom = T - 1
        mu, sigma = norm.fit(y)
        rng = np.random.default_rng()
        forecast = np.empty([uncertainty_samples, horizon])
        for h in range(0, horizon):
            forecast[:, h] = mu + sigma * np.sqrt(1 + (1 / T)) * rng.standard_t(
                df=deg_freedom, size=uncertainty_samples
            )
        np_predictions = forecast.transpose().reshape(
            uncertainty_samples * horizon, 1
        )
        df_pred = pd.DataFrame(np_predictions, columns=["y_sim"])
        df_result = pd.concat([df_cross, df_pred], axis=1)
        ## fitted params
        params_list = []
        for h in range(0, horizon):
            params_dict = {}
            # make dictionary of parameters
            params_dict["ds"] = dates[h]
            params_dict["mu"] = np.round(mu, 2)
            params_dict["sigma"] = np.round(sigma * np.sqrt(1 + (1 / T)),2)
            mu = params_dict["mu"]; sd = params_dict["sigma"];
            params_dict['dist'] = f"N({mu}, {sd})"
            params_list.append(params_dict)
        return pd.DataFrame(params_list)


class Snaive(object):
    """
    Seasonal Naive Forecaster
    """

    def __init__(
        self,
        history_dates=None,
        include_history=False,
        y=None,
    ):
        self.include_history = include_history
        self.history_dates = history_dates
        self.include_history = include_history

    def predict(
        self,
        df_historical,
        horizon=30,
        frequency=None,
        lag=7,
        uncertainty_samples=5000,
        include_history=False,
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

        Example:
            ```py
            import tablespoon as tbsp
            from tablespoon.data import SEAS
            sn = tbsp.Snaive()
            df_f = sn.predict(SEAS, horizon=7 * 4, frequency="D", lag=7, uncertainty_samples=800).assign(model="snaive")
            df_f.head(10)
            ```
        """
        self.y = df_historical["y"]
        self.history_dates = get_sorted_dates(df_historical)
        last_date = self.history_dates.max()
        min_date = self.history_dates.min()
        check_historical_dates_are_contiguous(
            self.history_dates, min_date, last_date, frequency
        )
        dates = pd.date_range(
            start=last_date, periods=horizon + 1, freq=frequency
        )  # An extra in case we include start  # 'M','D', etc.
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:horizon]  # Return correct number of periods
        if include_history:
            dates = np.concatenate((np.array(self.history_dates), dates))
        df_dates = pd.DataFrame({"ds": dates})
        df_samples = pd.DataFrame({"rep": np.arange(uncertainty_samples)})
        df_cross = df_dates.merge(df_samples, how="cross")
        # fit
        y = self.y.to_numpy()
        last_start = len(y) - lag
        last_end = len(y)
        yt = y[lag:last_end]
        yt_lag = y[0:last_start]
        mod = sm.GLM(yt, yt_lag, family=sm.families.Gaussian())
        sigma = np.sqrt(mod.fit().scale) # attempt to use glm to get sigma
        rng = np.random.default_rng()
        forecast = np.empty([uncertainty_samples, horizon])
        for h in range(0, horizon):
            forecast[:, h] = y[(len(y)) - (lag - ((h) % lag))] + sigma * np.sqrt(np.trunc(((h) * 1) / (lag)) + 1) * rng.standard_normal(uncertainty_samples)
        np_predictions = forecast.transpose().reshape(
            uncertainty_samples * horizon, 1
        )
        df_pred = pd.DataFrame(np_predictions, columns=["y_sim"])
        df_result = pd.concat([df_cross, df_pred], axis=1)
        return df_result

    def fitted_params(
        self,
        df_historical,
        horizon=30,
        frequency=None,
        lag=7,
        uncertainty_samples=5000,
        include_history=False,
    ):
        """Fitted Parameters - Normal(mu, sigma)

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
            dictionary: A dictionary of fitted parameters as `ds`, `mu`, and `sigma`

        Example:
            ```py
            import tablespoon as tbsp
            from tablespoon.data import SEAS
            sn = tbsp.Snaive()
            df_f = sn.fitted_params(SEAS, horizon=7 * 4, frequency="D", lag=7, uncertainty_samples=800).assign(model="snaive")
            df_f.head(10)
            ```
        """
        self.y = df_historical["y"]
        self.history_dates = get_sorted_dates(df_historical)
        last_date = self.history_dates.max()
        min_date = self.history_dates.min()
        check_historical_dates_are_contiguous(
            self.history_dates, min_date, last_date, frequency
        )
        dates = pd.date_range(
            start=last_date, periods=horizon + 1, freq=frequency
        )  # An extra in case we include start  # 'M','D', etc.
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:horizon]  # Return correct number of periods
        if include_history:
            dates = np.concatenate((np.array(self.history_dates), dates))
        df_dates = pd.DataFrame({"ds": dates})
        df_samples = pd.DataFrame({"rep": np.arange(uncertainty_samples)})
        df_cross = df_dates.merge(df_samples, how="cross")
        # fit
        y = self.y.to_numpy()
        last_start = len(y) - lag
        last_end = len(y)
        yt = y[lag:last_end]
        yt_lag = y[0:last_start]
        mod = sm.GLM(yt, yt_lag, family=sm.families.Gaussian())
        # the variable sigma represents the estimated standard deviation of the residuals from the generalized linear model (GLM) fit.
        sigma = np.sqrt(mod.fit().scale)
        params_list = []
        for h in range(0, horizon):
            params_dict = {}
            # make dictionary of parameters
            params_dict["ds"] = dates[h]
            params_dict["mu"] = y[(len(y)) - (lag - ((h) % lag))]
            params_dict["sigma"] = np.round(sigma * np.sqrt(np.trunc(((h) * 1) / (lag)) + 1),2)
            mu = params_dict["mu"]; sd = params_dict["sigma"];
            params_dict['dist'] = f"N({mu}, {sd})"
            params_list.append(params_dict)
        return pd.DataFrame(params_list)