from numpy import arange


class TimeSeriesInitialSplit():
    """Time Series cross-validator with initial period

    Provides time series splits for rolling origin type 
    cross validation. This means users may set an initial
    time period. gap size, and increment_size.

    Parameters:
        initial : int, default=21
            Number of splits.
        increment_size : int, default=7
            Sets the size of the test set to be added at each iteration
        gap : int, default=0
            Number of samples to exclude from the end of each train set before
            the test set.
    Examples:
    ```py
    import numpy as np
    from tablespoon.model_selection import TimeSeriesInitialSplit
    X = np.arange(0,50)
    tscv = TimeSeriesInitialSplit()
    for train_index, test_index in tscv.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]

    > TRAIN: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20] TEST: [21 22 23 24 25 26 27]
    > TRAIN: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27] TEST: [28 29 30 31 32 33 34]
    > TRAIN: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34] TEST: [35 36 37 38 39 40 41]
    > TRAIN: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41] TEST: [42 43 44 45 46 47 48]
    ```
    """

    def __init__(self, initial=7 * 3, increment_size=7, gap=0):
        self.initial = initial
        self.increment_size = increment_size
        self.gap = gap

    def split(self, X):
        """Generate indices to split data into training and test set.
        Parameters:
            X : array-like of shape (n_samples, n_features)
                Training data, where `n_samples` is the number of samples
                and `n_features` is the number of features.
        Yields: 
            train : ndarray
                The training set indices for that split.
            test : ndarray
                The testing set indices for that split.
        """
        n_samples = len(X)
        initial = self.initial
        gap = self.gap
        increment_size = self.increment_size

        # Make sure we have enough samples for the given split parameters
        if initial > n_samples:
            raise ValueError(
                f"Cannot have number of initial_size={initial} greater"
                f" than the number of samples={n_samples}."
            )
        if n_samples - initial - increment_size - gap < 0:
            raise ValueError(
                f"Size of initial + increment_size + gap too large given sample"
                f"={n_samples} with initial={initial} increment_size={increment_size} and gap={gap}."
            )

        indices = arange(n_samples)
        test_starts = range(initial, n_samples, increment_size)
        for test_start in test_starts:
            test = indices[test_start + gap: test_start + increment_size + gap]
            if len(test) < increment_size:
                # break if the test set is smaller than a complete increment_size
                break
            else:
                yield (
                    indices[:test_start],
                    indices[test_start + gap: test_start + increment_size + gap],
                )
