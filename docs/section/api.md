# Methods

Each forecast class has the `predict` method. This is the same parameterization for each of the forecast classes.

## ::: tablespoon.forecasters.Mean.predict
    handler: python
    rendering:
      show_root_heading: true
      show_source: true
      heading_level: 4

## ::: tablespoon.forecasters.Naive.predict
    handler: python
    rendering:
      show_root_heading: true
      show_source: true
      heading_level: 4

## ::: tablespoon.forecasters.Snaive.predict
    handler: python
    rendering:
      show_root_heading: true
      show_source: true
      heading_level: 4

# CV Class

## ::: tablespoon.model_selection.TimeSeriesInitialSplit
    handler: python
    selection:
      members:
        split
    rendering:
      show_root_heading: true
      show_source: true
      heading_level: 4

# Forecaster Classes

## ::: tablespoon.forecasters.Naive
    handler: python
    selection:
      members:
        predict
    rendering:
      show_root_heading: true
      show_source: true
      heading_level: 4

## ::: tablespoon.forecasters.Snaive
    handler: python
    selection:
      members:
        predict
    rendering:
      show_root_heading: true
      show_source: true
      heading_level: 4

## ::: tablespoon.forecasters.Mean
    handler: python
    selection:
      members:
        predict
    rendering:
      show_root_heading: true
      show_source: true
      heading_level: 4

# Data

### APPL

APPL stock price data. A time series data set with non-seasonal patterns

### SEAS

A seasonal time series

### WALMART

Walmart sales for California from M5. A time series data set with seasonal patterns.
