# API

## ::: tablespoon.forecasters.Naive
    handler: python
    selection:
      members:
        predict
    rendering:
      show_root_heading: false
      show_source: false
      heading_level: 4

## ::: tablespoon.forecasters.Snaive
    handler: python
    selection:
      members:
        predict
    rendering:
      show_root_heading: false
      show_source: false
      heading_level: 4

## ::: tablespoon.forecasters.Mean
    handler: python
    selection:
      members:
        predict
    rendering:
      show_root_heading: false
      show_source: false
      heading_level: 4

# Data

## APPL

APPL stock price data. A time series data set with non-seasonal patterns

## ::: tablespoon.data.APPL
    handler: python
    selection:
      members:
        - APPL
    rendering:
      show_root_heading: true
      show_source: false
      heading_level: 4

## WALMART

Walmart sales for California from M5. A time series data set with seasonal patterns.

## ::: tablespoon.data.WALMART
    handler: python
    selection:
      members:
        - APPL
    rendering:
      show_root_heading: true
      show_source: false
      heading_level: 4