# Acceptable File Formats for Tide

You can upload your own tides to be used for tide correction. CoastSeg accepts the tides from CSV files in formats below.

Note: The exact columns names (capitalizations too) must match

## Acceptable Forms for dates

1. Dates in ISO 8601, specifying year-month-day followed by the time in 24-hour format with an offset from UTC (Coordinated Universal Time)
      - Example 2023-12-25 18:40:14+00:00
      - We recommend this format since this is the same format used to save the dates in the timeseries CSV files saved by CoastSeg
2. Dates in YYY-MM-DD
      - Example 2021-04-05

## Acceptable CSV formats
Any of the following CSV formats will be accepted.
- Note: The exact columns names (capitalizations too) must match

1. Format 1: Tides for each transect ID
      - columns : transect_id, tide
      - Please be sure to supply a tide for each transect id
      - All your transect ids can be found in the raw timeseries csv files


      | transect_id   | tide | 
      | -------- | ------- | 
      | 1          |  0.05                    |
      | 2         | 0.04        |
      | 3         | 0.02   | |

2. Format 2: Dates and tides for each transect ID
      - columns : transect_id, tide, dates
      - Uses the tide for everything up to and including the date for that transect
      - For example for transect 2 tide `0.04` will be used for all tide corrections for dates before or on 2021-04-05, but for dates after 2021-04-05 the tide `0.03` will be used

      | transect_id | tide | dates       |
      |-------------|-------|------------|
      | 1           | 0.05  | 2021-04-05 |
      | 2           | 0.04  | 2021-04-05 |
      | 2           | 0.03  | 2021-04-08 |
      | 3           | 0.02  | 2021-04-06 |

3. Format 3: Dates and latitude & longitude values for each tide
      - columns : tide,longitude,latitude,dates
      - Use the tide for the transect closet to the longitude(x) and latitude(y) location and the closest date

      | tide | latitude  |  longitude | dates |
      |-------|---------|----------|----------|
      | 0.05  | 34.0522 | -118.2437|2021-04-05|
      | 0.04  | 36.7783 | -119.4179|2021-04-07|
      | 0.02  | 37.7749 | -122.4194|2021-04-08|

4. Form 4: Transect IDs as the columns and dates as the row index

      - columns: transect_ids
      - Basically its a pivot of form 2
      - This format has the tide value for each transect id and date
      - In the examples below 123, 1234 ... are transect ids

      | 123 | 1234 |12345 | 123456 |
      | -------- | ------- | ---------| ---------|
      | 2004-04-07 00:00:00+00:00          |  0.05                   | 0.04     | 0.04     |
      |2004-04-08 00:00:00+00:00       | 0.04       |0.05      | 0.08     |
      | 2004-04-09 00:00:00+00:00      | 0.03     |0.09    |0.07   |
      | 2004-04-10 00:00:00+00:00     | 0.02   |  0.01  | 0.07      |