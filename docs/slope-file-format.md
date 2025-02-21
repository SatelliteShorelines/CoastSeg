# Acceptable File Formats for Slope

You can upload your own slopes to be used for tide correction. CoastSeg accepts the slopes from CSV files in formats below.

Note: The exact columns names (capitalizations too) must match

## Table of Contents

- [Acceptable CSV formats for Specific Dates](#acceptable-csv-formats-for-specific-dates)
- [Acceptable CSV formats for Seasonal Data](#acceptable-csv-formats-for-seasonal-data)


## Acceptable CSV formats for Specific Dates

- Note: The exact columns names (capitalizations too) must match

### Acceptable Forms of Dates

1. **Dates in ISO 8601** 
      - specify year-month-day followed by the time in 24-hour format with an offset from UTC (Coordinated Universal Time)
      - Example: `2023-12-25 18:40:14+00:00`
      - We recommend this format since this is the same format used to save the dates in the timeseries CSV files saved by CoastSeg
2. **Dates in YYYY-MM-DD**
      - Example 2021-04-05

### CSV Formats for Specific Dates

1. **Format 1: Slopes for each transect ID**

      > columns : transect_id, slope

      - Please be sure to supply a slope for each transect id
      - All your transect ids can be found in the raw timeseries csv files

      | transect_id| slope |
      | --------   | ------|
      | 1          | 0.05  |
      | 2          | 0.04  |
      | 3          | 0.02  |

2. **Format 2: Dates and slopes for each transect ID**

      > columns : transect_id, slope, dates

      - Uses the slope for everything up to and including the date for that transect
      - For example for transect 2 slope `0.04` will be used for all tide corrections for dates before or on 2021-04-05, but for dates after 2021-04-05 the slope `0.03` will be used

      | transect_id | slope | dates      |
      |-------------|-------|------------|
      | 1           | 0.05  | 2021-04-05 |
      | 2           | 0.04  | 2021-04-05 |
      | 2           | 0.03  | 2021-04-08 |
      | 3           | 0.02  | 2021-04-06 |

3. **Format 3: Dates and latitude & longitude values for each slope**

      > columns : slope,longitude,latitude,dates

      - Use the slope for the transect closet to the longitude(x) and latitude(y) location and the closest date

      | slope | latitude  |  longitude | dates |
      |-------|---------|----------|----------|
      | 0.05  | 34.0522 | -118.2437|2021-04-05|
      | 0.04  | 36.7783 | -119.4179|2021-04-07|
      | 0.02  | 37.7749 | -122.4194|2021-04-08|

4. **Form 4: Transect IDs as the columns and dates as the row index**

      > columns: transect ids (the actual transect ids)

      - This is a pivot of **format 2**
      - This format has the slope value for each transect id and date
      - In the examples below transect_1, 1234, and 123456 are transect ids

      |  | transect_1 |transect_2 | transect_3 |
      | -------- | ------- | ---------| ---------|
      | 2004-04-07 00:00:00+00:00          |  0.05                   | 0.04     | 0.04     |
      |2004-04-08 00:00:00+00:00       | 0.04       |0.05      | 0.08     |
      | 2004-04-09 00:00:00+00:00      | 0.03     |0.09    |0.07   |
      | 2004-04-10 00:00:00+00:00     | 0.02   |  0.01  | 0.07      |

## Acceptable CSV formats for Seasonal Data

Any of the following CSV formats will be accepted for seasonal data

- Note: The exact columns names (capitalizations too) must match
- Note: Any month that is missing will use the median slope
- Months : [1,2,3,4,5,6,7,8,9,10,11,12]

</br>

1. **Format 1: Slopes for each transect ID**

      > columns : month, slope

      - Uses that the slope for any shoreline that was detected during that month
      - For any shorelines that don't have a corresponding slope for that month the median of all the slopes is used
      - In this example, the slope 0.05 will be used to tidally correct all shorelines that occured in the 1st month, the slope 0.04 will be used to tidally correct shorelines that were in February, and the slope 0.02 will be used to tidally correct shorelines that were in March.

      | month   | slope  |
      | --------| -------|
      | 1       |  0.05  |
      | 2       | 0.04   |
      | 3       | 0.02   |

2. **Format 2: month and transect_id values for each slope**

      > columns : transect_id, slope, month

      - Uses that the slope for any shoreline that was detected during that month
      - For any shorelines that don't have a corresponding slope for that month the median of all the slopes is used
      - For example for transect 2 slope `0.04` will be used for all tide corrections for dates that occured in the 5th month May, but for shorelines in June the slope `0.03` will be used

      | transect_id | slope | month       |
      |-------------|-------|------------|
      | 1           | 0.05  | 4 |
      | 2           | 0.04  | 5 |
      | 2           | 0.03  | 6 |
      | 3           | 0.02  | 3 |
