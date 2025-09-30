# Standard library imports
import logging
import os
import pathlib
import re
import traceback
from enum import Enum
from pathlib import Path
from typing import Collection, Dict, Iterable, List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import pyTMD.io
import pyTMD.io.model
import pyTMD.predict
import pyTMD.time
import pytz

# Third-party imports
from tqdm import tqdm

from coastseg import common, core_utilities, file_utilities
from coastseg.file_utilities import progress_bar_context

# Logger setup
logger = logging.getLogger(__name__)


class TimeSeriesTypes(str, Enum):
    RAW = "raw"
    TIDALLY_CORRECTED = "tidally_corrected"


# Compiled filename patterns, ordered by preference (preferred -> legacy).
_PATTERNS: dict[tuple[TimeSeriesTypes, bool], tuple[re.Pattern[str], ...]] = {
    (TimeSeriesTypes.RAW, False): (
        re.compile(r"^raw_transect_time_series\.csv$"),
        re.compile(r"^transect_time_series\.csv$"),
    ),
    (TimeSeriesTypes.RAW, True): (
        re.compile(r"^raw_transect_time_series_merged\.csv$"),
        re.compile(r"^transect_time_series_merged\.csv$"),
    ),
    (TimeSeriesTypes.TIDALLY_CORRECTED, False): (
        re.compile(r"^tidally_corrected_transect_time_series\.csv$"),
    ),
    (TimeSeriesTypes.TIDALLY_CORRECTED, True): (
        re.compile(r"^tidally_corrected_transect_time_series_merged\.csv$"),
    ),
}


def _find_first_by_patterns(
    session_path: str, patterns: Iterable[re.Pattern[str]]
) -> str:
    """
    Finds the first file matching any of the provided regex patterns.

    Args:
        session_path (str): Directory path to search for files.
        patterns (Iterable[re.Pattern[str]]): Regex patterns to match against filenames.

    Returns:
        str: Path to the first matching file.

    Raises:
        FileNotFoundError: If no file matches any pattern.
    """
    for pat in patterns:
        try:
            return file_utilities.find_file_by_regex(session_path, pat.pattern)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("No file matched any expected pattern.")


def get_timeseries_path(
    roi_id: str,
    session_name: str,
    series: TimeSeriesTypes = TimeSeriesTypes.RAW,
    is_merged: bool = False,
) -> str:
    """Return the path to a time-series CSV for an ROI and session.

    Builds the expected filename from the series type and is_ flag. For raw
    series, legacy filenames are supported as fallbacks.

    Args:
        roi_id: Region-of-interest identifier.
        session_name: Session name.
        series: Series type, either Series.RAW or Series.TIDALLY_CORRECTED. Defaults to Series.RAW.
        is_merged: Whether to return the merged file variant.

    Returns:
        Path to the matching CSV file.

    Raises:
        FileNotFoundError: If no matching file is found in the session directory.
        KeyError: If an invalid (series, is_merged) combination is requested.

    Examples:
        >>> get_timeseries_path("roi_001", "my_session", Series.RAW, is_merged=False)
        '/path/to/session/roi_001/raw_transect_time_series.csv'
    """
    session_path = file_utilities.get_session_contents_location(session_name, roi_id)
    patterns = _PATTERNS[(series, is_merged)]
    return _find_first_by_patterns(session_path, patterns)


def convert_col_to_ISO_8601(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Converts a DataFrame column to ISO 8601 format and timezone-aware datetime objects.

    Args:
        df (pd.DataFrame): The input DataFrame containing the column to convert.
        col_name (str): Name of the column to convert to ISO 8601 format.

    Returns:
        pd.DataFrame: DataFrame with the specified column converted to timezone-aware datetime objects in UTC.
    """
    if col_name not in df.columns:
        return df
    df[col_name] = pd.to_datetime(df[col_name], format="ISO8601")
    # Specify the desired timezone (e.g., UTC)
    timezone = pytz.timezone("UTC")
    # Convert the naive datetime objects to timezone-aware datetime objects
    df[col_name] = df[col_name].apply(
        lambda x: x if x.tzinfo is not None else timezone.localize(x)
    )
    return df


def compute_distance_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Computes the Euclidean distance between two points in 2D space.

    Args:
        x1 (float): X-coordinate of the first point.
        y1 (float): Y-coordinate of the first point.
        x2 (float): X-coordinate of the second point.
        y2 (float): Y-coordinate of the second point.

    Returns:
        float: Euclidean distance between the two points.

    Example:
        >>> distance = compute_distance_xy(0, 0, 3, 4)
        >>> print(distance)
        5.0
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _initialize_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Initializes a new column with NaN values if it doesn't exist.

    Args:
        df (pd.DataFrame): DataFrame to modify.
        column_name (str): Name of the column to initialize.

    Returns:
        pd.DataFrame: DataFrame with the column initialized if it didn't exist.
    """
    if column_name not in df.columns:
        df[column_name] = np.nan
    return df


def _find_closest_date(
    date: pd.Timestamp, reference_dates: pd.Series
) -> Union[pd.Timestamp, float]:
    """
    Finds the closest date from a series of reference dates.

    Args:
        date (pd.Timestamp): Target date to find the closest match for.
        reference_dates (pd.Series): Series containing reference dates to search within.

    Returns:
        Union[pd.Timestamp, float]: Date from reference_dates closest to the target date.
    """
    copy_of_reference_dates = reference_dates.copy()
    # reset index so that we can use the index to find the closest date
    copy_of_reference_dates.reset_index(drop=True, inplace=True)
    return copy_of_reference_dates.iloc[
        (copy_of_reference_dates - date).abs().argsort()[0]
    ]


def match_via_month(
    timeseries: pd.DataFrame, df: pd.DataFrame, column_name: str = "slope"
) -> pd.DataFrame:
    """
    Matches timeseries data with reference data based on month and adds the specified column.

    Args:
        timeseries (pd.DataFrame): DataFrame containing timeseries data with 'dates' column.
        df (pd.DataFrame): Reference DataFrame with 'month' column and specified column for matching.
        column_name (str, optional): Name of the column to match and add. Defaults to 'slope'.

    Returns:
        pd.DataFrame: Timeseries DataFrame with the specified column added, temporary 'month' column removed.

    Example:
        >>> timeseries_df = pd.DataFrame({
        ...     'dates': pd.to_datetime(['2021-01-15', '2021-06-15']),
        ...     'transect_id': ['1', '2']
        ... })
        >>> slope_df = pd.DataFrame({
        ...     'month': [1, 6],
        ...     'slope': [0.1, 0.2]
        ... })
        >>> result = match_via_month(timeseries_df, slope_df, 'slope')
        >>> print(result)
            dates transect_id  slope
        0 2021-01-15           1    0.1
        1 2021-06-15           2    0.2
    """
    # if the column already exists drop it else it will cause a duplicate column to appear after the merge
    if column_name in timeseries.columns:
        timeseries = timeseries.drop(columns=[column_name], errors="ignore")

    median_val = df[column_name].median()
    # For each transect, get the month of the date and add the slope
    timeseries["month"] = timeseries["dates"].dt.month
    timeseries = timeseries.merge(df, on="month", how="left")
    timeseries[column_name] = timeseries[column_name].fillna(median_val)
    # drop month column
    timeseries = timeseries.drop(columns=["month"])
    return timeseries


# Only cares about column name, 'transect_id', and 'dates' column
def match_via_id_and_month(
    timeseries: pd.DataFrame, df: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    """
    Matches values based on transect_id and closest month.

    Args:
        timeseries (pd.DataFrame): DataFrame containing shoreline data with 'dates' and 'transect_id' columns.
        df (pd.DataFrame): Reference DataFrame with 'transect_id', 'month', and specified column.
        column_name (str): Name of column to match and add to the timeseries.

    Returns:
        pd.DataFrame: Updated timeseries DataFrame with matched values. If column_name is 'slope',
                     NaN values are filled with the median slope from the reference DataFrame.

    Example:
        >>> timeseries_df = pd.DataFrame({
        ...     'dates': pd.to_datetime(['2021-01-15', '2021-06-15']),
        ...     'transect_id': ['1', '1']
        ... })
        >>> slope_df = pd.DataFrame({
        ...     'transect_id': ['1', '1'],
        ...     'month': [1, 6],
        ...     'slope': [0.1, 0.2]
        ... })
        >>> result = match_via_id_and_month(timeseries_df, slope_df, 'slope')
        >>> print(result)
            dates transect_id  slope
        0 2021-01-15           1    0.1
        1 2021-06-15           1    0.2
    """
    # if the column already exists drop it else it will cause a duplicate column to appear after the merge
    if column_name in timeseries.columns:
        timeseries = timeseries.drop(columns=[column_name], errors="ignore")

    df["transect_id"] = df["transect_id"].astype(str)
    timeseries["month"] = timeseries["dates"].dt.month
    timeseries = timeseries.merge(df, on=["month", "transect_id"], how="left")
    timeseries = timeseries.drop(columns=["month"])

    if column_name == "slope":
        median_slope = np.median(np.unique(df[column_name]))
        # Fill NaN values in the 'slope' column with the calculated median
        timeseries[column_name].fillna(median_slope, inplace=True)

    return timeseries


# Only cares about column name and 'dates' column
def match_via_date(
    timeseries: pd.DataFrame, df: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    """
    Matches values based on closest date.

    Args:
        timeseries (pd.DataFrame): DataFrame containing shoreline data with 'dates' column.
        df (pd.DataFrame): Reference DataFrame with 'dates' and specified column.
        column_name (str): Name of column to match and add to the timeseries.

    Returns:
        pd.DataFrame: Updated timeseries DataFrame with matched values added.

    Example:
        >>> timeseries_df = pd.DataFrame({
        ...     'dates': pd.to_datetime(['2021-01-15', '2021-06-15'])
        ... })
        >>> reference_df = pd.DataFrame({
        ...     'dates': pd.to_datetime(['2021-01-10', '2021-06-20']),
        ...     'tide': [1.2, 1.8]
        ... })
        >>> result = match_via_date(timeseries_df, reference_df, 'tide')
        >>> print(result)
            dates  tide
        0 2021-01-15   1.2
        1 2021-06-15   1.8
    """
    timeseries = _initialize_column(timeseries, column_name)

    # Loop through shoreline dates
    for date in timeseries["dates"]:
        closest_date = _find_closest_date(date, df["dates"])
        timeseries.loc[timeseries["dates"] == date, column_name] = df.loc[
            df["dates"] == closest_date, column_name
        ].values[0]

    return timeseries


# Only cares about column name, 'transect_id', and 'dates' column
def match_via_id_and_date(
    timeseries: pd.DataFrame, df: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    """
    Matches values based on transect_id and closest date.

    Performs temporal matching for each transect, finding the closest date
    in the reference DataFrame for each shoreline observation.

    Args:
        timeseries (pd.DataFrame): DataFrame containing shoreline data with 'dates' and 'transect_id' columns.
        df (pd.DataFrame): Reference DataFrame with 'dates', 'transect_id', and specified column.
        column_name (str): Name of column to match and add to the timeseries.

    Returns:
        pd.DataFrame: Updated timeseries DataFrame with matched values. If column_name is 'slope',
                     NaN values are filled with the median slope from the reference DataFrame.

    Example:
        >>> timeseries_df = pd.DataFrame({
        ...     'dates': pd.to_datetime(['2021-01-15', '2021-06-15']),
        ...     'transect_id': ['1', '1']
        ... })
        >>> reference_df = pd.DataFrame({
        ...     'dates': pd.to_datetime(['2021-01-10', '2021-06-20']),
        ...     'transect_id': ['1', '1'],
        ...     'slope': [0.1, 0.2]
        ... })
        >>> result = match_via_id_and_date(timeseries_df, reference_df, 'slope')
        >>> print(result)
            dates transect_id  slope
        0 2021-01-15           1    0.1
        1 2021-06-15           1    0.2
    """
    timeseries = _initialize_column(timeseries, column_name)
    unique_ids = np.unique(df["transect_id"])

    for transect_id in unique_ids:
        matching_rows = df.loc[df["transect_id"] == transect_id]

        if len(matching_rows) == 1:
            # Single tide value case
            matching_value = matching_rows[column_name].values[0]
            timeseries.loc[timeseries["transect_id"] == transect_id, column_name] = (
                matching_value
            )
        elif len(matching_rows) > 1:
            # Multiple dates case
            shoreline_dates = timeseries.loc[
                timeseries["transect_id"] == transect_id, "dates"
            ]
            for shoreline_date in shoreline_dates:
                # gets the closest date from the df data for that transect_id
                matching_date = _find_closest_date(
                    shoreline_date, matching_rows["dates"]
                )
                matching_value = matching_rows.loc[
                    matching_rows["dates"] == matching_date, column_name
                ].values[0]
                timeseries.loc[
                    (timeseries["transect_id"] == transect_id)
                    & (timeseries["dates"] == shoreline_date),
                    column_name,
                ] = matching_value

    if column_name == "slope":
        unique_ids = np.unique(timeseries["transect_id"])
        median_slope = np.median(np.unique(df[column_name]))
        # Fill NaN values in the 'slope' column with the calculated median
        timeseries[column_name].fillna(median_slope, inplace=True)

    return timeseries


# Only cares about column name, 'transect_id', and 'dates' column
def match_via_points_and_date(
    timeseries: pd.DataFrame, df: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    """
    Matches measurements to transects based on closest spatial and temporal proximity.

    Finds spatially closest points in the reference DataFrame for each transect,
    then temporally matches each shoreline observation to the closest measurement.

    Args:
        timeseries (pd.DataFrame): DataFrame with columns: 'transect_id', 'x', 'y', 'dates'.
        df (pd.DataFrame): Reference DataFrame with columns: 'latitude', 'longitude', 'dates', and specified column.
        column_name (str): Name of the column in df containing the values to match.

    Returns:
        pd.DataFrame: Input timeseries DataFrame with added specified column containing matched values.

    Example:
        >>> timeseries_df = pd.DataFrame({
        ...     'transect_id': ['1', '1'],
        ...     'x': [-120.5, -120.5],
        ...     'y': [35.2, 35.2],
        ...     'dates': pd.to_datetime(['2021-01-15', '2021-06-15'])
        ... })
        >>> reference_df = pd.DataFrame({
        ...     'latitude': [35.19, 35.21],
        ...     'longitude': [-120.51, -120.49],
        ...     'dates': pd.to_datetime(['2021-01-10', '2021-06-20']),
        ...     'tide': [1.2, 1.8]
        ... })
        >>> result = match_via_points_and_date(timeseries_df, reference_df, 'tide')
        >>> print(result)
        transect_id      x     y      dates  tide
        0           1  -120.5  35.2 2021-01-15   1.2
        1           1  -120.5  35.2 2021-06-15   1.8
    """
    # Get unique transect IDs
    df_transect_id = np.unique(timeseries["transect_id"])

    for transect_id in df_transect_id:
        # Get transect coordinates
        transect_x, transect_y = timeseries.loc[
            timeseries["transect_id"] == transect_id, ["x", "y"]
        ].values[0]

        df["distance"] = df.apply(
            lambda row: compute_distance_xy(
                transect_x, transect_y, row["longitude"], row["latitude"]
            ),
            axis=1,
        )

        # Find points with minimum distance
        min_distance = np.min(df["distance"])
        closest_tides = df.loc[df["distance"] == min_distance]

        # For each date in this transect, find the closest tide temporally
        for date in timeseries.loc[timeseries["transect_id"] == transect_id, "dates"]:
            closest_date = closest_tides.iloc[
                (closest_tides["dates"] - date).abs().argsort()[:1]
            ]
            value = closest_date[column_name].values[0]

            # Assign the value to the matching transect and date
            timeseries.loc[
                (timeseries["transect_id"] == transect_id)
                & (timeseries["dates"] == date),
                column_name,
            ] = value

    return timeseries


def melt_df(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Transforms a DataFrame by melting it, converting columns into rows.

    Handles two formats: dates in 'Unnamed: 0' column or dates as row index.
    Converts the 'dates' column to datetime format and melts the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to be transformed.
        column_name (str): Name for the values column in the melted DataFrame.

    Returns:
        pd.DataFrame: Melted DataFrame with 'dates', 'transect_id', and specified column.

    Example:
        >>> df = pd.DataFrame({
        ...     'Unnamed: 0': ['2021-01-01', '2021-01-02'],
        ...     '1': [1.2, 1.3],
        ...     '2': [1.3, 1.4]
        ... })
        >>> result = melt_df(df, 'tide')
        >>> print(result)
            dates transect_id  tide
        0 2021-01-01           1   1.2
        1 2021-01-01           2   1.3
        2 2021-01-02           1   1.3
        3 2021-01-02           2   1.4
    """
    if "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0": "dates"}, inplace=True)
    else:
        df = df.reset_index(names="dates")
    df["dates"] = pd.to_datetime(df["dates"])
    df = pd.melt(df, id_vars=["dates"], var_name="transect_id", value_name=column_name)
    return df


def clean_dataframe(
    df: pd.DataFrame,
    keep_columns: Union[List[str], None] = None,
    convert_to_lower: bool = True,
    remove_s: bool = True,
) -> pd.DataFrame:
    """
    Cleans DataFrame by transforming column names and filtering columns.

    Args:
        df (pd.DataFrame): DataFrame to be cleaned.
        keep_columns (Union[List[str], None], optional): Column names to retain.
                                                   If None, keeps all columns after cleaning. Defaults to None.
        convert_to_lower (bool, optional): Whether to convert column names to lowercase. Defaults to True.
        remove_s (bool, optional): Whether to remove trailing 's' from column names. Defaults to True.

    Returns:
        pd.DataFrame: Cleaned DataFrame with transformed column names and optionally filtered columns.

    Example:
        >>> df = pd.DataFrame({
        ...     'TransectIds': [1, 2, 3],
        ...     'Slopes': [0.1, 0.2, 0.3],
        ...     'ExtraColumn': ['a', 'b', 'c']
        ... })
        >>> result = clean_dataframe(df, keep_columns=['transectid', 'slope'])
        >>> print(result.columns.tolist())
        ['transectid', 'slope']
    """
    if convert_to_lower:
        df.columns = df.columns.str.lower()

    if remove_s:
        # remove 's' from end of column names
        df.columns = df.columns.str.replace(r"s$", "", regex=True)
    if keep_columns is None:
        return df
    cols_to_drop = [col for col in df.columns if col not in keep_columns]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    return df


def read_content_csv(
    file: Union[str, Path], timeseries: pd.DataFrame, column_name: str = "tide"
) -> pd.DataFrame:
    """
    Reads CSV data and merges it with timeseries DataFrame using appropriate matching strategy.

    Supports various CSV formats including seasonal data with monthly information
    and time series data with different matching strategies based on available columns.

    Args:
        file (Union[str, Path]): Path to the CSV file containing data to be added to timeseries.
        timeseries (pd.DataFrame): DataFrame containing shoreline data.
        column_name (str, optional): Name of column to match and add. Defaults to 'tide'.

    Returns:
        pd.DataFrame: DataFrame containing merged data from the CSV file with the timeseries.

    Raises:
        ValueError: If the CSV format is not supported or doesn't match expected column structures.

    Example:
        >>> timeseries_df = pd.DataFrame({
        ...     'dates': pd.to_datetime(['2021-01-15', '2021-06-15']),
        ...     'transect_id': ['1', '2']
        ... })
        >>> # CSV file with monthly data
        >>> result = read_content_csv('monthly_data.csv', timeseries_df, 'slope')
        >>> print(result)
            dates transect_id  slope
        0 2021-01-15           1    0.1
        1 2021-06-15           2    0.2
    """
    df = pd.read_csv(file)

    # clean the dataframe
    df = clean_dataframe(df, keep_columns=None, remove_s=False)

    # If any of the columns contain 'month' then use the logic for seasonal data matching
    if any(df.columns.str.contains(r"(?i)month")):
        df = clean_dataframe(df, keep_columns=["transect_id", "month", column_name])
        if (
            "transect_id" in df.columns
            and "month" in df.columns
            and column_name in df.columns
        ):
            merged_csv = match_via_id_and_month(timeseries, df, column_name)
        elif "month" in df.columns and column_name in df.columns:
            merged_csv = match_via_month(timeseries, df, column_name)
        else:
            raise ValueError(
                f'CSV format not supported. If you are using a CSV file with monthly data then the columns should be "month" and "{column_name}" or  "transect_id", "month" and "{column_name}"'
            )
    else:
        # Convert dataframe to column based format so that transect_id, dates and column_name are columns
        if "dates" not in df.columns or "Unnamed: 0" in df.columns:
            df = melt_df(df, column_name)

        df = clean_dataframe(
            df,
            keep_columns=["transect_id", "dates", column_name, "latitude"],
            remove_s=False,
        )

        # Convert the dates column to ISO 8601 format and ensure it is timezone-aware
        df = convert_col_to_ISO_8601(df, "dates")

        # if it has columns 'transect_id', 'tide', 'dates'
        if "transect_id" in df.columns:
            timeseries[column_name] = np.nan
            merged_csv = match_via_id_and_date(timeseries, df, column_name)
        # if it has columns 'latitude', 'longitude', 'tide', 'dates'
        elif "latitude" in df.columns:
            merged_csv = match_via_points_and_date(timeseries, df, column_name)
        # if it has columns 'dates', 'tide'
        elif "dates" in df.columns and column_name in df.columns:
            merged_csv = match_via_date(timeseries, df, column_name)
        else:
            if column_name == "tide":
                raise ValueError(
                    "CSV format not supported. Must be in one of the following formats as listed on the documentation: https://satelliteshorelines.github.io/CoastSeg/tide-file-format/"
                )
            else:
                raise ValueError(
                    "CSV format not supported. Must be in one of the following formats as listed on the documentation: https://satelliteshorelines.github.io/CoastSeg/slope-file-format/"
                )
    return merged_csv


def compute_tidal_corrections(
    session_name: str,
    roi_ids: Collection[str],
    beach_slope: Union[float, str],
    reference_elevation: float,
    only_keep_points_on_transects: bool = False,
    model: str = "FES2022",
    tides_file: str = "",
    use_progress_bar: bool = True,
) -> None:
    """
    Computes tidal corrections for specified regions of interest (ROIs).

    Args:
        session_name (str): Name of the session.
        roi_ids (Collection[str]): Collection of ROI identifiers.
        beach_slope (Union[float, str]): Beach slope value or path to file containing slopes.
        reference_elevation (float): Reference elevation in meters relative to MSL.
        only_keep_points_on_transects (bool, optional): If True, only keep points on transects. Defaults to False.
        model (str, optional): Tidal model to use ("FES2022", "FES2014", or ""). Defaults to "FES2022".
        tides_file (str, optional): Path to CSV file containing tide data. Defaults to "".
        use_progress_bar (bool, optional): Whether to display progress bar. Defaults to True.

    Returns:
        None

    Raises:
        ValueError: If neither model nor tides_file is provided.
        Exception: If an error occurs during tidal correction computation.
    """
    logger.info(
        f"Computing tides for ROIs {roi_ids} beach_slope: {beach_slope} reference_elevation: {reference_elevation}"
    )

    if model == "" and tides_file == "":
        raise ValueError(
            "Cannot correct tides\nEither set model='FES2014'/model='FES2022' or a provide a file containing tides"
        )

    try:
        correct_all_tides(
            roi_ids,
            session_name,
            reference_elevation,
            beach_slope,
            only_keep_points_on_transects=only_keep_points_on_transects,
            use_progress_bar=use_progress_bar,
            model=model,
            tides_file=tides_file,
        )
    except Exception as e:
        print(f"Tide Model Error \n {e}")
        print(traceback.format_exc())
    else:
        print("\ntidal corrections completed")


def correct_all_tides(
    roi_ids: Collection[str],
    session_name: str,
    reference_elevation: float,
    beach_slope: Union[float, str],
    only_keep_points_on_transects: bool = False,
    use_progress_bar: bool = True,
    model: str = "FES2022",
    tides_file: str = "",
) -> None:
    """
    Corrects tides for all specified regions of interest (ROIs).

    Validates tide model existence, loads model regions, and corrects tides for each ROI.

    Args:
        roi_ids (Collection[str]): IDs of ROIs to correct tides for.
        session_name (str): Name of session containing extracted shorelines.
        reference_elevation (float): Reference elevation for tide correction.
        beach_slope (Union[float, str]): Beach slope for tide correction.
        only_keep_points_on_transects (bool, optional): Whether to keep only points on transects. Defaults to False.
        use_progress_bar (bool, optional): Whether to display progress bar. Defaults to True.
        model (str, optional): Tide model to use. Defaults to "FES2022".
        tides_file (str, optional): Path to tides file. Defaults to "".

    Returns:
        None
    """
    model_location = ""
    tide_regions_file = ""
    # validate tide model exists at CoastSeg/tide_model
    if model != "":
        model_location = get_tide_model_location(model=model.lower())
        # load the regions the tide model was clipped to from geojson file
        tide_regions_file = file_utilities.load_package_resource(
            "tide_model", "tide_regions_map.geojson"
        )

    with progress_bar_context(
        use_progress_bar,
        total=len(roi_ids),
        description=f"Correcting Tides for {len(roi_ids)} ROIs",
    ) as update:
        for roi_id in roi_ids:
            correct_tides(
                roi_id,
                session_name,
                reference_elevation,
                beach_slope,
                only_keep_points_on_transects=only_keep_points_on_transects,
                use_progress_bar=use_progress_bar,
                tides_file=tides_file,
                model=model,
                tide_regions_file=tide_regions_file,
                model_location=model_location,
            )
            logger.info(f"{roi_id} was tidally corrected")
            update(f"{roi_id} was tidally corrected")


def save_transect_settings(
    session_path: str,
    reference_elevation: float,
    beach_slope: float,
    filename: str = "transects_settings.json",
) -> None:
    """
    Updates and saves transect settings with provided reference elevation and beach slope.

    Args:
        session_path (str): Path to session directory containing the transect settings JSON file.
        reference_elevation (float): Reference elevation value to update.
        beach_slope (float): Beach slope value to update.
        filename (str, optional): Name of the JSON settings file. Defaults to "transects_settings.json".

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified settings file does not exist.
    """
    transects_settings = file_utilities.read_json_file(
        os.path.join(session_path, filename), raise_error=False
    )
    transects_settings["reference_elevation"] = reference_elevation
    transects_settings["beach_slope"] = beach_slope
    file_utilities.to_file(transects_settings, os.path.join(session_path, filename))


def get_tides_from_model(
    model_location: str,
    transects_gdf: gpd.GeoDataFrame,
    raw_timeseries_df: pd.DataFrame,
    tide_regions_file: str,
    model: str = "FES2022",
) -> pd.DataFrame:
    """
    Retrieves tide predictions from a specified tide model.

    Args:
        model_location (str): Path to the tide model location.
        transects_gdf (gpd.GeoDataFrame): GeoDataFrame containing transect data.
        raw_timeseries_df (pd.DataFrame): DataFrame containing raw timeseries data with 'dates' column.
        tide_regions_file (str): Path to file containing tide regions information.
        model (str, optional): Name of tide model to use. Defaults to "FES2022".

    Returns:
        pd.DataFrame: DataFrame with columns 'dates', 'transect_id', and 'tide' containing predicted tides.
    """
    # maybe print an error message if the tides model location and tide_regions file are not provided
    tide_model_config = setup_tide_model_config(model_location, model=model)
    predicted_tides_df = predict_tides(
        transects_gdf,
        raw_timeseries_df,
        tide_regions_file,
        tide_model_config,
    )
    return predicted_tides_df


def apply_tide_correction_df(timeseries: pd.DataFrame) -> pd.DataFrame:
    """
    Applies tide correction to timeseries DataFrame by adjusting cross-shore distances.

    Calculates tidal corrections based on tide-reference elevation difference
    normalized by beach slope, then applies to cross-shore distance measurements.

    Args:
        timeseries (pd.DataFrame): DataFrame with required columns:
            - 'tide': Tide level at each time point.
            - 'reference_elevation': Reference elevation to compare against.
            - 'slope': Beach slope.
            - 'cross_distance': Original cross-shore distance to adjust.

    Returns:
        pd.DataFrame: Modified DataFrame with 'cross_distance' adjusted for tide correction.
                     Temporary 'correction' column is removed.

    Raises:
        KeyError: If any required columns are missing.

    Example:
        >>> df = pd.DataFrame({
        ...     'tide': [1.2, 0.8],
        ...     'reference_elevation': [0.0, 0.0],
        ...     'slope': [0.1, 0.1],
        ...     'cross_distance': [50.0, 45.0]
        ... })
        >>> result = apply_tide_correction_df(df)
        >>> print(result)
           tide  reference_elevation  slope  cross_distance
        0   1.2                  0.0    0.1            62.0
        1   0.8                  0.0    0.1            53.0

    Notes:
        Correction formula: correction = (tide - reference_elevation) / slope
        Adjusted distance: cross_distance + correction
    """
    reference_elevation = timeseries["reference_elevation"]
    beach_slope = timeseries["slope"]
    timeseries["correction"] = (timeseries["tide"] - reference_elevation) / beach_slope
    timeseries["cross_distance"] = (
        timeseries["cross_distance"] + timeseries["correction"]
    )
    # drop correction
    timeseries.drop(columns=["correction"], inplace=True)
    return timeseries


def save_to_transect_settings(
    session_name: str,
    roi_id: str,
    reference_elevation: float,
    beach_slope: Union[float, str],
) -> None:
    """
    Saves tide correction parameters to the transect settings file for a specific ROI.

    Updates the transect settings JSON file with reference elevation and beach slope values.
    If beach_slope is a string (file path), it's converted to NaN.

    Args:
        session_name (str): Name of the session containing the ROI.
        roi_id (str): Identifier for the region of interest.
        reference_elevation (float): Reference elevation value to save.
        beach_slope (Union[float, str]): Beach slope value or file path. If string, converted to NaN.

    Returns:
        None

    Example:
        >>> save_to_transect_settings("my_session", "roi_001", 0.5, 0.1)
        # Saves reference_elevation=0.5 and beach_slope=0.1 to the settings file

        >>> save_to_transect_settings("my_session", "roi_001", 0.5, "slopes.csv")
        # Saves reference_elevation=0.5 and beach_slope=NaN (since beach_slope is a file path)
    """
    # optionally save to session location in ROI save the predicted tides to csv
    session_path = file_utilities.get_session_contents_location(session_name, roi_id)

    if isinstance(beach_slope, str):
        beach_slope = np.nan
    # read in transect_settings.json from session_path save the beach slope and reference shoreline
    save_transect_settings(
        session_path, reference_elevation, beach_slope, "transects_settings.json"
    )


def save_predicted_tides_to_csv(session_path, predicted_tides_df):
    """
    Saves the predicted tides DataFrame to a CSV file after pivoting it.

    Args:
        session_path (str): The directory path where the CSV file will be saved.
        predicted_tides_df (pd.DataFrame): DataFrame containing predicted tides with columns 'dates', 'transect_id', and 'tide'.

    Returns:
        pd.DataFrame: The pivoted DataFrame with 'dates' as the index and 'transect_id' as columns.
    """
    pivot_df = predicted_tides_df.pivot_table(
        index="dates", columns="transect_id", values="tide", aggfunc="first"
    )
    # Reset index if you want 'dates' back as a column
    pivot_df.reset_index(inplace=True)
    pivot_df.to_csv(os.path.join(session_path, "predicted_tides.csv"), index=False)
    return pivot_df


def get_matrix_timeseries(timeseries: pd.DataFrame):
    """
    Returns a timeseries DataFrame as a matrix to a CSV file.

    This function takes a timeseries DataFrame, pivots it to create a matrix
    where the rows are dates and the columns are transect IDs, and then saves
    this matrix to a CSV file.

    Parameters:
    timeseries (pd.DataFrame): The input DataFrame containing the timeseries data.
                                It must have columns 'dates', 'transect_id', and 'cross_distance'.

    Returns:
    pd.DataFrame: The pivoted DataFrame (matrix) with 'dates' as the index and
                    'transect_id' as the columns.
                    Note : transect id is converted to string
    """
    matrix_timeseries = timeseries.copy()
    # convert the transect ids to string
    matrix_timeseries["transect_id"] = matrix_timeseries["transect_id"].astype(str)
    matrix_timeseries.set_index("dates", inplace=True)
    matrix_timeseries = matrix_timeseries.pivot_table(
        index="dates",
        columns="transect_id",
        values="cross_distance",
    )
    # Reset index if you want 'dates' back as a column
    matrix_timeseries.reset_index(inplace=True)
    return matrix_timeseries


def correct_tides(
    roi_id: str,
    session_name: str,
    reference_elevation: float,
    beach_slope: Union[float, str],
    only_keep_points_on_transects: bool = False,
    use_progress_bar: bool = True,
    tides_file: str = "",
    model: str = "FES2022",
    tide_regions_file: str = "",
    model_location: str = "",
) -> pd.DataFrame:
    """
    Corrects the timeseries using the tide data provided

    Parameters:
    -----------
    roi_id : str
        Identifier for the Region Of Interest.
    session_name : str
        Name of the session.
    reference_elevation : float
        Reference elevation value.
    beach_slope : float
        Slope of the beach.
    model_location : str
        Path to the tide model.
    tide_regions_file : str
        Path to the file containing the regions the tide model was clipped to.
    only_keep_points_on_transects : bool
    If True, keeps only the shoreline points that are on the transects. Default is True.
        - This will generated a file called "dropped_points_time_series.csv" that contains the points that were filtered out. If keep_points_on_transects is True.
        - Any shoreline points that were not on the transects will be removed from "raw_transect_time_series.csv" by setting those values to NaN.v If keep_points_on_transects is True.
        - The "raw_transect_time_series_merged.csv" will not contain any points that were not on the transects. If keep_points_on_transects is True.

    use_progress_bar : bool, optional
        If True, a tqdm progress bar will be displayed. Default is True.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing tide-corrected time series data for the specified ROI.

    Notes:
    ------
    - This function will set up the tide model, get the time series for the ROI, and predict tides.
    - Tidally corrected time series data will be saved to the session location for the ROI.
    """
    with progress_bar_context(use_progress_bar, total=6) as update:
        update(f"Getting time series for ROI : {roi_id}")
        # load the time series
        try:
            # read the merged csv of the raw timeseries (aka not tidally corrected)
            timeseries = get_timeseries(roi_id, session_name, is_merged=True)
        except FileNotFoundError:
            print(
                f"No time series data found for {roi_id} cannot perform tide correction"
            )
            logger.warning(
                f"No time series data found for {roi_id} cannot perform tide correction"
            )
            update(
                f"No time series data found for {roi_id} cannot perform tide correction"
            )
            return pd.DataFrame()
        # this means that only the date column exists but no transects intersected any of the shorelines for any of these dates
        if timeseries.empty:
            print(
                f"No time series data found for {roi_id} cannot perform tide correction"
            )
            logger.warning(
                f"No time series data found for {roi_id} cannot perform tide correction"
            )
            update(
                f"No time series data found for {roi_id} cannot perform tide correction"
            )
            return pd.DataFrame()

        session_path = file_utilities.get_session_contents_location(
            session_name, roi_id
        )
        save_to_transect_settings(
            session_name, roi_id, reference_elevation, beach_slope
        )
        # read the transects from the config_gdf.geojson file
        update(f"Getting transects for ROI : {roi_id}")
        transects_gdf = get_transects(roi_id, session_name)

        # timeseries load from file
        if tides_file != "":
            if not os.path.exists(tides_file):
                raise FileNotFoundError(f"Tide CSV file not found at {tides_file}")
            update(f"Reading tides from file : {roi_id}")
            timeseries = read_content_csv(tides_file, timeseries, column_name="tide")
        else:
            # predict tides
            update(f"Predicting tides : {roi_id}")
            timeseries_matrix = get_matrix_timeseries(timeseries)
            tides = get_tides_from_model(
                model_location=model_location,
                transects_gdf=transects_gdf,
                raw_timeseries_df=timeseries_matrix,
                tide_regions_file=tide_regions_file,
                model=model,
            )

            # convert tides's transect id and timeseries transect id to str
            tides["transect_id"] = tides["transect_id"].astype(str)
            timeseries["transect_id"] = timeseries["transect_id"].astype(str)

            # convert tides and timeseries columns to str
            # if the tides dataframe has columns x and y drop them
            tides.drop(columns=["x", "y"], inplace=True, errors="ignore")

            timeseries = common.merge_dataframes(tides, timeseries)

        # load the slopes if they are passed in
        if isinstance(beach_slope, str):
            timeseries["slope"] = np.nan
            timeseries = read_content_csv(beach_slope, timeseries, column_name="slope")
        else:
            timeseries["slope"] = beach_slope

        # For any tides missing just skip tide correction and set the result as NaN
        # drop rows with NaN values in 'tide' column
        timeseries.dropna(subset=["tide"], inplace=True)
        timeseries["reference_elevation"] = reference_elevation
        update(f"Tidally correcting time series for ROI : {roi_id}")
        # Apply tide correction to the time series using the tide predictions
        # assumes columns are tide, reference_elevation and slope
        tide_corrected_timeseries_df = apply_tide_correction_df(timeseries)

        # Saves the predicted tides as a csv file called 'predicted_tides.csv' in the session location
        # predicted tides is a matrix of dates and transect ids with the tide values
        tides_matrix = save_predicted_tides_to_csv(
            session_path, tide_corrected_timeseries_df
        )
        tides_matrix.to_csv(
            os.path.join(session_path, "predicted_tides.csv"), index=False
        )

        # make the matrix of dates x transect_ids
        tide_corrected_matrix = get_matrix_timeseries(tide_corrected_timeseries_df)

        # tidally corrected timeseries in format dates, transect_id, x, y, shore_x, shore_y, tide, reference_elevation, slope, correction, cross_distance

        update(f"Saving tidally corrected time series for ROI : {roi_id}")
        tide_corrected_timeseries_merged_df, timeseries_df = (
            common.add_lat_lon_to_timeseries(
                tide_corrected_timeseries_df,
                transects_gdf.to_crs("epsg:4326"),
                tide_corrected_matrix,
                session_path,
                only_keep_points_on_transects,
                "tidally_corrected",
            )
        )

        # Save the Tidally corrected time series
        sorted_columns = [timeseries_df.columns[0]] + sorted(
            timeseries_df.columns[1:],
            key=lambda x: int("".join(filter(str.isdigit, x))),
        )
        timeseries_df = timeseries_df[sorted_columns]
        timeseries_df.to_csv(
            os.path.join(session_path, "tidally_corrected_transect_time_series.csv"),
            index=False,
        )

        # optionally save to session location in ROI the tide_corrected_timeseries_df to csv
        tide_corrected_timeseries_merged_df.to_csv(
            os.path.join(
                session_path, "tidally_corrected_transect_time_series_merged.csv"
            ),
            index=False,
        )

        update(f"{roi_id} was tidally corrected")
    return tide_corrected_timeseries_merged_df


def get_timeseries(
    ROI_ID: str, session_name: str, is_merged: bool = False
) -> pd.DataFrame:
    """
    Retrieves the raw timeseries DataFrame for a given ROI ID and session name.

    Args:
        ROI_ID (str): ID of the region of interest.
        session_name (str): Name of the session.
        is_merged (bool, optional): If True, retrieves merged timeseries file. Defaults to False.

    Returns:
        pd.DataFrame: Raw timeseries DataFrame containing shoreline data.

    Raises:
        FileNotFoundError: If the timeseries file is not found for the specified ROI and session.
    """
    # get the contents of the session directory containing the data for the ROI id
    time_series_location = get_timeseries_path(
        ROI_ID, session_name, TimeSeriesTypes.RAW, is_merged
    )
    timeseries_df = timeseries_read_csv(time_series_location, is_merged)
    return timeseries_df


def get_transects(roi_id: str, session_name: str) -> gpd.GeoDataFrame:
    """
    Retrieves the transects GeoDataFrame for a specific ROI and session.

    Locates and reads the transects configuration file (config_gdf.geojson)
    for the specified region of interest, filtering for transect features only.

    Args:
        roi_id (str): Identifier for the region of interest.
        session_name (str): Name of the session containing the ROI data.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing transect geometries and metadata for the specified ROI.

    Raises:
        FileNotFoundError: If the configuration file is not found.

    Example:
        >>> transects_gdf = get_transects("roi_001", "my_session")
        >>> print(transects_gdf.columns.tolist())
        ['id', 'type', 'geometry']
        >>> print(len(transects_gdf))
        25  # Number of transects for this ROI
    """
    # open the sessions directory
    session_path = file_utilities.get_session_location(session_name)
    roi_location = file_utilities.find_matching_directory_by_id(session_path, roi_id)
    if roi_location is not None:
        session_path = roi_location
    # locate the config_gdf.geojson containing the transects
    config_path = file_utilities.find_file_by_regex(
        session_path, r"^config_gdf\.geojson$"
    )
    # Load the GeoJSON file containing transect data
    transects_gdf = read_and_filter_geojson(config_path, feature_type="transect")
    # get only the transects that intersect with this ROI
    # this may not be necessary because these should have NaN values
    return transects_gdf


def setup_tide_model_config(
    model_path: str, model: str
) -> Dict[str, Union[str, list, bool, float, int]]:
    """
    Set up configuration dictionary for tide model computations.

    This function creates a standardized configuration dictionary containing
    all the parameters needed for tide model predictions using pyTMD.

    Args:
        model_path (str): Path to the directory containing the tide model data files.
        model (str): Name of the tide model in uppercase (e.g., 'FES2022', 'FES2014').

    Returns:
        Dict[str, Union[str, list, bool, float, int]]: Configuration dictionary containing:
            - DIRECTORY: Path to model data
            - MODEL: Model name in uppercase
            - ATLAS_FORMAT: File format ('netcdf')
            - EXTRAPOLATE: Whether to extrapolate beyond model domain
            - METHOD: Interpolation method ('bilinear')
            - TYPE: Model type ('drift')
            - TIME: Time format ('datetime')
            - EPSG: Coordinate system (4326)
            - FILL_VALUE: Value for missing data (NaN)
            - CUTOFF: Extrapolation cutoff distance
            - REGION_DIRECTORY: Path to regional model files

    Example:
        >>> config = setup_tide_model_config('/path/to/model', 'FES2022')
        >>> print(config['MODEL'])
        'FES2022'
        >>> print(config['EXTRAPOLATE'])
        True
    """
    return {
        "DIRECTORY": model_path,
        "DELTA_TIME": [0],
        "GZIP": False,
        "MODEL": model.upper(),  # name of the model in uppercase eg FES2022
        "ATLAS_FORMAT": "netcdf",
        "EXTRAPOLATE": True,
        "METHOD": "spline",
        "TYPE": "drift",
        "TIME": "datetime",
        "EPSG": 4326,
        "FILL_VALUE": np.nan,
        "CUTOFF": 10,
        "METHOD": "bilinear",
        "REGION_DIRECTORY": os.path.join(model_path, "region"),
    }


def get_tide_model_location(location: str = "", model: str = "fes2022") -> str:
    """
    Validates the existence of a tide model at the specified location and returns the absolute path.

    This function checks if a tide model exists at the given location. If the model exists,
    it returns the absolute path of the location. If the model does not exist, it raises an exception.

    Args:
        location (str, optional): The location to check for the tide model.
                                If empty, defaults to "tide_model" directory in the CoastSeg base directory.
        model (str, optional): The tide model to use. Defaults to 'fes2022'.
                              Available options: 'fes2022', 'fes2014'.

    Returns:
        str: The absolute path of the location if the tide model exists.

    Raises:
        Exception: If the tide model does not exist at the specified location.

    Example:
        >>> model_path = get_tide_model_location("/path/to/model", "fes2022")
        >>> print(model_path)
        '/path/to/model'

        >>> # Using default location
        >>> model_path = get_tide_model_location()
        >>> print(model_path)
        '/path/to/coastseg/tide_model'
    """
    # if not location is provided use the default location of the tide model at CoastSeg/tide_model
    if not location:
        base_dir = os.path.abspath(core_utilities.get_base_dir())
        location = os.path.join(base_dir, "tide_model")

    logger.info(f"Checking if tide model exists at {location}")
    if validate_tide_model_exists(location, model=model):
        return os.path.abspath(location)
    else:
        raise Exception(
            f"Tide model not found at: '{os.path.abspath(location)}'. Ensure the model is downloaded to this location."
        )


def validate_tide_model_exists(location: str, model: str = "fes2022") -> bool:
    """
    Validates if a given directory exists and adheres to the expected tide model structure.

    This function checks if the specified directory contains the required sub-directories
    for a tide model, specifically looking for "region0" to "region10" directories
    with the appropriate model files.

    Args:
        location (str): The path to the directory to validate.
        model (str, optional): The tide model name to check. Defaults to 'fes2022'.
                              Available options: 'fes2022', 'fes2014'.

    Returns:
        bool: True if the directory adheres to the expected tide model structure, False otherwise.

    Raises:
        Exception: If the tide model structure is invalid, with specific error messages
                  indicating what components are missing.

    Example:
        >>> is_valid = validate_tide_model_exists("/path/to/tide_model", "fes2022")
        >>> print(is_valid)
        True

        >>> # Invalid structure
        >>> is_valid = validate_tide_model_exists("/invalid/path", "fes2022")
        >>> print(is_valid)
        False
    """

    location = os.path.abspath(location)
    logger.info(f"Tide model absolute path {location}")
    # check if tide directory exists and if the model was clipped to the 10 regions
    if os.path.isdir(location) and contains_sub_directories(location, 10, model):
        return True
    return False


def sub_directory_contains_files(
    sub_directory_path: str, extension: str, count: int
) -> bool:
    """
    Check if a sub-directory contains a specified number of files with a given extension.

    This function validates that a directory exists and contains exactly the expected
    number of files with the specified file extension.

    Args:
        sub_directory_path (str): The path to the sub-directory to check.
        extension (str): The file extension to look for (e.g., '.nc').
        count (int): The expected number of files with the specified extension.

    Returns:
        bool: True if the sub-directory contains the exact number of specified files, False otherwise.

    Raises:
        Exception: If the sub-directory does not exist.

    Example:
        >>> has_files = sub_directory_contains_files("/path/to/dir", ".nc", 34)
        >>> print(has_files)
        True  # If the directory contains exactly 34 .nc files
    """

    if not os.path.isdir(sub_directory_path):
        raise Exception(
            f" Missing directory {os.path.basename(sub_directory_path)} at {sub_directory_path}"
        )
        return False

    files_with_extension = [
        f for f in os.listdir(sub_directory_path) if f.endswith(extension)
    ]
    return len(files_with_extension) == count


def contains_sub_directories(
    directory_name: str, num_regions: int, model="fes2014"
) -> bool:
    """
    Check if a directory contains sub-directories in the format "regionX/fes2014/load_tide"
    and "regionX/fes2014/ocean_tide", and if each of these sub-directories contains 34 .nc files.

    Args:
    - directory_name (str): The name of the main directory.
    - num_regions (int): The number of regions to check (e.g., for 10 regions, it'll check region0 to region10).

    Returns:
    - bool: True if all conditions are met, False otherwise.
    """
    if "fes2022" in model.lower():
        model = "fes2022b"

    for i in range(num_regions + 1):
        region_dir = os.path.join(directory_name, f"region{i}")
        load_tide_path = os.path.join(region_dir, model, "load_tide")
        ocean_tide_path = os.path.join(region_dir, model, "ocean_tide")

        if not os.path.isdir(region_dir):
            raise Exception(
                f"Tide Model was not clipped correctly. Missing the directory '{os.path.basename(region_dir)}' for region {i} at {region_dir}"
            )

        if not os.path.isdir(load_tide_path):
            raise Exception(
                f"Tide Model was not clipped correctly. Region {i} was missing directory '{os.path.basename(load_tide_path)}' at {load_tide_path}"
            )

        if not os.path.isdir(ocean_tide_path):
            raise Exception(
                f"Tide Model was not clipped correctly. Region {i} was missing directory '{os.path.basename(ocean_tide_path)}' at {ocean_tide_path}"
            )

        if not sub_directory_contains_files(load_tide_path, ".nc", 34):
            raise Exception(
                f"Tide Model was not clipped correctly. Region {i} '{os.path.basename(load_tide_path)}' directory did not contain all 34 .nc files at {load_tide_path}.Please download again"
            )
        if not sub_directory_contains_files(ocean_tide_path, ".nc", 34):
            raise Exception(
                f"Tide Model was not clipped correctly. Region {i} '{os.path.basename(ocean_tide_path)}' directory did not contain all 34 .nc files at {ocean_tide_path}. Please download again"
            )

    return True


def get_tide_predictions(
    x: float,
    y: float,
    timeseries_df: pd.DataFrame,
    model_region_directory: str,
    transect_id: str = "",
    model: str = "FES2022",
) -> pd.DataFrame:
    """
    Get tide predictions for a given location and transect ID.

    Args:
        x (float): The x-coordinate of the location to predict tide for.
        y (float): The y-coordinate of the location to predict tide for.
        - timeseries_df: A DataFrame containing time series data for each transect.
            - It expects a DataFrame with columns 'dates'
                 - If a transect ID is available, it should be included as a column in the DataFrame.
                 - It expects all the dates in a single column called dates in datetime format.
                 - All the transect ids should be type string

       - model_region_directory: The path to the model region that will be used to compute the tide predictions
         ex."CoastSeg/tide_model/region"
        transect_id (str): The ID of the transect. Pass "" if no transect ID is available.
        model (str): The tide model to use. Defaults to 'FES2022'.
            Available options FES2014 and FES2022.
    Returns:
            - pd.DataFrame: A DataFrame containing tide predictions for all the dates that the selected transect_id using the
    fes 2014 model region specified in the "region_id".
    """
    # Check if the transect id is available in the timeseries_df. Either as a column or in the 'transect_id' column
    if transect_id != "":
        # get unique dates for the transect_id and convert to a numpy array
        if "transect_id" in timeseries_df.columns:
            matching_rows = timeseries_df[timeseries_df["transect_id"] == transect_id]
            # get the unique dates for the transect_id
            dates_for_transect_id_df = matching_rows[["dates"]].dropna()
        elif transect_id not in timeseries_df.columns:
            return pd.DataFrame()
        else:
            dates_for_transect_id_df = timeseries_df[["dates", transect_id]].dropna()
    else:
        dates_for_transect_id_df = timeseries_df[["dates"]].dropna()

    tide_predictions_df = model_tides(
        x,
        y,
        dates_for_transect_id_df.dates.values,
        transect_id=transect_id,
        directory=model_region_directory,
        model=model.upper(),
    )
    return tide_predictions_df


def predict_tides_for_df(
    seaward_points_gdf: gpd.GeoDataFrame,
    timeseries_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Predict tides for a points in the DataFrame.

    Parameters:
    - seaward_points_gdf: A GeoDataFrame containing seaward points for each transect
    - timeseries_df: A DataFrame containing time series data for each transect.
    - config: Configuration dictionary.
        Must contain keys:
        "REGION_DIRECTORY" : contains full path to the FES model region folder
        "MODEL" : The tide model to use. Defaults to 'FES2022'

    Returns:
    - pd.DataFrame: A DataFrame containing predicted tides.
    Contains columns dates, x, y, tide, transect_id
    """
    region_directory = config["REGION_DIRECTORY"]
    # Use tqdm to add a progress bar for the apply function
    tqdm.pandas(desc=f"  Predicting tides for {len(seaward_points_gdf)} transects")
    all_tides = seaward_points_gdf.progress_apply(
        lambda row: get_tide_predictions(
            row.geometry.x,
            row.geometry.y,
            timeseries_df,
            f"{region_directory}{row['region_id']}",
            row["transect_id"],
            model=config["MODEL"],
        ),
        axis=1,
    )
    # Filter out None values
    all_tides = all_tides.dropna()
    # If no tides are predicted return an empty dataframe
    if all_tides.empty:
        return pd.DataFrame(columns=["dates", "x", "y", "tide", "transect_id"])

    # Concatenate all the results
    all_tides_df = pd.concat(all_tides.tolist())

    return all_tides_df


def model_tides(
    x,
    y,
    time,
    transect_id: str = "",
    model="FES2022",
    directory=None,
    epsg=4326,
    method="bilinear",
    extrapolate=True,
    cutoff=10.0,
):
    """
    Compute tides at points and times using tidal harmonics.
    If multiple x, y points are provided, tides will be
    computed for all timesteps at each point.

    This function supports any tidal model supported by
    `pyTMD`, including the FES2014 Finite Element Solution
    tide model, and FES2022 Finite Element Solution
    tide model.

    This function is a modification of the `pyTMD`
    package's `compute_tide_corrections` function, adapted
    to process multiple timesteps for multiple input point
    locations. For more info:
    https://pytmd.readthedocs.io/en/stable/user_guide/compute_tide_corrections.html

    Parameters:
    -----------
    x, y : float or list of floats
        One or more x and y coordinates used to define
        the location at which to model tides. By default these
        coordinates should be lat/lon; use `epsg` if they
        are in a custom coordinate reference system.
    time : A datetime array or pandas.DatetimeIndex
        An array containing 'datetime64[ns]' values or a
        'pandas.DatetimeIndex' providing the times at which to
        model tides in UTC time.
    model : string
        The tide model used to model tides. Options include:
        - "fes2022b" (only pre-configured option on DEA Sandbox)
        - "TPXO8-atlas"
        - "TPXO9-atlas-v5"
    directory : string
        The directory containing tide model data files. These
        data files should be stored in sub-folders for each
        model that match the structure provided by `pyTMD`:
        https://pytmd.readthedocs.io/en/latest/getting_started/Getting-Started.html#directories
        For example:
        - {directory}/fes2014/ocean_tide/
          {directory}/fes2014/load_tide/
    epsg : int
        Input coordinate system for 'x' and 'y' coordinates.
        Defaults to 4326 (WGS84).
    method : string
        Method used to interpolate tidal contsituents
        from model files. Options include:
        - bilinear: quick bilinear interpolation
        - spline: scipy bivariate spline interpolation
        - linear, nearest: scipy regular grid interpolations
    extrapolate : bool
        Whether to extrapolate tides for locations outside of
        the tide modelling domain using nearest-neighbor
    cutoff : int or float
        Extrapolation cutoff in kilometers. Set to `np.inf`
        to extrapolate for all points.

    Returns
    -------
    A pandas.DataFrame containing tide heights for all the xy points and their corresponding time
    Contains the columns dates, x, y, tide, transect_id

    """
    # Check tide directory is accessible
    if directory is not None:
        directory = pathlib.Path(directory).expanduser()
        if not directory.exists():
            raise FileNotFoundError("Invalid tide directory")
    # Validate input arguments
    assert method in ("bilinear", "spline", "linear", "nearest")

    if "fes2022" in model.lower():
        model = "FES2022"
    # Get parameters for tide model; use custom definition file for
    model = pyTMD.io.model(directory, format="netcdf", compressed=False).elevation(
        model
    )

    # If time passed as a single Timestamp, convert to datetime64
    if isinstance(time, pd.Timestamp):
        time = time.to_datetime64()

    # Handle numeric or array inputs
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    time = np.atleast_1d(time)

    # Determine point and time counts
    assert len(x) == len(y), "x and y must be the same length"
    n_points = len(x)
    n_times = len(time)

    # Converting x,y from EPSG to latitude/longitude
    try:
        # EPSG projection code string or int
        crs1 = pyproj.CRS.from_epsg(int(epsg))
    except (ValueError, pyproj.exceptions.CRSError):
        # Projection SRS string
        crs1 = pyproj.CRS.from_string(epsg)

    # Output coordinate reference system
    crs2 = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    lon, lat = transformer.transform(x.flatten(), y.flatten())

    # Convert datetime
    timescale = pyTMD.time.timescale().from_datetime(time.flatten())
    n_points = len(x)
    # number of time points
    n_times = len(time)

    amp, ph = pyTMD.io.FES.extract_constants(
        lon,
        lat,
        model.model_file,
        type=model.type,
        version=model.version,
        method=method,
        extrapolate=extrapolate,
        cutoff=cutoff,
        scale=model.scale,
        compressed=model.compressed,
    )
    # Available model constituents
    c = model.constituents
    # Delta time (TT - UT1)
    # calculating the difference between Terrestrial Time (TT) and UT1 (Universal Time 1),
    deltat = timescale.tt_ut1

    # Calculate complex phase in radians for Euler's
    cph = -1j * ph * np.pi / 180.0

    # Calculate constituent oscillation
    hc = amp * np.exp(cph)

    # Repeat constituents to length of time and number of input
    # coords before passing to `predict_tide_drift`

    # deltat likely represents the time interval between successive data points or time instances.
    # t =  replicating the timescale.tide array n_points times
    # hc = creates an array with the tidal constituents repeated for each time instance
    # Repeat constituents to length of time and number of input
    # coords before passing to `predict_tide_drift`
    t, hc, deltat = (
        np.tile(timescale.tide, n_points),
        hc.repeat(n_times, axis=0),
        np.tile(deltat, n_points),
    )

    # Predict tidal elevations at time and infer minor corrections
    npts = len(t)
    tide = np.ma.zeros((npts), fill_value=np.nan)
    tide.mask = np.any(hc.mask, axis=1)

    # Predict tides
    tide.data[:] = pyTMD.predict.drift(
        t, hc, c, deltat=deltat, corrections=model.format
    )
    minor = pyTMD.predict.infer_minor(t, hc, c, deltat=deltat, corrections=model.format)
    tide.data[:] += minor.data[:]

    # Replace invalid values with fill value
    tide.data[tide.mask] = tide.fill_value
    if transect_id:
        df = pd.DataFrame(
            {
                "dates": np.tile(time, n_points),
                "x": np.repeat(x, n_times),
                "y": np.repeat(y, n_times),
                "tide": tide,
                "transect_id": transect_id,
            }
        )
        df["dates"] = pd.to_datetime(df["dates"], utc=True)
        df.set_index("dates")
        return df
    else:
        df = pd.DataFrame(
            {
                "dates": np.tile(time, n_points),
                "x": np.repeat(x, n_times),
                "y": np.repeat(y, n_times),
                "tide": tide,
            }
        )
        df["dates"] = pd.to_datetime(df["dates"], utc=True)
        df.set_index("dates")
        return df


def read_and_filter_geojson(
    file_path: str,
    columns_to_keep: Tuple[str, ...] = ("id", "type", "geometry"),
    feature_type: str = "transect",
) -> gpd.GeoDataFrame:
    """
    Read and filter a GeoJSON file based on specified columns and feature type.

    Parameters:
    - file_path: Path to the GeoJSON file.
    - columns_to_keep: A tuple containing column names to be retained in the resulting GeoDataFrame. Default is ("id", "type", "geometry").
    - feature_type: Type of feature to be retained in the resulting GeoDataFrame. Default is "transect".

    Returns:
    - gpd.GeoDataFrame: A filtered GeoDataFrame.
    """
    # Read the GeoJSON file into a GeoDataFrame
    gdf = gpd.read_file(file_path)
    # Drop all other columns in place
    gdf.drop(
        columns=[col for col in gdf.columns if col not in columns_to_keep], inplace=True
    )
    # Filter the features with "type" equal to the specified feature_type
    filtered_gdf = gdf[gdf["type"] == feature_type]

    return filtered_gdf


def load_regions_from_geojson(geojson_path: str) -> gpd.GeoDataFrame:
    """
    Load regions from a GeoJSON file and assign a region_id based on index.

    Parameters:
    - geojson_path: Path to the GeoJSON file containing regions.

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame containing the loaded regions with an added 'region_id' column.
    """
    gdf = gpd.read_file(geojson_path)
    gdf["region_id"] = gdf.index
    return gdf


def perform_spatial_join(
    seaward_points_gdf: gpd.GeoDataFrame, regions_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Perform a spatial join between seaward points and regions based on intersection.
    BOTH GeoDataFrames must be in crs 4326. Otherwise the spatial join will not work.

    Parameters:
    - seaward_points_gdf: A GeoDataFrame containing seaward points.
    - regions_gdf: A GeoDataFrame containing regions.

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame resulting from the spatial join.
    """
    joined_gdf = gpd.sjoin(
        seaward_points_gdf, regions_gdf, how="left", predicate="intersects"
    )
    joined_gdf.drop(columns="index_right", inplace=True)
    return joined_gdf


def model_tides_for_all(
    seaward_points_gdf: gpd.GeoDataFrame,
    timeseries_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Model tides for all points in the provided GeoDataFrame.

    Parameters:
    - seaward_points_gdf: A GeoDataFrame containing seaward point for each transect.
    - timeseries_df: A DataFrame containing time series data for each transect. A DataFrame containing time series data for tides.
    - config: Configuration dictionary.

    Returns:
    - pd.DataFrame: A DataFrame containing predicted tides for all points.
    """
    return predict_tides_for_df(seaward_points_gdf, timeseries_df, config)


def model_tides_by_region_id(
    seaward_points_gdf: gpd.GeoDataFrame,
    timeseries_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Model tides for each unique region ID in the provided GeoDataFrame.

    Parameters:
    - seaward_points_gdf: A GeoDataFrame containing seaward point for each transect.
    - timeseries_df: A DataFrame containing time series data for each transect. A DataFrame containing time series data for tides.
    - config: Configuration dictionary.

    Returns:
    - pd.DataFrame: A DataFrame containing predicted tides segmented by region ID.
    """
    unique_ids = seaward_points_gdf["region_id"].unique()
    all_tides_dfs = []

    for uid in unique_ids:
        subset_gdf = seaward_points_gdf[seaward_points_gdf["region_id"] == uid]
        tides_for_region_df = predict_tides_for_df(subset_gdf, timeseries_df, config)
        all_tides_dfs.append(tides_for_region_df)

    return pd.concat(all_tides_dfs)


def handle_tide_predictions(
    seaward_points_gdf: gpd.GeoDataFrame,
    timeseries_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Handle tide predictions based on the number of unique region IDs in the provided GeoDataFrame.

    Parameters:
    - seaward_points_gdf: A GeoDataFrame containing seaward point for each transect.
    - timeseries_df: A DataFrame containing time series data for each transect. A DataFrame containing time series data for each transect.
    - config: Configuration dictionary.

    Returns:
    - pd.DataFrame: A DataFrame containing predicted tides.
    """
    if seaward_points_gdf["region_id"].nunique() == 1:
        all_tides_df = model_tides_for_all(seaward_points_gdf, timeseries_df, config)
    else:
        all_tides_df = model_tides_by_region_id(
            seaward_points_gdf, timeseries_df, config
        )
    return all_tides_df


def predict_tides(
    transects_gdf: gpd.GeoDataFrame,
    timeseries_df: pd.DataFrame,
    model_regions_geojson_path: str,
    config: dict,
) -> pd.DataFrame:
    """
    Predict tides based on input data and configurations.

    Parameters:
    - geojson_file_path: Path to the GeoJSON file containing transect data.
    - timeseries_df: A DataFrame containing time series data for each transect. A DataFrame containing raw time series data.
    - model_regions_geojson_path: Path to the GeoJSON file containing model regions.
    - config: Configuration dictionary.

    Returns:
    - pd.DataFrame: A DataFrame containing predicted tides.
    """
    # Read in the model regions from a GeoJSON file
    regions_gdf = load_regions_from_geojson(model_regions_geojson_path)
    # convert to crs 4326 if it is not already
    if regions_gdf.crs is None:
        regions_gdf = regions_gdf.set_crs("epsg:4326")
    else:
        regions_gdf = regions_gdf.to_crs("epsg:4326")
    # Get the seaward points in CRS 4326
    seaward_points_gdf = common.get_seaward_points_gdf(transects_gdf)
    # Perform a spatial join to get the region_id for each point in seaward_points_gdf
    regional_seaward_points_gdf = perform_spatial_join(seaward_points_gdf, regions_gdf)
    # predict the tides
    all_tides_df = handle_tide_predictions(
        regional_seaward_points_gdf, timeseries_df, config
    )
    return all_tides_df


def apply_tide_correction(
    df: pd.DataFrame, reference_elevation: float, beach_slope: float
):
    """
    Applies tidal correction to the timeseries data.

    Args:
    - df (DataFrame): Input data with tide predictions and timeseries data.
    - reference_elevation (float): Reference elevation value.
    - beach_slope (float): Beach slope value.

    Returns:
    - DataFrame: Tidally corrected data.
    """
    correction = (df["tide"] - reference_elevation) / beach_slope
    df["cross_distance"] = df["cross_distance"] + correction
    return df.drop(columns=["correction"], errors="ignore")


def timeseries_read_csv(file_path: str, is_merged: bool = False) -> pd.DataFrame:
    """
    Reads the timeseries from a CSV file.
    It converts the dates column to datetime in UTC.
    It drops the columns 'x', 'y', and 'Unnamed: 0' if they exist.

    Args:
        file_path (str): Path to the CSV file.
        is_merged (bool): Indicates if the timeseries is merged. Defaults to False.

    Returns:
    - DataFrame: Processed data.
    """
    df = pd.read_csv(file_path, parse_dates=["dates"])
    df["dates"] = pd.to_datetime(df["dates"], utc=True)
    # only for the non merged timeseries do we drop the x, y and Unnamed: 0 columns
    if is_merged is False:
        for column in ["x", "y", "Unnamed: 0"]:
            if column in df.columns:
                df.drop(columns=column, inplace=True)
    return df
