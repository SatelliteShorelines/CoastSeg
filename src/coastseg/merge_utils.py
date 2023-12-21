# Standard library imports
from collections import defaultdict
import os
from typing import List, Optional, Union

# Related third party imports
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point
from shapely.ops import unary_union

# Local application/library specific imports
from coastseg import geodata_processing


def convert_multipoints_to_linestrings(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert MultiPoint geometries in a GeoDataFrame to LineString geometries.

    Args:
    - gdf (gpd.GeoDataFrame): The input GeoDataFrame.

    Returns:
    - gpd.GeoDataFrame: A new GeoDataFrame with LineString geometries. If the input GeoDataFrame
                        already contains LineStrings, the original GeoDataFrame is returned.
    """

    # Create a copy of the GeoDataFrame
    gdf_copy = gdf.copy()

    # Check if all geometries in the gdf are LineStrings
    if all(gdf_copy.geometry.type == "LineString"):
        return gdf_copy

    def multipoint_to_linestring(multipoint):
        if isinstance(multipoint, MultiPoint):
            return LineString(multipoint.geoms)
        return multipoint

    # Convert each MultiPoint to a LineString
    gdf_copy["geometry"] = gdf_copy["geometry"].apply(multipoint_to_linestring)

    return gdf_copy


def dataframe_to_dict(df: pd.DataFrame, key_map: dict) -> dict:
    """
    Converts a DataFrame to a dictionary, with specific mapping between dictionary keys and DataFrame columns.

    Parameters:
    df : DataFrame
        The DataFrame to convert.
    key_map : dict
        A dictionary where keys are the desired dictionary keys and values are the corresponding DataFrame column names.

    Returns:
    dict
        The resulting dictionary.
    """
    result_dict = defaultdict(list)

    for dict_key, df_key in key_map.items():
        if df_key in df.columns:
            if df_key == "date":
                # Assumes the column to be converted to date is the one specified in the mapping with key 'date'
                result_dict[dict_key] = list(
                    df[df_key].apply(
                        lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
                        if pd.notnull(x)
                        else None
                    )
                )
            elif df_key == "geometry":
                # Assumes the column to be converted to geometry is the one specified in the mapping with key 'geometry'
                result_dict[dict_key] = list(
                    df[df_key].apply(
                        lambda x: np.array([list(point.coords[0]) for point in x.geoms])
                        if pd.notnull(x)
                        else None
                    )
                )
            else:
                result_dict[dict_key] = list(df[df_key])

    return dict(result_dict)


def convert_lines_to_multipoints(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert LineString or MultiLineString geometries in a GeoDataFrame to MultiPoint geometries.

    Parameters
    ----------
    gdf : GeoDataFrame
        The input GeoDataFrame containing LineString or MultiLineString geometries.

    Returns
    -------
    GeoDataFrame
        A new GeoDataFrame with MultiPoint geometries.

    """
    # Create a copy of the input GeoDataFrame to avoid modifying it in place
    gdf = gdf.copy()

    # Define a function to convert LineString or MultiLineString to MultiPoint
    def line_to_multipoint(geometry):
        if isinstance(geometry, LineString):
            return MultiPoint(geometry.coords)
        elif isinstance(geometry, MultiLineString):
            points = [MultiPoint(line.coords) for line in geometry.geoms]
            return MultiPoint([point for multi in points for point in multi.geoms])
        elif isinstance(geometry, MultiPoint):
            return geometry
        elif isinstance(geometry, Point):
            return MultiPoint([geometry.coords])
        else:
            raise TypeError(f"Unsupported geometry type: {type(geometry)}")

    # Apply the conversion function to each row in the GeoDataFrame
    gdf["geometry"] = gdf["geometry"].apply(line_to_multipoint)

    return gdf


def read_first_geojson_file(
    directory: str,
    filenames=["extracted_shorelines_lines.geojson", "extracted_shorelines.geojson"],
):
    """
    Reads the first available GeoJSON file from the given directory.

    Args:
        directory (str): The directory path where the files are located.
        filenames (list, optional): List of filenames to search for. Defaults to ["extracted_shorelines_lines.geojson", "extracted_shorelines.geojson"].

    Returns:
        geopandas.GeoDataFrame: The GeoDataFrame read from the first available file.

    Raises:
        FileNotFoundError: If none of the specified files exist in the directory.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    # Loop over the filenames
    for filename in filenames:
        filepath = os.path.join(directory, filename)

        # If the file exists, read it and return the GeoDataFrame
        if os.path.exists(filepath):
            return geodata_processing.read_gpd_file(filepath)

    # If none of the files exist, raise an exception
    raise FileNotFoundError(
        f"None of the files {filenames} exist in the directory {directory}"
    )


def clip_gdfs(gdfs, overlap_gdf):
    """
    Clips GeoDataFrames to an overlapping region.

    Parameters:
    gdfs : list of GeoDataFrames
        The GeoDataFrames to be clipped.
    overlap_gdf : GeoDataFrame
        The overlapping region to which the GeoDataFrames will be clipped.

    Returns:
    list of GeoDataFrames
        The clipped GeoDataFrames.
    """
    clipped_gdfs = []
    for gdf in gdfs:
        clipped_gdf = gpd.clip(gdf, overlap_gdf)
        if not clipped_gdf.empty:
            clipped_gdfs.append(clipped_gdf)
    return clipped_gdfs


def calculate_overlap(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculates the intersection of all pairs of polygons in a GeoDataFrame.

    Parameters:
    -----------
    gdf : GeoDataFrame
        A GeoDataFrame containing polygons.

    Returns:
    --------
    overlap_gdf : GeoDataFrame
        A GeoDataFrame containing the intersection of all pairs of polygons in gdf.
    """
    # Check if the input GeoDataFrame is empty
    if not hasattr(gdf, "empty"):
        return gpd.GeoDataFrame(geometry=[])
    if gdf.empty:
        # Return an empty GeoDataFrame with the same CRS if it exists
        return gpd.GeoDataFrame(
            geometry=[], crs=gdf.crs if hasattr(gdf, "crs") else None
        )

    # Initialize a list to store the intersections
    intersections = []

    # Loop over each pair of rows in gdf
    for i in range(len(gdf) - 1):
        for j in range(i + 1, len(gdf)):
            # Check for intersection
            if gdf.iloc[i].geometry.intersects(gdf.iloc[j].geometry):
                # Calculate the intersection
                intersection = gdf.iloc[i].geometry.intersection(gdf.iloc[j].geometry)
                # Append the intersection to the intersections list
                intersections.append(intersection)

    # Create a GeoSeries from the intersections
    intersection_series = gpd.GeoSeries(intersections, crs=gdf.crs)

    # Create a GeoDataFrame from the GeoSeries
    overlap_gdf = gpd.GeoDataFrame(geometry=intersection_series)
    return overlap_gdf


def get_overlapping_features(
    roi_gdf: gpd.GeoDataFrame, gdfs: list[gpd.GeoDataFrame]
) -> list[gpd.GeoDataFrame]:
    """
    Get the overlapping features between the region of interest (ROI) and the provided GeoDataFrames (gdfs).

    Parameters:
    - roi_gdf (GeoDataFrame): The region of interest GeoDataFrame.
    - gdfs (GeoDataFrame or list): The GeoDataFrame(s) to check for overlap with the ROI.

    Returns:
    - list: The overlapping features as a list of GeoDataFrames. Where each element in the list is a GeoDataFrame corresponds
            to the overlapping features between the ROI and one of the provided GeoDataFrames (gdfs).

    """
    # calculate the overlapping regions between the ROIs
    overlap_gdf = calculate_overlap(roi_gdf)
    if overlap_gdf.empty:
        return []
    if isinstance(gdfs, gpd.GeoDataFrame):
        gdfs = [gdfs]
    # clip the gdfs  to the overlapping regions ex. clip the extracted shorelines to the overlapping regions
    return clip_gdfs(gdfs, overlap_gdf)


def average_multipoints(multipoints) -> MultiPoint:
    """
    Calculate the average MultiPoint geometry from a list of MultiPoint geometries.

    This function takes a list of shapely MultiPoint geometries, ensures they all have the same number of points
    by padding shorter MultiPoints with their last point, and then calculates the average coordinates
    for each point position across all the input MultiPoint geometries.

    The result is a new MultiPoint geometry that represents the average shape of the input MultiPoints.

    Parameters:
    multipoints (list of shapely.geometry.MultiPoint): A list of shapely MultiPoint geometries to be averaged.

    Returns:
    shapely.geometry.MultiPoint: A MultiPoint geometry representing the average shape of the input MultiPoints.

    Raises:
    ValueError: If the input list of MultiPoint geometries is empty.

    Example:
    >>> from shapely.geometry import MultiPoint
    >>> multipoint1 = MultiPoint([(0, 0), (1, 1), (2, 2)])
    >>> multipoint2 = MultiPoint([(1, 1), (2, 2)])
    >>> multipoint3 = MultiPoint([(0, 0), (1, 1), (2, 2), (3, 3)])
    >>> average_mp = average_multipoints([multipoint1, multipoint2, multipoint3])
    >>> print(average_mp)
    MULTIPOINT (0.3333333333333333 0.3333333333333333, 1.3333333333333333 1.3333333333333333, 2 2, 3 3)
    """
    if not multipoints:
        raise ValueError("The list of MultiPoint geometries is empty")

    # Find the maximum number of points in any MultiPoint
    max_len = max(len(mp.geoms) for mp in multipoints)

    # Pad shorter MultiPoints with their last point
    padded_multipoints = []
    for mp in multipoints:
        if len(mp.geoms) < max_len:
            padded_multipoints.append(
                MultiPoint(list(mp.geoms) + [mp.geoms[-1]] * (max_len - len(mp.geoms)))
            )
        else:
            padded_multipoints.append(mp)

    # Calculate the average coordinates for each point
    num_multipoints = len(padded_multipoints)
    average_coords = []
    for i in range(max_len):
        avg_left = sum(mp.geoms[i].x for mp in padded_multipoints) / num_multipoints
        avg_right = sum(mp.geoms[i].y for mp in padded_multipoints) / num_multipoints
        average_coords.append((avg_left, avg_right))

    return MultiPoint(average_coords)


def merge_geometries(merged_gdf, columns=None, operation=unary_union):
    """
    Performs a specified operation for the geometries with the same date and satname.

    Parameters:
    merged_gdf : GeoDataFrame
        The GeoDataFrame to perform the operation on.
    columns : list of str, optional
        The columns to perform the operation on. If None, all columns with 'geometry' in the name are used.
    operation : function, optional
        The operation to perform. If None, unary_union is used.

    Returns:
    GeoDataFrame
        The GeoDataFrame with the operation performed.
    """
    if columns is None:
        columns = [col for col in merged_gdf.columns if "geometry" in col]
    else:
        columns = [col for col in columns if col in merged_gdf.columns]

    # set the geometry of th merged_gdf to the result of the operation
    # if no operation is provided unary_union is used to combine the geometries for the provided columns
    merged_gdf.set_geometry(
        merged_gdf[columns].apply(lambda row: operation(row.tolist()), axis=1),
        inplace=True,
    )

    for col in columns:
        if col in merged_gdf.columns and col != "geometry":
            merged_gdf = merged_gdf.drop(columns=col)
    return merged_gdf


def read_geojson_files(
    filepaths, column="type", value=None, keep_columns=None, crs=None
):
    """Read GeoJSON files into GeoDataFrames and return a list."""
    gdfs = []
    for path in filepaths:
        gdf = gpd.read_file(path)
        if crs:
            gdf = gdf.to_crs(crs)
        print(f"Read {len(gdf)} features from {path}")
        # print(gdf[gdf[column] == value])
        if column in gdf.columns and value is not None:
            gdf = gdf[gdf[column] == value]
        if keep_columns is not None:
            gdf = gdf[keep_columns]
        gdfs.append(gdf)
    return gdfs


def concatenate_gdfs(gdfs):
    """Concatenate a list of GeoDataFrames into a single GeoDataFrame."""
    return pd.concat(gdfs, ignore_index=True)


def filter_and_join_gdfs(gdf, feature_type, predicate="intersects"):
    """Filter GeoDataFrame by feature type, ensure spatial index, and perform a spatial join."""
    if "type" not in gdf.columns:
        raise ValueError("The GeoDataFrame must contain a column named 'type'")
    # Filter GeoDataFrame by feature type
    filtered_gdf = gdf[gdf["type"] == feature_type].copy()[["geometry"]]
    filtered_gdf.sindex  # Ensure spatial index
    # perform a spatial join
    return gpd.sjoin(gdf, filtered_gdf[["geometry"]], how="inner", predicate=predicate)


def aggregate_gdf(gdf: gpd.GeoDataFrame, group_fields: list) -> gpd.GeoDataFrame:
    """
    Aggregate a GeoDataFrame by specified fields using a custom combination function.

    Parameters:
        gdf (GeoDataFrame): The input GeoDataFrame to be aggregated.
        group_fields (list): The fields to group the GeoDataFrame by.

    Returns:
        GeoDataFrame: The aggregated GeoDataFrame.
    """

    def combine_non_nulls(series):
        unique_values = series.dropna().unique()
        return (
            unique_values[0]
            if len(unique_values) == 1
            else ", ".join(map(str, unique_values))
        )

    if "index_right" in gdf.columns:
        gdf = gdf.drop(columns=["index_right"])

    return (
        gdf.drop_duplicates()
        .groupby(group_fields, as_index=False)
        .agg(combine_non_nulls)
    )


def process_geojson_files(
    session_locations,
    filenames,
    transform_funcs=None,
    read_func=None,
    crs=None,
):
    """
    Reads and optionally transforms GeoDataFrames from given session locations.

    Args:
        session_locations (list): List of paths to session directories.
        filenames (list): List of filenames to read in each session directory.
        transform_funcs (list, optional): List of functions to apply to each file.
        read_func (callable, optional): Function to use for reading files.
        crs (str, optional): Coordinate reference system to convert GeoDataFrames to. Defaults to 'epsg:4326'.

    Returns:
        list: List of processed GeoDataFrames.
    """
    if transform_funcs is None:
        transform_funcs = []
    if transform_funcs and not isinstance(transform_funcs, list):
        transform_funcs = [transform_funcs]
    if read_func is None:
        raise ValueError("read_func must be specified")

    if not isinstance(session_locations, list):
        session_locations = [session_locations]

    gdfs = []
    for session_dir in session_locations:
        try:
            gdf = read_func(session_dir, filenames)
            for func in transform_funcs:
                gdf = func(gdf)
            if isinstance(gdf, gpd.GeoDataFrame):
                if "geometry" in gdf.columns and not gdf.crs:
                    gdf.set_crs(crs, inplace=True)
                if crs:
                    gdf = gdf.to_crs(crs)
            gdfs.append(gdf)
        except Exception as e:
            print(f"Error processing {session_dir}: {e}")
            continue

    return gdfs


def merge_geojson_files(session_locations, dest, crs=None):
    """
    Merge GeoJSON files from different session locations.

    Args:
        session_locations (list): List of session locations containing GeoJSON files.
        dest (str): Path to the location where the merged GeoJSON file will be saved.

    Returns:
        merged_config (GeoDataFrame): Merged GeoDataFrame containing the merged GeoJSON data.

    """
    filepaths = [
        os.path.join(location, "config_gdf.geojson") for location in session_locations
    ]
    gdfs = read_geojson_files(filepaths, crs=crs)
    merged_gdf = gpd.GeoDataFrame(concatenate_gdfs(gdfs), geometry="geometry")

    # Filter the geodataframe to only elements that intersect with the rois (dramatically drops the size of the geodataframe)
    merged_config = filter_and_join_gdfs(merged_gdf, "roi", predicate="intersects")
    # apply a group by operation to combine the rows with the same type and geometry into a single row
    merged_config = aggregate_gdf(merged_config, ["type", "geometry"])
    # applying the group by function in aggregate_gdf() turns the geodataframe into a dataframe
    merged_config = gpd.GeoDataFrame(merged_config, geometry="geometry")

    if os.path.isdir(dest):
        output_path = os.path.join(dest, "merged_config.geojson")
        merged_config.to_file(output_path, driver="GeoJSON")
    else:
        raise ValueError(f"Output directory {dest} does not exist.")

    return merged_config


def create_csv_per_transect(
    save_path: str,
    cross_distance_transects: dict,
    extracted_shorelines_dict: dict,
    roi_id: str = None,  # ROI ID is now optional and defaults to None
    filename_suffix: str = "_timeseries_raw.csv",
):
    """
    Create a CSV file for each transect containing time-series data.

    Args:
        save_path (str): The directory path where the CSV files will be saved.
        cross_distance_transects (dict): A dictionary containing cross-distance transects.
        extracted_shorelines_dict (dict): A dictionary containing extracted shorelines data.
        roi_id (str, optional): The ROI ID. Defaults to None.
        filename_suffix (str, optional): The suffix to be added to the CSV filenames. Defaults to "_timeseries_raw.csv".
    """
    for key, distances in cross_distance_transects.items():
        # Initialize the dictionary for DataFrame with mandatory keys
        data_dict = {
            "dates": extracted_shorelines_dict["dates"],
            "satname": extracted_shorelines_dict["satname"],
            key: distances,
        }

        # Add roi_id to the dictionary if provided
        if roi_id is not None:
            data_dict["roi_id"] = [roi_id] * len(extracted_shorelines_dict["dates"])

        # Create a DataFrame directly with the data dictionary
        df = pd.DataFrame(data_dict).set_index("dates")

        # Construct the full file path
        csv_filename = f"{key}{filename_suffix}"
        fn = os.path.join(save_path, csv_filename)

        # Save to CSV file, 'mode' set to 'w' for overwriting
        try:
            df.to_csv(fn, sep=",", mode="w")
            print(f"Time-series for transect {key} saved to {fn}")
        except Exception as e:
            print(f"Failed to save time-series for transect {key}: {e}")


def merge_and_average(df1: gpd.GeoDataFrame, df2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merge two GeoDataFrames based on the 'satname' and 'date' columns, and average the common numeric columns.

    Args:
        df1 (gpd.GeoDataFrame): The first GeoDataFrame.
        df2 (gpd.GeoDataFrame): The second GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: The merged GeoDataFrame with averaged numeric columns.
    """
    # Perform a full outer join
    merged = pd.merge(
        df1, df2, on=["satname", "date"], how="outer", suffixes=("_df1", "_df2")
    )

    # Identify numeric columns from both dataframes
    numeric_columns_df1 = df1.select_dtypes(include="number").columns
    numeric_columns_df2 = df2.select_dtypes(include="number").columns
    common_numeric_columns = set(numeric_columns_df1).intersection(numeric_columns_df2)

    # Average the numeric columns
    for column in common_numeric_columns:
        merged[column] = merged[[f"{column}_df1", f"{column}_df2"]].mean(axis=1)

    # Drop the original numeric columns
    merged.drop(
        columns=[f"{column}_df1" for column in common_numeric_columns]
        + [f"{column}_df2" for column in common_numeric_columns],
        inplace=True,
    )

    # Merge geometries
    geometry_columns = [col for col in merged.columns if "geometry" in col]
    merged = merge_geometries(merged, columns=geometry_columns)

    return merged
