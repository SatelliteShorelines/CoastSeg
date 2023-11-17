from collections import defaultdict
import os
from typing import List, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, MultiPoint
from shapely.ops import unary_union

from coastseg import geodata_processing

# from coastseg.file_utilities import to_file
# from coastseg.common import get_cross_distance_df
# from coastseg.common import convert_linestrings_to_multipoints, stringify_datetime_columns
# from coastsat import SDS_transects


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


from geopandas import GeoDataFrame
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point


def convert_lines_to_multipoints(gdf: GeoDataFrame) -> GeoDataFrame:
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


def merge_geodataframes(
    on, how="inner", aggregation_funcs=None, crs="epsg:4326", *gdfs
):
    """
    Merges multiple GeoDataFrames based on a common column.

    Parameters:
    on : str or list of str
        Column name or list of column names to merge on.
    how : str, optional
        Type of merge to be performed (default is 'inner').
    aggregation_funcs : dict, optional
        Dictionary of column names to aggregation functions.
        Example: for the columns 'cloud_cover' and 'geoaccuracy', the mean aggregation function can be specified as:
        aggregation_funcs = {
            'cloud_cover': 'mean',
            'geoaccuracy': 'mean'
        }
    *gdfs : GeoDataFrames
        Variable number of GeoDataFrames to be merged.

    Returns:
    GeoDataFrame
        The merged GeoDataFrame with aggregated columns as specified.
    """
    if len(gdfs) < 2:
        raise ValueError("At least two GeoDataFrames must be provided for merging")

    # Set default aggregation functions if none are provided
    if aggregation_funcs is None:
        aggregation_funcs = {}

    # Perform the merge while applying the custom aggregation functions
    merged_gdf = gdfs[0]
    merged_gdf.set_crs(crs)
    for gdf in gdfs[1:]:
        merged_gdf = pd.merge(
            merged_gdf, gdf, on=on, how=how, suffixes=("_left", "_right")
        )

        # Apply aggregation functions
        for col, func in aggregation_funcs.items():
            col_left = f"{col}_left"
            col_right = f"{col}_right"

            # Check if the columns exist in both GeoDataFrames
            if col_left in merged_gdf.columns and col_right in merged_gdf.columns:
                # Apply the aggregation function and drop the original columns
                merged_gdf[col] = merged_gdf[[col_left, col_right]].agg(func, axis=1)
                merged_gdf = merged_gdf.drop(columns=[col_left, col_right])

    return merged_gdf


def read_first_geojson_file(
    directory: str,
    filenames=["extracted_shorelines_lines.geojson", "extracted_shorelines.geojson"],
):
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
            clipped_gdf.plot()
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
        return gpd.GeoDataFrame()
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

    # # Loop over each pair of rows in gdf
    # for i in range(len(gdf)):
    #     for j in range(i+1, len(gdf)):
    #         # Check for intersection
    #         if gdf.iloc[i].geometry.intersects(gdf.iloc[j].geometry):
    #             # Calculate the intersection
    #             intersection = gdf.iloc[i].geometry.intersection(gdf.iloc[j].geometry)

    #             # Create a new row with the intersection and append to the result list
    #             overlap_list.append({'geometry': intersection})

    # # Create a DataFrame from the results list
    # overlap_df = pd.DataFrame(overlap_list)

    # # Convert the result DataFrame to a GeoDataFrame and set the CRS
    # overlap_gdf = gpd.GeoDataFrame(overlap_df, geometry='geometry', crs=gdf.crs)

    return overlap_gdf


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


def average_columns(df, col1, col2, new_col):
    df[new_col] = df[[col1, col2]].mean(axis=1, skipna=True)
    return df


def combine_dataframes(df1, df2, join_columns):
    # Perform an outer join and mark the origin of each row
    all_rows = pd.merge(df1, df2, on=join_columns, how="outer", indicator=True)

    # Keep only the rows that are in 'df1' but not in 'df2'
    df1_unique = all_rows[all_rows["_merge"] == "left_only"]
    if "cloud_cover_x" in df1_unique.columns and "cloud_cover_y" in df1_unique.columns:
        df1_unique = average_columns(
            df1_unique, "cloud_cover_x", "cloud_cover_y", "cloud_cover"
        )
        df1_unique.drop(columns=["cloud_cover_x", "cloud_cover_y"], inplace=True)
    if "geoaccuracy_x" in df1_unique.columns and "geoaccuracy_y" in df1_unique.columns:
        df1_unique = average_columns(
            df1_unique, "geoaccuracy_x", "geoaccuracy_y", "geoaccuracy"
        )
        df1_unique.drop(columns=["geoaccuracy_x", "geoaccuracy_y"], inplace=True)
    df1_unique.drop(columns=["_merge"], inplace=True)

    # Concatenate 'df2' and the unique rows from 'df1'
    result = pd.concat([df2, df1_unique], ignore_index=True)

    def assign_geometry(row):
        if pd.isnull(row["geometry"]):
            if pd.notnull(row["geometry_x"]):
                return row["geometry_x"]
            elif pd.notnull(row["geometry_y"]):
                return row["geometry_y"]
        else:
            return row["geometry"]

    if "geometry_x" in result.columns and "geometry_y" in result.columns:
        result["geometry"] = result.apply(assign_geometry, axis=1)
        result.drop(columns=["geometry_x", "geometry_y"], inplace=True)
    return result


def combine_geodataframes(gdf1, gdf2, join_columns, average_columns=None):
    """
    Combines two GeoDataFrames, performing an outer join and averaging specified numerical columns.

    Parameters:
    gdf1, gdf2 : GeoDataFrame
        The GeoDataFrames to combine.
    join_columns : list of str
        The columns to join on.
    average_columns : list of str, optional
        The columns to average. If None, all numerical columns with the same name in both GeoDataFrames will be averaged.

    Returns:
    GeoDataFrame
        The combined GeoDataFrame.
    """
    # Ensure that the 'geometry' column is present in both GeoDataFrames
    if "geometry" not in gdf1.columns or "geometry" not in gdf2.columns:
        raise ValueError("Both GeoDataFrames must have a 'geometry' column.")

    # Combine GeoDataFrames using an outer join
    combined_gdf = pd.merge(
        gdf1, gdf2, on=join_columns, how="outer", suffixes=("_gdf1", "_gdf2")
    )

    if average_columns is None:
        # List of numerical columns to be averaged
        average_columns = [
            col
            for col in gdf1.columns
            if col in gdf2.columns
            and col not in join_columns + ["geometry"]
            and np.issubdtype(gdf1[col].dtype, np.number)
            and np.issubdtype(gdf2[col].dtype, np.number)
        ]

    # Average specified numerical columns
    for col in average_columns:
        if (
            f"{col}_gdf1" in combined_gdf.columns
            and f"{col}_gdf2" in combined_gdf.columns
        ):
            combined_gdf[col] = combined_gdf[[f"{col}_gdf1", f"{col}_gdf2"]].mean(
                axis=1
            )
            combined_gdf.drop(columns=[f"{col}_gdf1", f"{col}_gdf2"], inplace=True)

    # Resolve geometry conflicts by prioritizing non-null values
    combined_gdf["geometry"] = combined_gdf["geometry_gdf1"].combine_first(
        combined_gdf["geometry_gdf2"]
    )
    combined_gdf.drop(columns=["geometry_gdf1", "geometry_gdf2"], inplace=True)

    return gpd.GeoDataFrame(combined_gdf, geometry="geometry")


def mergeRightUnique(
    left_df: gpd.GeoDataFrame,
    right_df: gpd.GeoDataFrame,
    join_columns: Union[str, List[str]] = ["date", "satname"],
    CRS: str = "EPSG:4326",
) -> pd.DataFrame:
    """
    Merges two GeoDataFrames, keeping only the unique rows from the right GeoDataFrame based on the specified join columns.

    Parameters:
    left_df : GeoDataFrame
        The left GeoDataFrame to merge. Its CRS is set to the specified CRS if not already set.
    right_df : GeoDataFrame
        The right GeoDataFrame to merge. Its CRS is set to the specified CRS if not already set.
    join_columns : str or list of str, default ['date', 'satname']
        The columns to join on. These columns are set as the index for both GeoDataFrames. If a string is passed, it is converted to a list.
    CRS : str, default 'EPSG:4326'
        The Coordinate Reference System to set for the GeoDataFrames if not already set.

    Returns:
    GeoDataFrame
        The merged GeoDataFrame, containing all rows from the left GeoDataFrame and only the unique rows from the right GeoDataFrame based on the join columns.
    """
    if not left_df.crs:
        left_df.set_crs(CRS, inplace=True)
    if not right_df.crs:
        right_df.set_crs(CRS, inplace=True)

    if isinstance(join_columns, str):
        join_columns = [join_columns]
    # Ensure that join are set as the index for both DataFrames
    left_df.set_index(join_columns, inplace=True)
    right_df.set_index(join_columns, inplace=True)

    # Find the difference in the MultiIndex between right_df and merged_gdf
    unique_indices = right_df.index.difference(merged_gdf.index)

    # Select only those rows from right_df that have unique indices
    unique_to_right_df = right_df.loc[unique_indices]
    if unique_to_right_df.crs:
        unique_to_right_df.crs = right_df.crs

    # Now concatenate the merged_gdf with the unique_to_right_df
    combined_gdf = pd.concat(
        [merged_gdf.reset_index(), unique_to_right_df.reset_index()], ignore_index=True
    )
    return combined_gdf


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

    merged_gdf["geometry"] = merged_gdf[columns].apply(
        lambda row: operation(row.tolist()), axis=1
    )
    for col in columns:
        if col in merged_gdf.columns:
            merged_gdf = merged_gdf.drop(columns=col)
    return merged_gdf


def merge_geojson_files(
    *file_paths: str,
) -> gpd.GeoDataFrame:
    """
    Merges any number of GeoJSON files into a single GeoDataFrame, removing any duplicate rows.

    Parameters:
    - *file_paths (str): Paths to the GeoJSON files.

    Returns:
    - GeoDataFrame: A GeoDataFrame containing the merged data from all input files, with duplicates removed.
    """
    merged_gdf = gpd.GeoDataFrame()
    for filepath in file_paths:
        gdf = geodata_processing.read_gpd_file(filepath)
        # Merging the two dataframes
        merged_gdf = gpd.GeoDataFrame(pd.concat([merged_gdf, gdf], ignore_index=True))

    # Dropping any duplicated rows based on all columns
    merged_gdf_cleaned = merged_gdf.drop_duplicates()
    return merged_gdf_cleaned


def create_csv_per_transect(
    save_path: str,
    cross_distance_transects: dict,
    extracted_shorelines_dict: dict,
    roi_id: str = None,  # ROI ID is now optional and defaults to None
    filename_suffix: str = "_timeseries_raw.csv",
):
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


# better way of mergine multiple gdfs together
# from shapely.ops import unary_union
# from coastseg.merge_utils import merge_geometries
# from functools import reduce
# import pandas as pd


# def merge_geometries(merged_gdf, columns=None, operation=unary_union):
#     """
#     Performs a specified operation for the geometries with the same date and satname.

#     Parameters:
#     merged_gdf : GeoDataFrame
#         The GeoDataFrame to perform the operation on.
#     columns : list of str, optional
#         The columns to perform the operation on. If None, all columns with 'geometry' in the name are used.
#     operation : function, optional
#         The operation to perform. If None, unary_union is used.

#     Returns:
#     GeoDataFrame
#         The GeoDataFrame with the operation performed.
#     """
#     if columns is None:
#         columns = [col for col in merged_gdf.columns if "geometry" in col]
#     else:
#         columns = [col for col in columns if col in merged_gdf.columns]

#     merged_gdf["geometry"] = merged_gdf[columns].apply(
#         lambda row: operation(row.tolist()), axis=1
#     )
#     for col in columns:
#         if col in merged_gdf.columns:
#             merged_gdf = merged_gdf.drop(columns=col)
#     return merged_gdf

# def merge_and_average(df1, df2):
#     # Perform a full outer join
#     merged = pd.merge(df1, df2, on=['satname', 'date'], how='outer', suffixes=('_df1', '_df2'))

#     # Loop over all columns
#     for column in set(df1.columns).intersection(df2.columns):
#         # Merge the geometries

#         if isinstance(df1[column].dtype, gpd.array.GeometryDtype):
#             print(f"merging {{['{column}_df1', '{column}_df2']}}")
#             print(df1[column])
#             print(df2[column])
#             # merged = merge_geometries(merged, columns=[f'{column}_df1', f'{column}_df2'], operation=unary_union)
#             merged = merge_geometries(merged)
#             continue
#         # Skip non-numeric columns
#         if not pd.api.types.is_numeric_dtype(df1[column]):
#             continue
#         # Average the values in the two columns
#         merged[column] = merged[[f'{column}_df1', f'{column}_df2']].mean(axis=1)
#         merged.drop(columns=[f'{column}_df1', f'{column}_df2'], inplace=True)

#     return merged

# # List of GeoDataFrames
# gdfs = [extracted_gdf1, extracted_gdf2, extracted_gdf3]

# # Perform a full outer join and average the numeric columns across all GeoDataFrames
# result = reduce(merge_and_average, gdfs)

# result
