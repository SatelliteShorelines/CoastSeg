import os
import geopandas as gpd
import pandas as pd
import glob
from shapely.geometry import Polygon
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def read_geojson(file):
    try:
        return gpd.read_file(file)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        return gpd.GeoDataFrame()

def calculate_bbox(gdf, file):
    if gdf.empty:
        return None
    bbox = gdf.total_bounds
    geometry = [Polygon([(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1])])]
    bbox_gdf = gpd.GeoDataFrame({'geometry': geometry})
    bbox_gdf['shoreline_id'] = os.path.basename(file)
    return bbox_gdf

def parallel_read_geojson_files_bbox(folder_path, num_workers=None, chunk_size=100):
    # Get a list of all GeoJSON files in the folder
    files = sorted(glob.glob(os.path.join(folder_path, "*")))
    geojson_files = [os.path.join(folder_path, file) for file in files if file.endswith(".geojson")]

    all_chunks = []
    # i=1100
    # print(f"Processing files {i} to {i + chunk_size}")
    # chunk_files = geojson_files[i:i + chunk_size]

    # # Read GeoJSON files in parallel using multiple workers (I/O-bound task)
    # with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     geojson_gdfs = list(executor.map(read_geojson, chunk_files))
    # # Calculate bounding boxes in parallel using multiple workers (CPU-bound task)
    # with ProcessPoolExecutor(max_workers=num_workers) as executor:
    #     bbox_gdfs = list(executor.map(calculate_bbox, geojson_gdfs, chunk_files))
    # bbox_gdfs = [result for result in bbox_gdfs if result is not None]
    # # Concatenate the list of GeoDataFrames
    # combined_df = gpd.GeoDataFrame(pd.concat(bbox_gdfs, ignore_index=True), crs=bbox_gdfs[0].crs)
    # # Save the chunk to a file
    # save_path = f"C:\\1_USGS\\1_CoastSeg\\1_official_CoastSeg_repo\\CoastSeg\\world_refernce_shorelines_bbox_{i}_{i + chunk_size}.geojson"
    # combined_df.to_file(save_path, driver="GeoJSON")

    # all_chunks.append(save_path)

    # for i in range(0, len(geojson_files), chunk_size):
    #     if i == 1100:
    #         continue
    #     save_path = f"C:\\1_USGS\\1_CoastSeg\\1_official_CoastSeg_repo\\CoastSeg\\world_refernce_shorelines_bbox_{i}_{i + chunk_size}.geojson"
    #     all_chunks.append(save_path)

    # return all_chunks

    for i in range(0, len(geojson_files), chunk_size):

        print(f"Processing files {i} to {i + chunk_size}")
        chunk_files = geojson_files[i:i + chunk_size]

        # Read GeoJSON files in parallel using multiple workers (I/O-bound task)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            geojson_gdfs = list(executor.map(read_geojson, chunk_files))

        # Calculate bounding boxes in parallel using multiple workers (CPU-bound task)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            bbox_gdfs = list(executor.map(calculate_bbox, geojson_gdfs, chunk_files))

        bbox_gdfs = [result for result in bbox_gdfs if result is not None]
        # Concatenate the list of GeoDataFrames
        combined_df = gpd.GeoDataFrame(pd.concat(bbox_gdfs, ignore_index=True), crs=bbox_gdfs[0].crs)

        # Save the chunk to a file
        save_path = f"C:\\1_USGS\\1_CoastSeg\\1_official_CoastSeg_repo\\CoastSeg\\world_refernce_shorelines_bbox_{i}_{i + chunk_size}.geojson"
        combined_df.to_file(save_path, driver="GeoJSON")
        all_chunks.append(save_path)

    return all_chunks

def main():
    folder_path = r"C:\1_USGS\1_CoastSeg\1_official_CoastSeg_repo\CoastSeg\downloaded_shorelines"
    num_workers = os.cpu_count()  # Adjust this value to control the number of parallel workers

    all_chunks = parallel_read_geojson_files_bbox(folder_path, num_workers)

    # Read back all the saved GeoJSON files and concatenate them into a single GeoDataFrame
    all_gdfs = [gpd.read_file(chunk) for chunk in all_chunks]
    final_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=all_gdfs[0].crs)

    # Save the final GeoDataFrame to a file
    final_save_path = r"C:\1_USGS\1_CoastSeg\1_official_CoastSeg_repo\CoastSeg\world_refernce_shorelines_bbox_combined_2.geojson"
    final_gdf.to_file(final_save_path, driver="GeoJSON")

if __name__ == '__main__':
    main()
